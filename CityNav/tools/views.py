"""Raster access uses public calibration only; no annotations or GT are loaded."""
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import homography, project, rect_corners, box_points, pixels, world, normalized, bearing

class Environment:
    def __init__(self, specification, output):
        self.spec = specification
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        old = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            self.image = Image.open(specification['image_path'])
        finally:
            Image.MAX_IMAGE_PIXELS = old
        if list(self.image.size) != specification['image_size']:
            raise ValueError('Raster dimensions changed since preparation')
        self.views = {}

    def close(self):
        self.image.close()

    def check_position(self, xy):
        if len(xy)!=2 or not np.isfinite(xy).all():
            raise ValueError('Position must be finite east/north XY')
        p = project(self.spec['world_to_raster'], [xy])[0]
        w,h = self.image.size
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            bounds=project(np.linalg.inv(self.spec['world_to_raster']),[[0,0],[w,0],[w,h],[0,h]])
            raise ValueError('Requested position '+str(list(map(float,xy)))+
                ' is outside map. Valid east/north map corners: '+str(bounds)+
                '. Recompute heading/step or choose a valid image point or public landmark; no movement occurred.')
        return list(map(float,xy))

    def render(self, corners, view_id, source, parent_id=None):
        n = 768
        uv = box_points([0,0,n-1,n-1])
        raster = project(self.spec['world_to_raster'], corners)
        to_source = homography(uv, raster)
        # Decode only one source map per episode; crop before RGB conversion.
        a = np.asarray(raster)
        w,h = self.image.size
        left,top = max(0,int(np.floor(a[:,0].min()))-3), max(0,int(np.floor(a[:,1].min()))-3)
        right,bottom = min(w,int(np.ceil(a[:,0].max()))+4), min(h,int(np.ceil(a[:,1].max()))+4)
        if right<=left or bottom<=top:
            raise ValueError('View has no intersection with map')
        window = self.image.crop((left,top,right,bottom)).convert('RGB')
        local = np.array([[1,0,-left],[0,1,-top],[0,0,1]]) @ to_source
        out = window.transform((n,n),Image.Transform.PERSPECTIVE,tuple(local.flatten()[:8]),
                               resample=Image.Resampling.BICUBIC)
        yy,xx = np.mgrid[:n,:n]
        coords = project(to_source, np.c_[xx.ravel(),yy.ravel()])
        coords = np.asarray(coords).reshape(n,n,2)
        valid = ((coords[:,:,0]>=0)&(coords[:,:,0]<w)&(coords[:,:,1]>=0)&(coords[:,:,1]<h))
        path = self.output/(view_id+'.png')
        mask = self.output/(view_id+'.valid.png')
        out.save(path)
        Image.fromarray(valid.astype('uint8')*255).save(mask)
        view = dict(view_id=view_id, image_path=str(path.resolve()),valid_mask_path=str(mask.resolve()),
                    image_size=[n,n],pixel_to_world=homography(uv,corners).tolist(),
                    world_corners_xy_m=corners,valid_fraction=float(valid.mean()),
                    image_up_heading_deg=bearing((np.asarray(corners[2])+corners[3])/2,
                                                 (np.asarray(corners[0])+corners[1])/2),
                    width_m=float(np.linalg.norm(np.asarray(corners[1])-corners[0])),
                    height_m=float(np.linalg.norm(np.asarray(corners[3])-corners[0])),
                    source=source,parent_id=parent_id,
                    valid_mask_scope='raster bounds only; holes and semantic visibility are not measured')
        self.views[view_id] = view
        return view

    def main(self, position, heading, view_id):
        self.check_position(position)
        corners=rect_corners(position,heading,self.spec['main_width_m'],self.spec['main_height_m'])
        for existing in self.views.values():
            if existing['source']=='main' and np.allclose(existing['world_corners_xy_m'],corners,rtol=0,atol=1e-9):
                return existing  # Same immutable raster/pose is the same observation.
        return self.render(corners,view_id,'main')

    def view(self, ident, allowed):
        if ident not in allowed or ident not in self.views:
            raise ValueError('view_id must refer to an image delivered to the model')
        return self.views[ident]

    def check_point(self, view, point):
        if len(point)!=2 or not np.isfinite(point).all() or not all(0<=x<=1000 for x in point):
            raise ValueError('Image point must be in [0,1000]')
        uv = pixels(view,[point])[0]
        with Image.open(view['valid_mask_path']) as mask:
            if not mask.getpixel(tuple(int(round(x)) for x in uv)):
                raise ValueError('Image point is in the invalid raster area')
        return self.check_position(world(view,[point])[0])

    def locate(self, view, box):
        if len(box)!=4 or not np.isfinite(box).all() or not all(0<=x<=1000 for x in box):
            raise ValueError('bbox must be finite [x1,y1,x2,y2] in [0,1000]')
        x1,y1,x2,y2 = box
        if x2<=x1 or y2<=y1:
            raise ValueError('bbox must have positive width and height')
        center = self.check_point(view,[(x1+x2)/2,(y1+y2)/2])
        for point in box_points(box):
            self.check_point(view,point)
        return dict(view_id=view['view_id'],bbox=box,
                    bbox_pixels=[x1*767/1000,y1*767/1000,x2*767/1000,y2*767/1000],
                    polygon_xy_m=world(view,box_points(box)),center_xy_m=center)

    def zoom(self, parent, box, view_id):
        # Every crop remains inside its observed parent, including map padding.
        self.locate(parent,box)
        return self.render(world(parent,box_points(box)),view_id,'zoom',parent['view_id'])

    def review(self, candidate, parent, view_id):
        # Build the crop in the parent MAIN view, even for a tool-view candidate.
        p = np.asarray(normalized(parent,candidate['polygon_xy_m']))
        lo,hi = p.min(axis=0),p.max(axis=0)
        if np.any(lo < -1e-5) or np.any(hi > 1000+1e-5):
            raise ValueError('Candidate is not within the parent main view')
        # Physical square; rectangular main footprints remain supported.
        width,height = parent['width_m'],parent['height_m']
        extent = (hi-lo)/1000*[width,height]
        side = max(2*max(extent), max(width,height)/4)
        sides = np.minimum([width,height],side)/[width,height]*1000
        center = (lo+hi)/2
        center = np.maximum(sides/2,np.minimum(1000-sides/2,center))
        box = [*(center-sides/2),*(center+sides/2)]
        return self.render(world(parent,box_points(box)),view_id,'mandatory_review',parent['view_id'])

    def mark_review(self, detail, candidate):
        """Mark only the unconfirmed proposal; keep raw RGB and exact view geometry."""
        from PIL import ImageDraw
        footprint=np.asarray(normalized(detail,candidate['polygon_xy_m']))
        box=[*footprint.min(axis=0).tolist(),*footprint.max(axis=0).tolist()]
        with Image.open(detail['image_path']) as source:
            image=source.convert('RGB')
        draw=ImageDraw.Draw(image)
        pixel_box=[v*767/1000 for v in box]
        draw.rectangle(pixel_box,outline='black',width=5)
        draw.rectangle(pixel_box,outline='yellow',width=2)
        draw.text((pixel_box[0]+3,max(0,pixel_box[1]-13)), 'UNCONFIRMED',fill='yellow',
                  stroke_width=1,stroke_fill='black')
        path=self.output/(detail['view_id']+'.marked.png')
        image.save(path)
        marked=dict(detail,image_path=str(path.resolve()),unmarked_image_path=detail['image_path'],
            proposal_bbox=box,annotation='Thin yellow rectangle marks the current UNCONFIRMED proposal; not a detection or proof')
        self.views[detail['view_id']]=marked
        return marked

    def tool_overlay(self, view, candidates, view_id):
        from PIL import ImageDraw
        with Image.open(view['image_path']) as source:
            image=source.convert('RGB')
        for c in candidates:
            self.locate(view,c['bbox'])
            if c.get('mask_path'):
                with Image.open(c['mask_path']) as mask:
                    if mask.size!=image.size:raise ValueError('SAM3 mask dimensions differ from source view')
                # Keep the service mask as an artifact, but never tint roof/wall
                # interiors in a model input used to judge object appearance.
            draw=ImageDraw.Draw(image)
            box=[v*767/1000 for v in c['bbox']]
            draw.rectangle(box,outline='black',width=3)
            draw.rectangle(box,outline='white',width=1)
            draw.text((box[0]+3,box[1]+3),str(c['candidate_id']),fill='white',stroke_width=1,stroke_fill='black')
        path=self.output/(view_id+'.png');image.convert('RGB').save(path)
        overlay=dict(view,view_id=view_id,image_path=str(path.resolve()),source='grounding_overlay',parent_id=view['view_id'],
            annotation='Monochrome candidate outlines and IDs only; object interiors are not color-filled. These are unconfirmed proposals.')
        self.views[view_id]=overlay
        return overlay

