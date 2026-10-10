"""Observed-only, pose-aligned evidence for map experiments G0--G4.

No navigation policy, ground-truth fields, model calls, or full-map imagery.
Geometry mirrors the existing clipped crop, including its degree-plane rotation.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


MODES = ('off', 'g0', 'g1', 'g2', 'g3', 'g4')

SPATIAL_RULES = ('Use the spatial overview and supplied geometric relations to interpret where the observations lie. '
    'Instruction direction is referenced to the current stage anchor; crop orientation and observation-center displacement are different. '
    'Evaluate forward/left/right using the supplied map geometry rather than assuming the upper half of every preview is ahead of the agent. '
    'Direction supports interpretation; the complete destination description remains the main matching criterion. '
    'Localize only on an explicitly permitted original view; overview, historical and SAM3 images are auxiliary evidence.')
MULTIVIEW_RULES = ('Propose up to 2 distinct plausible candidates across the permitted current search views. '
    "Return the actual source_view_id and a bbox in that view's pixel coordinates. "
    'The same object seen in overlapping views is one alternative, not two. Do not invent candidates to fill the budget.')
HISTORY_RULES = ('Retrieved search records are earlier model assessments for the stated stage and target, not ground truth. '
    'A previous failure to find a target does not prove absence, especially at a coarser resolution. '
    'Use the linked observations as evidence; current visual evidence takes priority. '
    'Do not localize on historical images or treat preview centers as visited agent positions.')


def transform(points, matrix):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(matrix).T
    if np.any(np.abs(q[:, 2]) < 1e-10):
        raise ValueError('Singular spatial projection')
    return q[:, :2] / q[:, 2:]


def direction(heading):
    t = math.radians(heading)
    return np.array([math.sin(t), math.cos(t)])


class SpatialMap:
    def __init__(self, origin_latlng, gps_bottom_left, gps_top_right, map_size, output_dir, mode='g0'):
        if mode not in MODES or mode == 'off':
            raise ValueError('SpatialMap requires an enabled map mode')
        self.mode = mode
        self.origin = np.asarray(origin_latlng, dtype=float)
        self.low = np.asarray(gps_bottom_left, dtype=float)
        self.high = np.asarray(gps_top_right, dtype=float)
        self.height, self.width = map(int, map_size[:2])
        if min(self.height, self.width) < 2 or np.any(self.high <= self.low):
            raise ValueError('Invalid map geometry')
        self.k = np.array([111320., 111320. * math.cos(math.radians(self.origin[0]))])
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.views = {}
        self.images = {}
        self.events = []
        self.path = []
        self.stage = None
        self.agent = None

    def to_xy(self, latlng):
        return ((np.asarray(latlng) - self.origin) * self.k)[..., ::-1]

    def to_latlng(self, xy):
        return np.asarray(xy)[..., ::-1] / self.k + self.origin

    def begin_stage(self, stage_id, target, anchor, heading):
        self.stage = dict(stage_id=stage_id, target=target, anchor_xy=self.to_xy(anchor).tolist(),
                          instruction_heading_deg=float(heading))
        self.move(anchor, heading, 'stage_start')

    def move(self, position, heading, reason):
        xy = self.to_xy(position).tolist()
        self.agent = dict(position_xy=xy, heading_deg=float(heading))
        if not self.path or np.linalg.norm(np.asarray(self.path[-1]['xy'])-xy) > 1e-5:
            self.path.append(dict(xy=xy, stage_id=self.stage['stage_id'], reason=reason))

    def observe(self, view_id, image, corners, center, heading, round_n,
                role='main', acquisition='actual', localizable=False, description=''):
        if image is None:
            return None
        if view_id in self.views:
            raise ValueError('View IDs must be unique across stages and rounds')
        h, w = image.shape[:2]
        requested = np.asarray(corners, dtype=float)
        src = np.column_stack(((requested[:, 1]-self.low[1])/(self.high[1]-self.low[1])*self.width,
                               (self.high[0]-requested[:, 0])/(self.high[0]-self.low[0])*self.height))
        src = src.astype(np.float32)
        clipped = src.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, self.width-1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, self.height-1)
        if cv2.contourArea(clipped) < 1:
            raise ValueError('Degenerate clipped observation')
        # Geographic reconstruction uses float64 to avoid rounding near small footprints.
        effective = np.column_stack((self.high[0]-clipped[:, 1].astype(float)/self.height*(self.high[0]-self.low[0]),
                                     self.low[1]+clipped[:, 0].astype(float)/self.width*(self.high[1]-self.low[1])))
        footprint = self.to_xy(effective)
        pixels = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        image_to_meters=np.array([[(self.high[1]-self.low[1])/self.width*self.k[1],0,(self.low[1]-self.origin[1])*self.k[1]],
                                 [0,-(self.high[0]-self.low[0])/self.height*self.k[0],(self.high[0]-self.origin[0])*self.k[0]],
                                 [0,0,1]],dtype=float)
        matrix = image_to_meters @ cv2.getPerspectiveTransform(pixels, clipped)
        np.linalg.inv(matrix)  # Fail explicitly on a collapsed border crop.
        poly = Polygon(footprint)
        if not poly.is_valid or poly.area <= 0:
            raise ValueError('Invalid observation footprint')
        old = self.coverage()
        top = transform([[w/2, 0], [w/2, h/2]], matrix)
        delta = top[0]-top[1]
        bearing = math.degrees(math.atan2(delta[0], delta[1])) % 360
        record = dict(view_id=view_id, stage_id=self.stage['stage_id'], round=round_n, role=role,
            acquisition=acquisition, localizable=bool(localizable), description=description,
            center_xy=self.to_xy(center).tolist(), crop_heading_deg=float(heading),
            top_bearing_deg=bearing, requested_corners_latlng=requested.tolist(),
            footprint_xy=footprint.tolist(), pixel_to_map=matrix.tolist(), image_size=[w, h],
            clipped=bool(np.any(clipped != src)), area_m2=poly.area,
            new_area_m2=poly.difference(old).area, resolution_m_per_pixel=math.sqrt(poly.area/(w*h)))
        suffix=view_id.rsplit(':',1)[-1]
        record['map_label']='S%sR%d%s'%(self.stage['stage_id'].rsplit('_step',1)[-1],round_n,
            {'main':'C','left':'L','forward':'F','right':'R','detail':'D','context':'W'}.get(suffix,suffix))
        filename=view_id.replace(':', '_')+'.png'
        if not cv2.imwrite(str(self.output/filename), image):
            raise OSError('Cannot save observation')
        record['image_path']=str((self.output/filename).resolve())
        self.views[view_id]=record
        self.images[view_id]=image.copy()
        return record

    def coverage(self, view_ids=None):
        ids = self.views if view_ids is None else view_ids
        return unary_union([Polygon(self.views[i]['footprint_xy']) for i in ids])

    def locate(self, view_id, bbox):
        x1, y1, x2, y2 = bbox
        view = self.views[view_id]
        center = transform([[(x1+x2)/2, (y1+y2)/2]], view['pixel_to_map'])[0]
        polygon = transform([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], view['pixel_to_map'])
        return dict(map_center=self.to_latlng(center).tolist(), polygon_xy=polygon.tolist(),
                    **self.relative(center))

    def relative(self, xy):
        heading=self.stage['instruction_heading_deg']
        f=direction(heading);right=direction(heading+90)
        delta=np.asarray(xy)-self.stage['anchor_xy']
        current=np.asarray(xy)-self.agent['position_xy']
        return dict(stage_forward_m=round(float(delta@f),1),stage_right_m=round(float(delta@right),1),
                    agent_forward_m=round(float(current@f),1),agent_right_m=round(float(current@right),1))

    def record_search(self, view_ids, result, round_n):
        proposals = result.get('candidates', [])
        # Transport/protocol failures are not model evidence that a target was absent.
        assessment = ('proposals' if proposals else 'operational_error'
                      if 'candidates' not in result or result.get('validation_errors')
                      else 'no_supported_candidate')
        self.events.append(dict(event_id='search_%04d'%len(self.events),stage_id=self.stage['stage_id'],
            target=self.stage['target'],round=round_n,view_ids=list(view_ids),
            result=assessment,
            proposals=[dict(source_view_id=p['source_view_id'], bbox_2d=p['bbox_2d'],
                            confidence=p.get('confidence'), supporting_evidence=p.get('supporting_evidence',''))
                       for p in proposals]))

    def retrieve(self, current_ids, round_n, budget=2):
        current=self.coverage(current_ids)
        agent=np.asarray(self.agent['position_xy'])
        candidates=[]
        for v in self.views.values():
            if v['stage_id'] != self.stage['stage_id'] or v['round'] >= round_n or v['role'] != 'main':
                continue
            poly=Polygon(v['footprint_xy'])
            novelty=poly.difference(current).area/max(poly.area,1)
            distance=np.linalg.norm(np.asarray(v['center_xy'])-agent)
            supported=any(e['result']=='proposals' and any(p['source_view_id']==v['view_id'] for p in e['proposals'])
                          for e in self.events if e['stage_id']==self.stage['stage_id'])
            score=2*float(supported)+novelty+0.2*v['round']-distance/1000
            candidates.append((score,v['view_id']))
        return [i for _,i in sorted(candidates,reverse=True)[:budget]]

    def summary(self, current_ids, historical_ids=()):
        rows=[]
        for i in list(current_ids)+list(historical_ids):
            v=self.views[i]
            rows.append(dict(view_id=i,map_label=v['map_label'],acquisition=v['acquisition'],
                historical=i in historical_ids,resolution_m_per_pixel=round(v['resolution_m_per_pixel'],2),
                image_top_bearing_deg=round(v['top_bearing_deg'],1),**self.relative(v['center_xy'])))
        relations=[]
        for n,a in enumerate(current_ids):
            for b in current_ids[n+1:]:
                p=Polygon(self.views[a]['footprint_xy']);q=Polygon(self.views[b]['footprint_xy'])
                intersection=p.intersection(q).area
                if intersection>0:
                    relations.append(dict(views=[a,b],overlap_fraction_of_smaller=round(intersection/min(p.area,q.area),2)))
        # No old bbox coordinates or unconstrained free-form summaries enter context.
        history=[dict(event_id=e['event_id'],round=e['round'],view_ids=[i for i in e['view_ids'] if i in historical_ids],
                      result=e['result']) for e in self.events
                 if e['stage_id']==self.stage['stage_id'] and any(i in historical_ids for i in e['view_ids'])]
        history=[e for e in history if e['result']!='operational_error']
        return dict(stage_direction_deg=self.stage['instruction_heading_deg'],
                    geometry_reference='stage anchor; meters; positive right and forward',
                    views=rows,overlaps=relations,retrieved_search_records=history)

    def preview_poses(self, main_id):
        v=self.views[main_id];heading=self.stage['instruction_heading_deg']
        f=direction(heading);r=direction(heading+90)
        quad=np.asarray(v['footprint_xy']);w=float(np.ptp(quad@r));h=float(np.ptp(quad@f))
        anchor=np.asarray(self.agent['position_xy'])
        for name,side in [('left',-0.35),('forward',0),('right',0.35)]:
            offset=side*w*r+0.4*h*f
            relative=math.degrees(math.atan2(side*w,0.4*h))
            yield name,self.to_latlng(anchor+offset).tolist(),heading,relative,float(np.linalg.norm(offset))

    def render(self, view_ids, path, long_side=1024):
        selected=[self.views[i] for i in dict.fromkeys(view_ids)]
        quads=[np.asarray(v['footprint_xy'])*[1,-1] for v in selected]
        points=np.vstack(quads+[np.asarray([self.agent['position_xy']])*[1,-1]])
        low,high=points.min(0),points.max(0)
        scale=(long_side-80)/max(float(np.max(high-low)),1)
        width,height=np.ceil((high-low)*scale+80).astype(int)
        canvas=np.full((height,width,3),235,np.uint8)
        union_mask=np.zeros((height,width),np.uint8)
        def canvas_xy(xy):return (np.asarray(xy)*[1,-1]-low)*scale+40
        # Fine observations take precedence. No pixels are fetched from the source TIFF.
        for v in sorted(selected,key=lambda v:v['resolution_m_per_pixel'],reverse=True):
            image=self.images[v['view_id']];h,w=image.shape[:2]
            target=canvas_xy(v['footprint_xy']).astype(np.float32)
            matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]),target)
            warped=cv2.warpPerspective(image,matrix,(width,height))
            mask=cv2.warpPerspective(np.full((h,w),255,np.uint8),matrix,(width,height),flags=cv2.INTER_NEAREST)
            canvas[mask>0]=warped[mask>0];union_mask=np.maximum(union_mask,mask)
        colors=[(40,160,40),(220,100,20),(20,160,210),(180,40,170)]
        for n,v in enumerate(selected):
            quad=canvas_xy(v['footprint_xy']).astype(np.int32);color=colors[n%len(colors)]
            cv2.polylines(canvas,[quad],True,color,2,cv2.LINE_AA)
            label=v['map_label']
            cv2.putText(canvas,label,tuple(quad[0]+[5,20]),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),3)
            cv2.putText(canvas,label,tuple(quad[0]+[5,20]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        if len(self.path)>1:
            cv2.polylines(canvas,[canvas_xy([p['xy'] for p in self.path]).astype(np.int32)],False,(0,210,240),2)
        pos=np.asarray(self.agent['position_xy']);start=tuple(canvas_xy(pos).astype(int))
        for bearing,color,length in [(self.stage['instruction_heading_deg'],(210,70,210),52),
                                     (self.agent['heading_deg'],(20,20,240),35)]:
            end=tuple((canvas_xy(pos)+direction(bearing)*[1,-1]*length).astype(int))
            cv2.arrowedLine(canvas,start,end,color,2,tipLength=.3)
        cv2.circle(canvas,start,4,(0,0,255),-1)
        cv2.putText(canvas,'N ^ | red: agent | purple: stage direction',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
        if not cv2.imwrite(str(path),canvas):raise OSError('Cannot save map')
        return canvas,union_mask

    def snapshot(self, name):
        value=dict(mode=self.mode,frame=dict(origin_latlng=self.origin.tolist(),axes='east_north_m'),
            stage=self.stage,agent=self.agent,views=self.views,search_events=self.events,actual_path=self.path)
        p=self.output/(name+'.json')
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        return str(p.resolve())

    def prepare_search(self, originals, anchor, heading, round_n, scale, ob, sam3_image=None, minimap=None):
        """Acquire fixed preview crops, retaining the exact original main/detail/context."""
        from .views import create_view_image, generate_view_corners_with_scale
        prefix=self.stage['stage_id']+':r%d:'%round_n
        inputs=[];main_ids=[];current_ids=[]
        for role,image,corners in originals:
            view_id=prefix+role
            record=self.observe(view_id,image,corners,anchor,heading,round_n,role=role,
                                localizable=role=='main')
            if record is None:continue
            current_ids.append(view_id)
            if role=='main':main_ids.append(view_id)
            labels={'main':'当前位置获取的搜索主图。','detail':'同位置、同朝向的细节视图。',
                    'context':'同位置、同朝向的广域视图。'}
            inputs.append(dict(view_id=view_id,role=role,image=image,corners=corners,
                path=record['image_path'],map_label=record['map_label'],localization_allowed=role=='main',
                description=labels[role]))
        if not main_ids:raise ValueError('Spatial search needs a valid main view')
        if self.mode in ('g2','g3','g4'):
            for name,center,crop_heading,offset,distance in self.preview_poses(main_ids[0]):
                corners=generate_view_corners_with_scale(center,ob,scale_factor=scale,angle_deg=crop_heading)
                image=create_view_image(corners,ob,out_px=768)
                if image is None:continue
                view_id=prefix+name
                label=('沿当前方向' if name=='forward' else ('左偏' if offset<0 else '右偏')+'%.1f度'%abs(offset))
                description='从当前位置%s、前进%.1fm处获取的视图裁剪。'%(label,distance)
                try:
                    record=self.observe(view_id,image,corners,center,crop_heading,round_n,
                        acquisition='preview',localizable=True,description=description)
                except ValueError:
                    continue
                main_ids.append(view_id);current_ids.append(view_id)
                inputs.append(dict(view_id=view_id,role='main',image=image,corners=corners,
                    path=record['image_path'],map_label=record['map_label'],localization_allowed=True,description=description))
        history=self.retrieve(main_ids,round_n) if self.mode=='g4' else []
        if self.mode in ('g1','g3','g4'):
            map_ids=current_ids+history
            map_path=self.output/(prefix.replace(':','_')+'overview.png')
            overview,_=self.render(map_ids,map_path)
            inputs.append(dict(view_id=prefix+'overview',role='overview',image=overview,
                path=str(map_path.resolve()),localization_allowed=False,
                description='已观察区域的北向空间地图，标签对应原始视图。'))
        elif self.mode=='g2' and minimap is not None:
            inputs.append(dict(view_id=prefix+'minimap',role='minimap',image=minimap,
                localization_allowed=False,description='原搜索历史地图。'))
        if sam3_image is not None:
            inputs.append(dict(view_id=prefix+'sam3',role='sam3',image=sam3_image,
                localization_allowed=False,description='当前主图 %s 的 SAM3 辅助标注。'%main_ids[0]))
        for i in history:
            v=self.views[i]
            inputs.append(dict(view_id=i,role='history',image=self.images[i],path=v['image_path'],
                map_label=v['map_label'],
                corners=v['requested_corners_latlng'],localization_allowed=False,
                description='第%d轮获取的历史视图。'%v['round']))
        return dict(mode=self.mode,views=inputs,main_ids=main_ids,
                    spatial_summary=self.summary(main_ids,history),historical_ids=history)


def build_spatial_request(system, destination, context, previous_grid=None):
    """Only approved additions plus mechanical replacement of single-source roles."""
    from util import data_url_from_image_array
    ids=context['main_ids'];first=ids[0]
    # Remove the singular-image assumptions while keeping the baseline analysis steps.
    system=system.replace('All candidate boxes use source_view_id="search_main".',
                          'All candidate boxes must name a permitted source_view_id.')
    system=system.replace('Propose up to 2 distinct plausible candidates on the current main search view. Do not invent candidates to fill the budget.',MULTIVIEW_RULES)
    system=system.replace('Main views have top = current heading. ',
                          'The input describes the pose of each view. ')
    system=system.replace('Detail/context images have the same center and heading as their main view, with narrower/wider fields of view.',
                          'Detail/context images have the same center and crop heading as the current-position main view.')
    system=system.replace('Localize only on the permitted main view;', 'Localize only on a permitted main view;')
    system=system.replace('"source_view_id":"search_main"','"source_view_id":"permitted view ID"')
    system=system.replace('search_main coordinates','the explicitly named original source view coordinates')
    system=system.replace('Target localization ON IMAGE 1 ONLY:', 'Target localization on the permitted original views:')
    system=system.replace('each proposed region ON IMAGE 1 ONLY.', 'each proposed region on its named original view.')
    system=system.replace("All bounding box coordinates MUST be given in Image 1's", "All bounding box coordinates MUST be given in the named source view's")
    system=system.replace('Propose candidate regions in Image 1 that satisfy', 'Propose candidate regions in the permitted views that satisfy')
    system=system.replace('but each proposed region must still come from Image 1.', 'and inspect the other permitted main views.')
    system=system.replace('Image 2, Image 3, or Image 4','detail, context, overview or historical images')
    system=system.replace('Image 2 and Image 3','the current-position detail and context views')
    system=system.replace('Image 4 (the minimap)','the supplied overview')
    # Baseline top-half preference refers only to the actual-position view, not previews.
    system=system.replace('Image 1','the current-position main view')
    system=system.replace('Images 2-4','the supplied auxiliary views').replace('Image 5','the current-main SAM3 overlay')
    if context['mode'] in ('g1','g3','g4'):
        system+='\n'+SPATIAL_RULES
    if context['mode']=='g4':system+='\n'+HISTORY_RULES
    content=[dict(type='text',text='Destination description: '+str(destination)+'\nPermitted localization views: '+
                  json.dumps(ids)+'\nReturn grid_5x5 only for the current-position main view '+first+'.')]
    if context['mode'] in ('g1','g3','g4'):
        content.append(dict(type='text',text='Spatial evidence: '+json.dumps(context['spatial_summary'],ensure_ascii=False)))
    for view in context['views']:
        allowed='可定位。' if view['localization_allowed'] else '辅助证据。'
        label=(' ['+view['map_label']+']') if view.get('map_label') else ''
        content.extend([dict(type='text',text=view['view_id']+label+'：'+view['description']+allowed),
            dict(type='image_url',image_url=dict(url=data_url_from_image_array(view['image'])))])
    if previous_grid:
        content.append(dict(type='text',text='Previous step grid_5x5 from the previous current-position main view '
            '(soft prior; its cells do not use the current image coordinates): '+json.dumps(previous_grid)))
    return system,content
