"""Offline stratified sampling and public/private input separation."""
import collections
import json
import math
from pathlib import Path
import random
import sys

import tifffile
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from CityNav.adapter import load_episodes, render_initial

DATA=Path('C:/Levir/2DNav/2DNav-project/datasets/CityNav/data')
OUT=Path('C:/Levir/2DNav/2DNav-project/out/23_citynav500_decision35/20261103')


def key(row):
    return row['area'],str(row['block']),tuple(row['object_ids']),tuple(row['ann_ids'])


def write_lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')


def main():
    annotation=DATA/'refined_label/processed_citynav/citynav_test_unseen.json'
    rows=json.loads(annotation.read_text(encoding='utf-8'))
    labels=collections.defaultdict(set)
    for level in ['easy','medium','hard']:
        for row in json.loads((DATA/f'original_label/citynav/citynav_test_unseen_{level}.json').read_text(encoding='utf-8')):
            labels[key(row)].add(level)
    strata=collections.defaultdict(list)
    for i,row in enumerate(rows):
        assert len(labels[key(row)])==1
        strata[next(iter(labels[key(row)]))].append(i)
    quotas={level:len(values)*500/len(rows) for level,values in strata.items()}
    allocation={level:math.floor(value) for level,value in quotas.items()}
    for level in sorted(quotas,key=lambda k:(-(quotas[k]-allocation[k]),k))[:500-sum(allocation.values())]:
        allocation[level]+=1
    rng=random.Random(20261103)
    selected=[]
    for level in sorted(strata):
        selected.extend(rng.sample(strata[level],allocation[level]))
    rng.shuffle(selected)
    assert len(set(selected))==500
    manifest=OUT/'sample_manifest.json'
    if manifest.exists():raise RuntimeError('Sample already frozen')
    manifest.write_text(json.dumps(dict(seed=20261103,population=len(rows),strata_counts={k:len(v) for k,v in strata.items()},
        allocation=allocation,selected_source_rows=selected,difficulty_mapping='area/block/object_ids/ann_ids matched to original difficulty splits',
        annotation=str(annotation),selection='largest-remainder proportional allocation; random sample without replacement; no performance filtering'),indent=2),encoding='utf-8')
    objects=DATA/'refined_label/cityrefer/objects.json'
    landmarks=DATA/'refined_label/cityrefer/landmarks.json'
    all_objects=json.loads(objects.read_text(encoding='utf-8'))
    entries={private['source_row']:(eid,agent,private,context) for eid,agent,private,context in load_episodes(annotation,objects,landmarks) if private['source_row'] in set(selected)}
    inputs=[];public=[];private=[];calibrations={}
    for index,i in enumerate(selected):
        eid,agent,gt,context=entries[i];name=context['map_name'];image=DATA/'rgbd'/(name+'.png')
        if name not in calibrations:
            with tifffile.TiffFile(image.with_suffix('.tif')) as tif:
                tags=tif.pages[0].tags;sx,sy,_=tags['ModelPixelScaleTag'].value;ix,iy,_,wx,wy,_=tags['ModelTiepointTag'].value
            Image.MAX_IMAGE_PIXELS=None
            with Image.open(image) as im:size=list(im.size)
            calibrations[name]=dict(image_path=str(image),image_size=size,world_to_raster=[[1/sx,0,ix-wx/sx],[0,-1/sy,iy+wy/sy],[0,0,1]],main_width_m=100.,main_height_m=100.)
        view=render_initial(context,DATA/'rgbd',OUT/'initialization/images'/(eid+'.png'))
        init_agent=dict(agent,views=[view]);inputs.append(dict(episode_id=eid,agent_input=init_agent))
        public.append(dict(episode_id=eid,agent_input=agent,environment=calibrations[name]))
        obj=all_objects[name][str(rows[i]['object_ids'][0])]
        private.append(dict(gt,goal_polygon_xy_m=[p[:2] for p in obj.get('contour',[])],difficulty=next(iter(labels[key(rows[i])]))))
        if (index+1)%50==0:print('prepared',index+1,flush=True)
    write_lines(OUT/'initialization/agent_inputs.jsonl',inputs)
    write_lines(OUT/'citynav/public_uninitialized.jsonl',public)
    write_lines(OUT/'citynav/evaluator_only.jsonl',private)
    print('allocation',allocation,flush=True)


if __name__=='__main__':main()
