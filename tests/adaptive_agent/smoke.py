"""Real local GPU proposals and Jev evidence on the frozen public observations."""
import importlib
from pathlib import Path
import re
import sys
import time
from experiment import ROOT,WORK,read,save


def main():
    sys.path.insert(0,str(WORK))
    batch=sys.argv[1] if len(sys.argv)>1 else 'visible_01'
    folder=ROOT/batch;plan=read(folder/'plan.json');dest=folder/'local_smoke'
    if dest.exists():raise RuntimeError('Keep original smoke; use a fresh frozen batch')
    dest.mkdir();rows=[]
    for index,job in enumerate(plan['jobs']):
        ds=job['dataset'];public=read(folder/job['public_input']);ep=public['episode']
        v=importlib.import_module(ds+'.tools.views');g=importlib.import_module(ds+'.tools.grounding_skill')
        j=importlib.import_module(ds+'.tools.jev_evidence')
        target=ep['initialization']['final_target']
        categories=[(m.start(),-len(c),c) for c in g.CATEGORIES for m in [re.search(r'\b'+re.escape(c)+r's?\b',target,re.I)] if m]
        category=min(categories)[2] if categories else 'building'
        out=dest/job['episode_id'];env=v.Environment(ep['environment'],out/'views')
        pose=public['checkpoint'] or ep['agent_input']['initial_pose']
        try:
            view=env.main(pose['xy_m'],pose['heading_deg'],'smoke-main');raw={}
            for name,cls,port in [('dino',g.GroundingDINO,8766),('sam3',g.SAM3,8765)]:
                tool=cls(f'http://127.0.0.1:{port}')
                if not tool.available():raise RuntimeError('Unavailable backend '+name)
                started=time.perf_counter();result=tool.candidates(view,category,out/(name+'.json'));raw[name]=result
                row=dict(dataset=ds,episode_id=job['episode_id'],backend=name,query=category,
                    status=result['status'],candidates=len(result['candidates']),seconds=time.perf_counter()-started)
                rows.append(row);print(row,flush=True)
            sources,overlays=g.prepare_sources(view,category,raw,out/'proposals')
            save(out/'sources.json',dict(sources=sources,views=overlays))
            # One preselected case per dataset; never choose a crop with GT.
            if index in (0,3):
                candidates=raw['dino']['candidates'] or raw['sam3']['candidates']
                if candidates:
                    tool=j.JevEvidence('http://127.0.0.1:8770',model='imajev-4b')
                    if not tool.available():raise RuntimeError('Jev unavailable')
                    box=candidates[0]['bbox']
                    questions=[dict(id=k,kind=k,object=category,bbox=box) for k in ('presence','color')]
                    started=time.perf_counter();result=tool.inspect(view,questions,out/'jev.json')
                    row=dict(dataset=ds,episode_id=job['episode_id'],backend='jev',status=result['status'],
                        answers=[dict(kind=a['kind'],choice=a.get('choice'),status=a['status']) for a in result['answers']],seconds=time.perf_counter()-started)
                    rows.append(row);print(row,flush=True)
        finally:env.close()
        save(dest/'summary.json',rows)


if __name__=='__main__':main()
