"""Offline replay of historical SAM3 responses and Image 5 evidence.

This exercises real request construction, reply parsing, views, motion and final
geometry, without calling a model or loading SAM3 weights. Run before/after in
separate processes and compare their audit.json records.
"""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
REF='57b9ecb31c2b64967d6dea9aaebdbce92c8f02a8'
BASE=ROOT/'out/02_sam3/search/preds_sam3_test_unseen_full'
OUT=ROOT/'out/sam3_core_restructure_20261006'
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['before','after'],required=True)
    parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--trace',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--spatial-mode',choices=('off','g0','g5'),default='off')
    args=parser.parse_args();OUT=args.output_root.resolve()
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    if args.version=='before':
        source=OUT/'Search_Confirmation.original.py'
        source.write_bytes(subprocess.check_output(['git','show',REF+':PSC-AVDN/script/ANDH-Full/Search_Confirmation.py'],cwd=ROOT))
        category=OUT/'category_terms.py'
        category.write_bytes(subprocess.check_output(['git','show',REF+':PSC-AVDN/script/ANDH-Full/category_terms.py'],cwd=ROOT))
        sys.path.insert(0,str(OUT))
        spec=importlib.util.spec_from_file_location('frozen_sam3',OUT/'Search_Confirmation.original.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);nav=runner
    else:
        from andn_core import runner,navigation as nav
    original=runner.search_and_reach_destination
    state={};requests=[];stages={}
    def search(**kwargs):
        stage=kwargs['instr_id'];archive=load(BASE/(stage+'_search.json'))
        state.update(stage=stage,replies=[s[k]['raw'] for s in archive['steps'] for k in ('qwen','confirm') if k in s],reply_index=0,images=[s['qwen']['sam3_image5'] for s in archive['steps'] if 'qwen' in s and 'sam3_image5' in s['qwen']],image_index=0)
        result=original(**kwargs)
        assert state['reply_index']==len(state['replies']),stage
        assert state['image_index']==len(state['images']),stage
        stages[stage]=load(Path(kwargs['out_dir'])/(stage+'_search.json'))
        return result
    def image5(patch_main_bgr,target_terms,*,image_path,artifact_dir,vision_tool_url):
        meta=deepcopy(state['images'][state['image_index']]);state['image_index']+=1
        expected=BASE/'sam3_image5'/Path(artifact_dir).name/'image5.png'
        image=cv2.imread(str(expected));assert image is not None,expected
        assert target_terms==meta['target_terms']
        return image,meta
    class Reply:
        headers={'Content-Type':'application/json'}
        def __init__(self,content):self.content=content
        def json(self):return {'choices':[{'message':{'content':self.content}}]}
    def post(url,*,json,headers,timeout):
        content=state['replies'][state['reply_index']];state['reply_index']+=1
        import json as js
        digest=hashlib.sha256(js.dumps(json,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
        requests.append(dict(stage=state['stage'],index=state['reply_index'],payload_sha256=digest,timeout=timeout,url=url))
        return Reply(content)
    land_records={}
    sample_ids={str(r['map_name'])+'__'+str(r['route_index']) for r in load(ROOT/'datasets/sample2/test_unseen_full_data.json')}
    for p in BASE.glob('*_analysis.json'):
        if any(p.name.startswith(i+'_') for i in sample_ids):
            d=load(p)
            if d.get('used_view')=='absolute':land_records[d['instruction']]=d
    def land(instruction,*a,**kw):
        d=land_records[instruction];return d['clock'],d.get('reason',''),d.get('meta',{})
    dest=OUT/('replay_'+args.version);dest.mkdir(exist_ok=False)
    os.chdir(dest)
    rescue_stub=patch.object(nav,'rescue_failed_stage',return_value=(False,None,[],None,dict(status='replay_no_selection',committed=False))) if args.spatial_mode=='g5' else nullcontext()
    with patch.object(runner,'search_and_reach_destination',search),patch.object(nav,'_sam3_image5',image5),patch.object(runner,'decide_turn_from_land',land),patch('requests.post',post),patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')),rescue_stub:
        options={'trace':args.trace,'spatial_mode':args.spatial_mode} if args.version=='after' else {}
        runner.run_turn_and_crop_full_trajectory(anno_dir=str(ROOT/'datasets/sample2'),dataset_dir=str(ROOT/'datasets/sample2'),split='test_unseen_full',results_csv=str(ROOT/'out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv'),out_dir=str(dest),api_key='',url='offline://recorded-model',model='qwen-vl-max',vision_tool_url='offline://recorded-sam3',**options)
    result=load(dest/'turn_and_crop_eval_results_full_traj.json');reference=load(BASE/'turn_and_crop_eval_results_full_traj.json')['predictions']
    assert len(result['predictions'])==20
    differences={iid:abs(p['final_iou']-reference[iid]['final_iou']) for iid,p in result['predictions'].items()}
    assert max(differences.values())<1e-8,differences
    output=dict(version=args.version,result=result,requests=requests,stages=stages,reference_iou_max_error=max(differences.values()),sam3_evidence='recorded Image 5; no GPU inference',land='recorded historical response including original fallback')
    if args.spatial_mode=='g5':output['rescue']='forced no selection to test primary SAM3 preservation; not a G5 online score'
    (dest/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPLAY_COMPLETE',args.version,'trajectories',len(result['predictions']),'calls',len(requests),'metrics',result['metrics'],flush=True)

if __name__=='__main__':main()
