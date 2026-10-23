"""Full unseen G5 evaluation with coverage preflight and a frozen source manifest."""
import argparse,csv,getpass,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]
from run_spatial_sample2 import sources,write

def preflight(out):
    from andn_core.tools.termination import stop_only_instruction
    anno=ROOT/'datasets/FULL/test_unseen_full_data.json'
    parse=ROOT/'out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv'
    baseline=ROOT/'out/02_sam3/search/preds_sam3_test_unseen_full/turn_and_crop_eval_results_full_traj.json'
    annotations=json.loads(anno.read_text(encoding='utf8'))
    ids=[str(a['map_name'])+'__'+str(a['route_index']) for a in annotations]
    assert len(ids)==len(set(ids))==432
    rows=list(csv.DictReader(parse.open(encoding='utf-8-sig')))
    parsed={r['instr_id'] for r in rows}
    terminal={i for i,a in zip(ids,annotations) if stop_only_instruction(a['instructions'])}
    missing=set(ids)-parsed
    assert missing<=terminal,('Unresolved missing parses',missing-terminal)
    maps={str(a['map_name']) for a in annotations}
    absent=[m for m in maps if not (ROOT/'datasets/AVDN/train_images'/(m+'.tif')).is_file()]
    assert not absent,absent
    previous=json.loads(baseline.read_text(encoding='utf8'))['predictions']
    # Audit executable constants, not test IDs or README experiment descriptions.
    import ast,re
    hits=[]
    for p in (ROOT/'PSC-AVDN/script/andn_core').rglob('*.py'):
        for node in ast.walk(ast.parse(p.read_text(encoding='utf8'))):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                if re.search(r'\b\d+__\d+\b|datasets[/\\]sample2',node.value):
                    hits.append(dict(path=str(p.relative_to(ROOT)),line=node.lineno,value=node.value))
    assert not hits,hits
    report=dict(annotation_count=len(ids),unique_ids=len(set(ids)),parsed_ids=len(parsed),
        missing_parse=sorted(missing),explicit_stop_ids=sorted(terminal),missing_images=absent,
        historical_count=len(previous),without_historical_baseline=sorted(set(ids)-set(previous)),
        executable_sample_or_trajectory_constants=hits,
        policy='No ID-dependent navigation. Stop-only command recognized by complete instruction semantics, not GT.',
        parse_sha256=hashlib.sha256(parse.read_bytes()).hexdigest(),
        annotation_sha256=hashlib.sha256(anno.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    write(out/'preflight.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    check=preflight(out);print('PREFLIGHT',json.dumps(check,ensure_ascii=False),flush=True)
    if args.preflight_only:return
    if (out/'g5').exists():raise RuntimeError('Refusing to overwrite an existing run')
    import requests
    from andn_core import runner
    from andn_core.config import QWEN_URL,QWEN_MODEL
    service='http://127.0.0.1:8765'
    health=requests.get(service+'/health',timeout=10).json()
    assert health.get('prewarmed'),health
    key=os.getenv('API_KEY') or os.getenv('QWEN_API_KEY') or getpass.getpass('API key (hidden, memory only): ')
    assert key,'Missing credential'
    response=requests.post(QWEN_URL,headers={'Authorization':'Bearer '+key},json=dict(model=QWEN_MODEL,
        messages=[dict(role='user',content='Reply OK.')],max_tokens=4),timeout=90)
    if response.status_code!=200 or response.json().get('error'):
        raise RuntimeError('Model preflight HTTP '+str(response.status_code))
    manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),source_sha256=sources(),
        model=QWEN_MODEL,sam3=health,spatial_mode='g5',candidate_count=0,confirmation_review='off',
        dataset='datasets/AVDN',annotations='datasets/FULL',split='test_unseen_full',preflight=check,
        changes='General stop-only instruction support; primary SAM3 plus approved G5 rescue; no new prompt',
        comparison='432 total; historical matched subset only; missing historical record is not a gain')
    write(out/'manifest.json',manifest);write(out/'status.json',dict(status='running',pid=os.getpid()))
    try:
        runner.run_turn_and_crop_full_trajectory(anno_dir=str(ROOT/'datasets/FULL'),dataset_dir=str(ROOT/'datasets/AVDN'),
            split='test_unseen_full',results_csv=str(ROOT/'out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv'),
            out_dir=str(out/'g5'),api_key=key,url=QWEN_URL,model=QWEN_MODEL,vision_tool_url=service,
            candidate_count=0,confirmation_review='off',spatial_mode='g5',trace=True)
        predictions=json.loads((out/'g5/turn_and_crop_eval_results_full_traj.json').read_text(encoding='utf8'))['predictions']
        assert len(predictions)==432
        assert sources()==manifest['source_sha256'],'Source changed during run'
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',pid=os.getpid(),error_type=type(exc).__name__))
        raise
    write(out/'status.json',dict(status='completed',pid=os.getpid(),trajectories=432))
    print('FULL_G5_COMPLETE',flush=True)

if __name__=='__main__':main()
