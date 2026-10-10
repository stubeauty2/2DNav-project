"""Frozen G1--G4 matrix with a contemporaneous C1 control; never overwrite a run."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)


def sources():
    paths=list((ROOT/'PSC-AVDN/script/andn_core').rglob('*.py'))+[ROOT/'PSC-AVDN/script/util.py',ROOT/'src/env.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    import requests
    from andn_core import runner
    from andn_core.config import QWEN_URL,QWEN_MODEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',choices=('off','g1','g2','g3','g4','g5'),default=['g4','off','g2','g3','g1'])
    parser.add_argument('--confirmation-review',choices=('off','once'),default='off',
                        help='Separate combined experiment using the previously approved STEP 03 review')
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    is_g5=args.modes==['g5']
    if 'g5' in args.modes and (not is_g5 or args.confirmation_review!='off'):
        raise ValueError('Run G5 separately with its original SAM3 primary and review off')
    if any((out/mode).exists() for mode in args.modes):raise RuntimeError('A requested run already exists; refusing overwrite')
    service='http://127.0.0.1:8765'
    health=requests.get(service+'/health',timeout=10).json()
    if not health.get('prewarmed'):raise RuntimeError('SAM3 is not ready')
    key=os.getenv('API_KEY') or getpass.getpass('API key (hidden, memory only): ')
    if not key:raise RuntimeError('Missing credential')
    r=requests.post(QWEN_URL,headers={'Authorization':'Bearer '+key},json=dict(model=QWEN_MODEL,
        messages=[dict(role='user',content='Reply OK.')],max_tokens=4),timeout=90)
    if r.status_code!=200 or r.json().get('error'):raise RuntimeError('Model preflight failed; HTTP '+str(r.status_code))
    manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),modes=args.modes,source_sha256=sources(),
        candidate_count=0 if is_g5 else 2,confirmation_review=args.confirmation_review,
        search_prompt='Original SAM3 primary; approved G4 only after found=false' if is_g5 else 'original C1 plus approved spatial additions; S1 not enabled',
        parse_csv='out/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv',
        annotation_sha256=hashlib.sha256((ROOT/'datasets/sample2/test_unseen_full_data.json').read_bytes()).hexdigest(),
        model=QWEN_MODEL,sam3=health,comparison='Historical SAM3 plus same-run primary SAM3 counterfactual' if is_g5 else 'Historical SAM3 20 matching trajectories, plus contemporary C1 off control',
        budget='G5: primary max3 rounds; one extra G4 Search and at most one C1 Confirmation per failed stage; max2 rescue candidates' if is_g5 else 'max 3 rounds/stage; max 2 candidates/round; SAM3 on actual-position main only',
        land='Original util defaults retained; any fallback remains in metrics',pid=os.getpid())
    write(out/'matrix_manifest.json',manifest)
    for mode in args.modes:
        assert sources()==manifest['source_sha256'],'Source changed during matrix'
        write(out/'matrix_status.json',dict(mode=mode,status='running',pid=os.getpid()))
        print('START_MODE',mode,flush=True)
        runner.run_turn_and_crop_full_trajectory(anno_dir=str(ROOT/'datasets/sample2'),dataset_dir=str(ROOT/'datasets/sample2'),
            split='test_unseen_full',results_csv=str(ROOT/manifest['parse_csv']),out_dir=str(out/mode),
            api_key=key,url=QWEN_URL,model=QWEN_MODEL,vision_tool_url=service,candidate_count=manifest['candidate_count'],
            confirmation_review=args.confirmation_review,spatial_mode=mode,trace=True)
        results=json.loads((out/mode/'turn_and_crop_eval_results_full_traj.json').read_text(encoding='utf8'))
        assert len(results['predictions'])==20
        assert sources()==manifest['source_sha256'],'Source changed during matrix'
        print('END_MODE',mode,results['metrics'],flush=True)
    write(out/'matrix_status.json',dict(status='completed',modes=args.modes,pid=os.getpid()))
    print('MATRIX_COMPLETE',flush=True)


if __name__=='__main__':main()
