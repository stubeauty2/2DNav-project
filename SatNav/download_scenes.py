"""Download a pinned official scene release and verify every listed file."""
import hashlib,json,time
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(__file__).resolve().parents[1]/'datasets'/'SatNav'/'scenes_release'
revision=(root/'revision.txt').read_text().strip()
results=[]
for line in (root/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split(maxsplit=1)
 for attempt in range(4):
  try:
   path=Path(hf_hub_download('Eku127/SatNav-Scenes-v0.1',name,repo_type='dataset',revision=revision,local_dir=root))
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   if h.hexdigest()!=expected:raise RuntimeError('SHA256 mismatch: '+name)
   results.append({'path':name,'bytes':path.stat().st_size,'sha256':expected,'verified':True})
   (root/'verification.json').write_text(json.dumps({'revision':revision,'files':results},indent=2))
   print('VERIFIED',len(results),name,flush=True);break
  except Exception as exc:
   print('RETRY',name,attempt+1,type(exc).__name__,flush=True)
   if attempt==3:raise
   time.sleep(5)
print('COMPLETE',len(results),flush=True)
