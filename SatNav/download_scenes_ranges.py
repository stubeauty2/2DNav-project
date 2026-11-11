"""Bounded HTTP range downloads with pinned revision and complete SHA256 checks."""
import hashlib,json,time,urllib.request
import subprocess,tempfile,re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import get_token
root=Path(__file__).resolve().parents[1]/'datasets/SatNav/scenes_release'
rev=(root/'revision.txt').read_text().strip(); token=get_token(); results=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def download(line):
 expected,name=line.split(maxsplit=1); dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if not (dest.exists() and digest(dest)==expected):
  candidates=list((root/'.cache/huggingface/download/scenes').glob('*.'+expected+'.incomplete'))
  part=candidates[0] if candidates else dest.with_suffix(dest.suffix+'.part')
  while True:
   offset=part.stat().st_size if part.exists() else 0
   for attempt in range(30):
    try:
     url=f'https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1/resolve/{rev}/{name}'
     with tempfile.TemporaryDirectory() as tmp:
      headers=Path(tmp)/'headers'
      config=f'header = "Authorization: Bearer {token}"\n'
      proc=subprocess.run(['curl.exe','--config','-','-fLsS','--connect-timeout','20','--max-time','60','--range',f'{offset}-{offset+1024*1024-1}','--dump-header',str(headers),url],input=config.encode(),capture_output=True)
      if proc.returncode not in (0,18,28):raise RuntimeError('curl exit '+str(proc.returncode))
      ranges=re.findall(r'(?im)^content-range:\s*(.*)$',headers.read_text())
      cr=ranges[-1].strip() if ranges else ''
      if not cr.startswith(f'bytes {offset}-'):raise RuntimeError('Invalid range response')
      data=proc.stdout;total=int(cr.split('/')[1]);end=int(cr.split('-')[1].split('/')[0])
      if not data or len(data)>end-offset+1:raise RuntimeError('Invalid chunk length')
      if proc.returncode==0 and len(data)!=end-offset+1:raise RuntimeError('Incomplete chunk')
     with part.open('ab') as f:f.write(data)
     break
    except Exception as exc:
     print('RETRY',name,offset,str(exc),flush=True)
     if attempt==29:raise
     time.sleep(min(30,2*(attempt+1)))
   if part.stat().st_size==total:break
  if digest(part)!=expected:raise RuntimeError('SHA256 mismatch: '+name)
  part.replace(dest)
 return {'path':name,'bytes':dest.stat().st_size,'sha256':expected,'verified':True}

with ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(download,line) for line in (root/'SHA256SUMS').read_text().splitlines()]
 for future in as_completed(futures):
  item=future.result();results.append(item)
  temp=root/'verification.json.tmp'
  temp.write_text(json.dumps({'revision':rev,'files':results},indent=2))
  temp.replace(root/'verification.json')
  print('VERIFIED',len(results),item['path'],flush=True)
print('COMPLETE',len(results),flush=True)
