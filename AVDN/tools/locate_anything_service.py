"""Loopback LocateAnything service using the official text-grounding inference API.

Download nvidia/LocateAnything-3B and the official locateanything_worker.py into the
model directory. Its code is external to the navigation package and loaded explicitly.
Uses standard SDPA inference when the optional CUDA kernels are unavailable.
"""
import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

def boxes_from_answer(answer):
    """Preserve native normalized coordinates; malformed output is an error."""
    remainder=re.sub(r'<ref>.*?</ref>','',answer,flags=re.S).replace('<|im_end|>','').strip().lower()
    if remainder in ('none','none.','<none>','<box>none</box>'):
        return []
    boxes=[]
    for match in re.finditer(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>',answer):
        b=list(map(int,match.groups()))
        if not (0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000):
            raise ValueError('LocateAnything returned invalid box coordinates')
        boxes.append(b)
    if answer.count('<box>')!=len(boxes):
        raise ValueError('Malformed box output')
    remainder=re.sub(r'<ref>.*?</ref>','',answer,flags=re.S).replace('<|im_end|>','').strip().lower()
    if not boxes and remainder not in ('none','none.','<none>'):
        raise ValueError('Unrecognized grounding response; not an empty detection')
    return boxes

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True)
    p.add_argument('--port',type=int,default=8767);p.add_argument('--device',default='cuda')
    a=p.parse_args();path=Path(a.model_dir).resolve()
    sys.path.insert(0,str(path))
    from locateanything_worker import LocateAnythingWorker
    from PIL import Image
    worker=LocateAnythingWorker(str(path),device=a.device)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data):
            body=json.dumps(data).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,dict(backend='locate_anything',model=str(path),device=a.device,ready=True))
        def do_POST(self):
            response=None
            try:
                if self.path!='/v1/candidates':raise ValueError('Unknown route')
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<2_000_000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size))
                if data['backend']!='locate_anything' or len(data['views'])!=1:raise ValueError('Expected one view')
                view=data['views'][0]
                image=Image.open(view['image_path']).convert('RGB')
                query=' '.join(c['text'] for c in data['concepts'])
                with lock:
                    response=worker.ground_multi(image,query,generation_mode='slow',
                        max_new_tokens=512,temperature=0.,verbose=False)
                boxes=boxes_from_answer(response['answer'])
                proposals=[dict(candidate_id='locate-'+str(i),view_id=view['view_id'],bbox_2d=b,
                    score=None,label=query) for i,b in enumerate(boxes[:data.get('max_candidates_per_view',24)])]
                self.reply(200,dict(backend='locate_anything',coordinate_extent=[1000,1000],
                    candidates=proposals,answer=response['answer']))
            except Exception as exc:self.reply(500,dict(error=type(exc).__name__+': '+str(exc)[:300],
                answer=response.get('answer') if isinstance(response,dict) else None))
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print('LocateAnything ready on 127.0.0.1:'+str(a.port),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
