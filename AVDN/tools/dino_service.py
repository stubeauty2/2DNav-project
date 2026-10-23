"""Local GroundingDINO service. Run in the GPU environment; no GT access."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True)
    p.add_argument('--port',type=int,default=8766);p.add_argument('--device',default='cuda')
    p.add_argument('--threshold',type=float,default=.2);a=p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(a.model_dir,local_files_only=True).to(a.device).eval()
    model.config.disable_custom_kernels=True
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data):
            body=json.dumps(data).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,dict(backend='dino',model=str(Path(a.model_dir).resolve()),device=a.device,ready=True))
        def do_POST(self):
            try:
                if self.path!='/v1/candidates':raise ValueError('Unknown route')
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<2_000_000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size))
                if data['backend']!='dino' or len(data['views'])!=1:raise ValueError('Expected one DINO view')
                view=data['views'][0]
                image=Image.open(view['image_path']).convert('RGB')
                text=' '.join(c['text'].strip().rstrip('.')+'.' for c in data['concepts'])
                with lock,torch.inference_mode():
                    inputs=processor(images=image,text=text,return_tensors='pt').to(a.device)
                    outputs=model(**inputs)
                    found=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
                        threshold=a.threshold,text_threshold=.2,target_sizes=[image.size[::-1]])[0]
                proposals=[]
                labels=found.get('text_labels',found.get('labels',[]))
                for box,score,label in zip(found['boxes'].cpu().tolist(),found['scores'].cpu().tolist(),labels):
                    box=[max(0.,min(float(v),image.size[j%2])) for j,v in enumerate(box)]
                    if box[2]<=box[0] or box[3]<=box[1]:continue
                    proposals.append(dict(candidate_id='dino-'+str(len(proposals)),view_id=view['view_id'],bbox_2d=box,score=score,label=str(label)))
                proposals=sorted(proposals,key=lambda r:-r['score'])[:data.get('max_candidates_per_view',24)]
                self.reply(200,dict(backend='dino',coordinate_extent=list(image.size),candidates=proposals))
            except Exception as exc:self.reply(500,dict(error=type(exc).__name__+': '+str(exc)[:300]))
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print('DINO ready on 127.0.0.1:'+str(a.port),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
