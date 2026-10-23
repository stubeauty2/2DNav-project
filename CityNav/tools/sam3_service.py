"""Local official SAM3 image inference; candidates are not target confirmations."""
import argparse
import json
import threading
import uuid
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', type=float, default=.08)
    parser.add_argument('--output', default='out/sam3_service_cache')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import numpy as np
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False,
        device=args.device, eval_mode=True, compile=False)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, dict(backend='sam3', checkpoint=str(checkpoint), ready=True,
                device=args.device, threshold=args.threshold))

        def do_POST(self):
            try:
                if self.path != '/v1/candidates':
                    raise ValueError('Unknown route')
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 2_000_000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if data['backend'] != 'sam3' or len(data['views']) != 1:
                    raise ValueError('Expected one SAM3 view')
                view = data['views'][0]
                image = Image.open(view['image_path']).convert('RGB')
                proposals = []
                precision = torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
                with lock, torch.inference_mode(), precision:
                    state = processor.set_image(image)
                    for concept in data['concepts']:
                        state = processor.set_text_prompt(prompt=concept['text'], state=state)
                        masks, scores = state.get('masks'), state.get('scores')
                        if masks is None or scores is None:
                            continue
                        for mask, score in zip(masks.detach().cpu().numpy(), scores.detach().float().cpu().numpy()):
                            binary = np.squeeze(mask) > 0
                            if binary.shape != (image.height, image.width):
                                raise ValueError('SAM3 mask size differs from source view')
                            ys, xs = np.where(binary)
                            if len(xs) < 4:
                                continue
                            proposals.append(dict(mask=binary, score=float(np.asarray(score).item()),
                                bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                                label=concept['text']))
                kept = []
                for item in sorted(proposals, key=lambda r:-r['score']):
                    # Match the previous SAM3 exact-mask duplicate suppression.
                    if any((np.logical_and(item['mask'], other['mask']).sum() /
                            np.logical_or(item['mask'], other['mask']).sum()) >= .65 for other in kept):
                        continue
                    kept.append(item)
                    if len(kept) >= min(24, max(1, int(data.get('max_candidates_per_view', 24)))):
                        break
                directory = output / uuid.uuid4().hex
                directory.mkdir()
                candidates = []
                for i, item in enumerate(kept):
                    mask_path = directory / f'{i}.png'
                    Image.fromarray(item['mask'].astype('uint8')*255).save(mask_path)
                    candidates.append(dict(candidate_id=f'sam3-{i}', view_id=view['view_id'],
                        bbox_2d=item['bbox'], score=item['score'], mask_path=str(mask_path), label=item['label']))
                self.reply(200, dict(backend='sam3', coordinate_extent=list(image.size), candidates=candidates))
            except Exception as exc:
                self.reply(500, dict(error=type(exc).__name__+': '+str(exc)[:300]))

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'SAM3 ready on 127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
