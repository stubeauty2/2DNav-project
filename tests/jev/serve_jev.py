"""Serve the official OneJev engine with pinned provenance and startup equivalence check."""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2] / 'out/jev_study_20261022'

def serial_inference_worker(app):
    """Run the unchanged upstream endpoint off-loop, with one inference at a time."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import Request
    route = next(r for r in app.router.routes if getattr(r, 'path', '') == '/v1/systemone')
    upstream = route.endpoint
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jev-inference')
    app.router.routes.remove(route)

    @app.post('/v1/systemone')
    async def systemone(request: Request):
        # Cache bytes on the request's own event loop before the upstream handler
        # reads JSON on its worker loop. Validation and inference stay upstream.
        await request.body()
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: asyncio.run(upstream(request)))

    app.state.inference_executor = executor
    return app

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['0.8B', '4B'], required=True)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--attention', choices=['original', 'efficient'], default='efficient')
    parser.add_argument('--manifest-tag', default='')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import qev
    import torch
    import transformers
    import uvicorn
    from qev.mm_engine import MMDecisionEngine
    from qev.server import create_app
    if args.attention == 'efficient':
        # Windows wheel has no FlashAttention. Native SDPA GQA falls back to
        # quadratic math attention at head_dim=256. Let the existing HF adapter
        # expand K/V heads, then use the supported memory-efficient kernel.
        import transformers.integrations.sdpa_attention as adapter
        adapter.use_gqa_in_sdpa = lambda attention_mask, key, value: False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    name = 'OneJev1.6-' + args.size
    weights = json.loads((ROOT / 'models' / (name + '.manifest.json')).read_text(encoding='utf-8'))
    folder = Path(weights['local_path'])
    if not weights['completed']:
        raise RuntimeError('Incomplete weight inventory')
    for rel, info in weights['files'].items():
        p = folder / rel
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if p.stat().st_size != info['bytes'] or h.hexdigest() != info['sha256']:
            raise RuntimeError('Weight file changed: ' + rel)
    # auto executes upstream's shared-prefix/full-prompt equivalence check.
    # Thereafter force sequential branches for all experiment requests.
    torch.manual_seed(20261022)
    engine = MMDecisionEngine(str(folder), device='cuda:0', dtype='bfloat16',
        head_dtype='float32', fork_mode='auto', cuda_graphs=False,
        gpu_preprocess=False, allow_local_paths=False)
    engine.info.fork_mode = 'sequential'
    code = Path(qev.__file__).parent
    source = {p.relative_to(code).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(code.rglob('*.py'))}
    commit = subprocess.check_output(['git', '-C', str(code.parent), 'rev-parse', 'HEAD'], text=True).strip()
    provenance = dict(schema='jev-service-v1', served_model=name, repository=weights['repository'],
        checkpoint_revision=weights['revision'], weight_files=weights['files'],
        upstream_commit=commit, source_sha256=source, wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__, qev=qev.__version__),
        settings=dict(device='cuda:0', dtype='bfloat16', head_dtype='float32',
                      fork_mode='sequential', debias=1, cuda_graphs=False, gpu_preprocess=False,
                      attention=args.attention, expanded_kv=args.attention=='efficient',
                      inference_workers=1, responsive_health=True),
        calibration='Upstream default T=1.0; no navigation-domain calibration')
    stem = name + args.manifest_tag
    manifest = ROOT / 'services' / (stem + '.manifest.json')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != provenance:
        raise RuntimeError('Service identity changed; use a new experiment root')
    manifest.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (manifest.parent / (stem + '.startup.json')).write_text(json.dumps(
        dict(pid=os.getpid(), engine=asdict(engine.info)), indent=2), encoding='utf-8')
    app = create_app(engine, model_name=name)
    serial_inference_worker(app)
    app.state.debias = 1
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, 'path', '') != '/health']
    @app.get('/health')
    async def health():
        return dict(status='ok', model=name, fork_mode=engine.info.fork_mode,
                    prompt_version=engine.info.prompt_version,
                    checkpoint_revision=weights['revision'], service_manifest_sha256=digest(provenance))
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')

if __name__ == '__main__':
    main()
