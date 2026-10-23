"""Experiment-only CPU grounder launch; navigation/model source files stay intact.

SAM3 contains explicit BF16 intermediate casts. Its ordinary CPU launch uses a
null precision context, which mixes these activations with FP32 linear weights.
Install CPU autocast at that existing context boundary, before starting service.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['sam3','dino'],default='sam3')
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--threads',type=int)
    parser.add_argument('--manifest',type=Path)
    args,service_args=parser.parse_known_args()
    if args.threads is not None and args.threads<1:
        parser.error('--threads must be positive')
    # Keep CUDA visible: upstream SAM3 precomputes a positional encoding on CUDA
    # even for its CPU builder. All model parameters/inference remain on CPU.
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    import torch
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    service=importlib.import_module('AVDN.tools.'+args.backend+'_service')
    if args.backend=='sam3':
        service.nullcontext=lambda:torch.autocast('cpu',dtype=torch.bfloat16)
        import sam3
        builder=sam3.build_sam3_image_model
        def cpu_value(value):
            if isinstance(value,torch.Tensor):return value.to('cpu')
            if isinstance(value,dict):return {k:cpu_value(v) for k,v in value.items()}
            if isinstance(value,tuple):return tuple(cpu_value(v) for v in value)
            if isinstance(value,list):return [cpu_value(v) for v in value]
            return value
        def cpu_builder(*positional,**options):
            model=builder(*positional,**options).to('cpu')
            # These upstream caches are ordinary attributes, not registered
            # buffers, so Module.to(cpu) cannot migrate their CUDA tensors.
            moved=[]
            for name,part in model.named_modules():
                for field in ('cache','coord_cache','compilable_cord_cache'):
                    if hasattr(part,field):
                        setattr(part,field,cpu_value(getattr(part,field)))
                        moved.append(name+'.'+field)
            parameter_devices=sorted({str(p.device) for p in model.parameters()})
            if parameter_devices!=['cpu']:
                raise RuntimeError('CPU wrapper must not leave model parameters on CUDA')
            print(json.dumps(dict(cpu_cache_migration=moved,parameter_devices=parameter_devices)),flush=True)
            return model
        sam3.build_sam3_image_model=cpu_builder
    provenance=dict(backend=args.backend,device='cpu',pid=os.getpid(),torch=torch.__version__,
        cpu_capability=torch.backends.cpu.get_cpu_capability(),threads=torch.get_num_threads(),
        cpu_autocast='bfloat16' if args.backend=='sam3' else None,
        cpu_cache_migration=['cache','coord_cache','compilable_cord_cache'] if args.backend=='sam3' else [],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','inherited'),
        note='SAM3 upstream initialization may allocate a transient CUDA positional encoding; model device is CPU',
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        service_source_sha256=hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest(),
        arguments=service_args)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance),flush=True)
    sys.argv=[sys.argv[0],*service_args,'--device','cpu']
    service.main()


if __name__=='__main__':main()
