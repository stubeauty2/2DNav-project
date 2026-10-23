"""Offline preparation and scoring, deliberately outside the online packages.

python tests/nav_offline/cli.py --dataset AVDN evaluate --folder out/run/avdn
"""
import argparse
import importlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['AVDN','CityNav'],required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare-init',help='Build public initial observations from raw annotations')
    p.add_argument('--annotation',required=True)
    p.add_argument('--image-dir')
    p.add_argument('--objects');p.add_argument('--landmarks')
    p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=8)
    p.add_argument('--seed',type=int,default=13)
    p=sub.add_parser('prepare-cached',help='Convert the preserved development initialization cache')
    p.add_argument('--root',required=True);p.add_argument('--count',type=int,default=10)
    p=sub.add_parser('evaluate-init')
    p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('evaluate')
    p.add_argument('--folder',required=True)
    args=parser.parse_args(argv)
    package='nav_offline.'+args.dataset.lower()
    if args.command=='prepare-init':
        if args.dataset=='CityNav' and (not args.objects or not args.landmarks):
            parser.error('CityNav prepare-init requires --objects and --landmarks')
        importlib.import_module(package+'.initialization').prepare(args)
    elif args.command=='prepare-cached':
        print(importlib.import_module(package+'.prepare').prepare(args.root,args.count))
    elif args.command=='evaluate-init':
        importlib.import_module(package+'.initialization').evaluate(args)
    else:
        print(importlib.import_module(package+'.evaluate').evaluate(args.folder))


if __name__=='__main__':main()
