"""SatNav preparation and initialization entrypoint."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--dataset', default='datasets/SatNav')
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--count', type=int, default=5)
    prepare.add_argument('--seed', type=int, default=20261109)
    init = sub.add_parser('initialize')
    init.add_argument('--root', required=True)
    init.add_argument('--model', default='qwen3.8-flash')
    run = sub.add_parser('run')
    run.add_argument('--root', required=True)
    run.add_argument('--model', default='qwen3.8-flash')
    run.add_argument('--limit', type=int, default=0)
    run.add_argument('--grounder', choices=['sam3', 'dino', 'locate_anything'], default='sam3')
    run.add_argument('--prompt-variant', choices=['baseline','satnav','exploration'], default='baseline')
    run.add_argument('--route-guidance', action='store_true')
    run.add_argument('--route-guard', action='store_true')
    run.add_argument('--route-stop-gate', action='store_true')
    run.add_argument('--semantic-route', action='store_true')
    run.add_argument('--bounded-exploration', action='store_true')
    run.add_argument('--partial-route', action='store_true')
    run.add_argument('--max-rounds', type=int, default=12)
    run.add_argument('--max-requests', type=int, default=48)
    run.add_argument('--ray-region-stop', action='store_true')
    run.add_argument('--search-prior', action='store_true')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--root', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from .prepare import prepare
        print(prepare(args.dataset, args.root, args.count, args.seed))
    elif args.command == 'initialize':
        from .initialization.client import initialize
        initialize(args.root, args.model)
    elif args.command == 'run':
        from .run import run
        run(args.root, args.model, args.limit, args.grounder, args.prompt_variant, args.route_guidance, args.route_guard, args.route_stop_gate, args.semantic_route, args.bounded_exploration, args.partial_route, args.max_rounds, args.max_requests, args.ray_region_stop, args.search_prior)
    else:
        from .evaluate import evaluate
        print(evaluate(args.root))


if __name__ == '__main__':
    main()
