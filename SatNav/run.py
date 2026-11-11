"""Run public episodes only; never import the offline evaluator."""
import hashlib
import os
import shutil
from pathlib import Path
from AVDN.agent.io import read_jsonl, load_json, save_json, digest
from AVDN.agent.contracts import public_episode
from AVDN.tools.llm import Model, AgentBudget
from AVDN.tools.sam3 import SAM3
from AVDN.tools.dino import GroundingDINO
from AVDN.tools.locate_anything import LocateAnything
from .environment import Environment
from .navigation import Agent


def check_transport_result(result, output):
    """Stop the batch after API failure; retain the failed episode and journals."""
    if result['termination'] != 'interface_error':
        return
    failures = []
    for path in sorted((Path(output) / 'requests').glob('*.json')):
        record = load_json(path)
        if record.get('status') == 'api_error':
            failures.append(dict(request=path.name, error=record.get('error'),
                                 http_status=record.get('response', {}).get('status')))
    if failures:
        save_json(Path(output).parent.parent / 'transport_failure.json',
                  dict(episode_id=Path(output).name, failures=failures,
                       action='Batch stopped; remaining episodes not attempted. Preserve original failed result.'))
        raise RuntimeError('API transport failure; batch stopped. See transport_failure.json')


def run(root, model='qwen3.8-flash', limit=0, grounder='sam3', prompt_variant='baseline', route_guidance=False, route_guard=False, route_stop_gate=False, semantic_route=False, bounded_exploration=False, partial_route=False, max_rounds=12, max_requests=48, ray_region_stop=False, search_prior=False):
    root = Path(root)
    if search_prior and (not partial_route or semantic_route):
        raise ValueError('search_prior requires partial_route and excludes semantic_route')
    if ray_region_stop and not partial_route:
        raise ValueError('ray_region_stop requires partial_route')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (max_rounds,max_requests)):
        raise ValueError('Positive integer navigation budgets required')
    if partial_route and (not route_guidance or semantic_route):
        raise ValueError('partial_route requires route_guidance and excludes semantic_route')
    if route_guard and not route_guidance:
        raise ValueError('route_guard requires route_guidance')
    if route_stop_gate and not route_guidance:
        raise ValueError('route_stop_gate requires route_guidance')
    from .prompts import SYSTEM, REVIEW_SYSTEM, EXPLORATION_SYSTEM
    if prompt_variant not in ('baseline', 'satnav', 'exploration'):
        raise ValueError('Unknown prompt variant')
    key = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('API_KEY')
    if not key:
        raise ValueError('Configure DASHSCOPE_API_KEY or API_KEY')
    episodes = read_jsonl(root / 'episodes.jsonl')
    route_plans = {}
    semantic_plans = {}
    if semantic_route or search_prior:
        from .semantic_plan import validate_plan
        for prediction in read_jsonl(root/'semantic_plans.jsonl'):
            if prediction['status']=='ok':
                semantic_plans[prediction['episode_id']]=validate_plan(prediction['plan'],allow_qualitative=search_prior)
    if route_guidance:
        for prediction in read_jsonl(root/'initialization_results.jsonl'):
            if prediction['status']=='ok':
                # Explicit whitelist: never forward arbitrary diagnostic metadata.
                route_plans[prediction['episode_id']] = dict(stop_condition=prediction.get('diagnostics',{}).get('stop_condition',''),stages=[
                    dict(heading_deg=s['heading_deg'],distance_m=s['distance_m'])
                    for s in prediction.get('diagnostics',{}).get('stages',[])])
    search_priors={}
    if search_prior:
        from .search_prior import search_region
        statistics=load_json(root/'frozen_training_text_prior.json')
        for entry in episodes:
            plan=semantic_plans.get(entry['episode_id'])
            if plan:
                search_priors[entry['episode_id']]=search_region(entry['agent_input']['initial_pose']['heading_deg'],plan,statistics)
    for entry in episodes:
        public_episode(entry)
    if limit:
        episodes = episodes[:limit]
    urls = dict(sam3=os.environ.get('SAM3_URL') or os.environ.get('VISION_TOOL_URL'),
                dino=os.environ.get('DINO_URL'), locate_anything=os.environ.get('LOCATE_ANYTHING_URL'))
    grounders = dict(sam3=SAM3(urls['sam3']), dino=GroundingDINO(urls['dino']),
                    locate_anything=LocateAnything(urls['locate_anything']))
    if not grounders[grounder].available():
        raise ValueError('Selected grounding service is unavailable: ' + grounder)
    # Hash imported base implementation as well as SatNav adaptations.
    files = [p for package in ('AVDN', 'SatNav') for p in Path(package).rglob('*.py')]
    settings = dict(model=model, grounder=grounder, prompt_variant=prompt_variant,
                    route_guidance=route_guidance, route_guard=route_guard, route_stop_gate=route_stop_gate, route_plans_sha256=digest(route_plans), public_sha256=digest(episodes),
                    semantic_route=semantic_route, semantic_plans_sha256=digest(semantic_plans),
                    bounded_exploration=bounded_exploration,
                    partial_route=partial_route,
                    ray_region_stop=ray_region_stop,
                    search_prior=search_prior,search_priors_sha256=digest(search_priors),
                    runtime_sha256=digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}),
                    service_configuration_sha256=digest(urls), max_rounds=max_rounds, max_requests=max_requests,
                    request_timeout_s=180, connect_upload_timeout_s=90,
                    action_protocol='continuous project actions', view_size_px=768, footprint_m=100)
    manifest = root / 'run_manifest.json'
    if manifest.exists() and load_json(manifest) != settings:
        raise ValueError('Run configuration changed; use a fresh experiment directory')
    save_json(manifest, settings)
    for path in files:
        destination = root / 'source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    budget = AgentBudget(root / 'navigation_budget.json', None, save_json)
    for entry in episodes:
        output = root / 'episodes' / entry['episode_id']
        result_path = output / 'result.json'
        if result_path.exists() and load_json(result_path)['completed']:
            print(entry['episode_id'], 'cached', flush=True)
            continue
        env = Environment(entry['environment'], output / 'views')
        try:
            decision_prompt=SYSTEM if prompt_variant=='satnav' else EXPLORATION_SYSTEM if prompt_variant=='exploration' else None
            review_prompt=REVIEW_SYSTEM if prompt_variant=='satnav' else None
            if semantic_route:
                from .route_geometry import numeric_route
                from .semantic_prompts import SEMANTIC_SYSTEM, SEMANTIC_REVIEW_SYSTEM
                if semantic_plans.get(entry['episode_id'],{}).get('legs') and numeric_route(route_plans.get(entry['episode_id'])) is None:
                    decision_prompt,review_prompt=SEMANTIC_SYSTEM,SEMANTIC_REVIEW_SYSTEM
            client = Model(output / 'requests', budget, model, key, max_requests=max_requests, request_timeout=180, connect_timeout=90,
                           decision_prompt=decision_prompt,review_prompt=review_prompt)
            agent_type=Agent
            extras={}
            if partial_route:
                from .ray_navigation import RayAgent
                agent_type=RayAgent
                extras['region_stop']=ray_region_stop
            if semantic_route:
                from .semantic_navigation import SemanticAgent
                agent_type=SemanticAgent
                extras['semantic_plan']=semantic_plans.get(entry['episode_id'])
            if search_prior:
                from .prior_navigation import PriorSearchAgent
                agent_type=PriorSearchAgent
                extras['search_prior']=search_priors.get(entry['episode_id'])
            result = agent_type(entry, env, client, grounders['sam3'], output,
                           max_rounds=max_rounds,
                           grounders=grounders, observation_grounder=grounder,
                           route_diagnostics=route_plans.get(entry['episode_id']),route_guard=route_guard,
                           route_stop_gate=route_stop_gate,bounded_exploration=bounded_exploration,**extras).run()
            print(entry['episode_id'], result['termination'], flush=True)
            check_transport_result(result, output)
        finally:
            env.close()
