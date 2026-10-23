"""Frozen paired Jev navigation experiment; prepare/analyze alone can read GT.

prepare freezes 20 AVDN + 20 CityNav public episodes and six arms per episode.
run sends paid requests; worker is its subprocess entry. Neither run nor worker
opens evaluation labels. An attempt is never retried, including pending calls.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import copy
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import threading

WORK = Path(__file__).resolve().parents[2]
ROOT = WORK / 'out/jev_study_20261022/online'
HISTORY = WORK / 'out/04_current_initialization/agent_loop'
POOLS = {
    'AVDN': HISTORY / 'avdn_full_flash_20261014/avdn',
    'CityNav': HISTORY / 'grounding_holdout100_20261014_seed20261016/citynav',
}
FIXED = WORK / 'out/jev_study_20261022/fixed_review/public_requests.jsonl'
REFACTOR = WORK / 'out/refactor_online_compare_20261021/sample_manifest.json'
SERVICE = WORK / 'out/jev_study_20261022/services/OneJev1.6-4B-worker.manifest.json'
SEED = 20261022
MODELS = ('qwen3.8-flash', 'qwen3.8-max')
ARMS = ('mllm', 'mllm_fixed', 'jev')
JEV_MODEL = 'OneJev1.6-4B'
JEV_REVISION = '7d55a87ce87ff77e25c4a43eb637dc3a52342977'
LIMITS = dict(max_rounds=12, max_tools=2, max_requests=48, max_output_tokens=2048)
DEFAULT_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
PRIVATE = {'goal_xy_m', 'goal_polygon_xy_m', 'evaluation_polygon_xy_m',
           'gt_path_corners', 'source_row', 'source_route', 'correct',
           'historical_success', 'historical_metric', 'evaluator_proposal_score'}
SECRET_ENVS = ('DASHSCOPE_API_KEY', 'QWEN_API_KEY', 'API_KEY')
TERMINAL = {'complete', 'failed'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f'.{os.getpid()}.writing')
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    staging.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_only(value):
    if isinstance(value, dict):
        require(not PRIVATE.intersection(value), 'Evaluator data in public input')
        for child in value.values():
            public_only(child)
    elif isinstance(value, list):
        for child in value:
            public_only(child)


def sources(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for ds in ('AVDN', 'CityNav') for p in sorted((Path(root) / ds).rglob('*.py'))}


def environment():
    # Metadata only; no CUDA initialization and no credentials in this fingerprint.
    packages = ('numpy', 'Pillow', 'shapely', 'requests', 'rasterio', 'tifffile')
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), packages=versions)


def exclusions(fixed, refactor):
    result = {ds: set() for ds in POOLS}
    for row in fixed:
        result[row['dataset']].add(row['episode_id'])
    for row in refactor['selected']:
        result[row['dataset']].add(row['episode_id'])
    return result


def select_ids(pools, excluded, count=20, seed=SEED):
    """Selection takes identities only, never historical outcomes or GT."""
    rng = random.Random(seed)
    selected = {}
    for ds in ('AVDN', 'CityNav'):
        require(len(pools[ds]) == len(set(pools[ds])), 'Duplicate public episode ID')
        eligible = sorted(set(pools[ds]) - excluded[ds])
        require(len(eligible) >= count, 'Too few eligible episodes: ' + ds)
        selected[ds] = rng.sample(eligible, count)
    return selected


def make_jobs(selected):
    jobs = []
    for ds in ('AVDN', 'CityNav'):
        for ident in selected[ds]:
            for model in MODELS:
                for arm in ARMS:
                    jobs.append(dict(job_id=f'{ds}-{ident}-{model}-{arm}', dataset=ds,
                                     episode_id=ident, model=model, arm=arm,
                                     public_input=f'public/{ds}/{ident}.json'))
    random.Random(SEED + 1).shuffle(jobs)
    return jobs


def validate_service(manifest):
    require(manifest.get('served_model') == JEV_MODEL, 'Expected exact OneJev1.6-4B service')
    require(manifest.get('repository') == 'OmniJev/OneJev1.6-4B', 'Wrong Jev repository')
    require(manifest.get('checkpoint_revision') == JEV_REVISION, 'Wrong Jev checkpoint revision')
    settings = manifest.get('settings', {})
    require(settings.get('fork_mode') == 'sequential' and settings.get('debias') == 1,
            'Jev experiment requires sequential branches and debias=1')
    require(bool(manifest.get('weight_files')) and bool(manifest.get('source_sha256')),
            'Jev service manifest lacks weight/source hashes')


def prepare(root=ROOT, service=SERVICE, endpoint=DEFAULT_URL,
            sam3_url='http://127.0.0.1:8765', dino_url='http://127.0.0.1:8766',
            jev_url='http://127.0.0.1:8770'):
    root = Path(root).resolve()
    if (root / 'plan.json').exists():
        plan = validate_frozen(root, check_images=True)
        require(plan['settings']['endpoint'] == endpoint and
                plan['settings']['sam3_url'] == sam3_url and plan['settings']['dino_url'] == dino_url and
                plan['settings']['jev_url'] == jev_url, 'Prepare settings changed; use a new root')
        require(sha(service) == plan['service_file_sha256'], 'Requested service manifest changed')
        return plan
    require(not root.exists() or not any(root.iterdir()), 'Prepare needs an empty root; do not overwrite partial artifacts')
    service_data = read(service)
    validate_service(service_data)
    fixed, prior = lines(FIXED), read(REFACTOR)
    require({ds: len({r['episode_id'] for r in fixed if r['dataset'] == ds}) for ds in POOLS}
            == {'AVDN': 20, 'CityNav': 6}, 'Fixed-review development set changed')
    require(len(prior['selected']) == 20, 'Refactor development set changed')
    excluded = exclusions(fixed, prior)
    pools = {ds: lines(folder / 'episodes.jsonl') for ds, folder in POOLS.items()}
    require({ds: len(rows) for ds, rows in pools.items()} == {'AVDN': 432, 'CityNav': 50},
            'Expected the existing 432/50 public pools')
    ids = select_ids({ds: [r['episode_id'] for r in rows] for ds, rows in pools.items()}, excluded)
    # Only AFTER selection, join evaluation labels and verify raw-split membership.
    truths, selected_rows, image_files, pool_metadata = {}, [], {}, {}
    for ds, folder in POOLS.items():
        public = {r['episode_id']: r for r in pools[ds]}
        truth = {r['episode_id']: r for r in lines(folder / 'evaluator_only.jsonl')}
        metadata = read(folder / 'sample_manifest.json')
        annotation = Path(metadata['annotation'])
        require(sha(annotation) == metadata['annotation_sha256'], 'Raw annotation changed: ' + ds)
        raw = read(annotation)
        require(metadata['split'] == ('test_unseen_full' if ds == 'AVDN' else 'test_unseen'), 'Wrong split')
        pool_metadata[ds] = dict(public_pool=str(folder / 'episodes.jsonl'),
            public_sha256=sha(folder / 'episodes.jsonl'), pool_size=len(public),
            eligible_size=len(set(public) - excluded[ds]), excluded_ids=sorted(excluded[ds]),
            excluded_in_pool=sorted(set(public) & excluded[ds]),
            annotation=str(annotation), annotation_sha256=sha(annotation), split=metadata['split'])
        truths[ds] = {}
        for ident in ids[ds]:
            entry = copy.deepcopy(public[ident])
            public_only(entry)
            require(entry.get('initialization') is not None, 'Missing cached initialization; never resample')
            require(entry['agent_input'].get('views') == [], 'Expected empty initial RGB list')
            gt = truth[ident]
            original = raw[gt['source_row']]
            instruction = original['instructions'] if ds == 'AVDN' else original['descriptions'][0]
            require(instruction == entry['agent_input']['instruction_raw'], 'Raw split input mismatch')
            image_path = str(Path(entry['environment']['image_path']).resolve())
            if image_path not in image_files:
                image_files[image_path] = dict(bytes=Path(image_path).stat().st_size, sha256=sha(image_path))
            relative = f'public/{ds}/{ident}.json'
            selected_rows.append(dict(dataset=ds, episode_id=ident, public_input=relative,
                                      public_sha256=digest(entry), image_path=image_path))
            truths[ds][ident] = gt
            save(root / relative, entry)
    frozen_sources = sources(WORK)
    require(bool(frozen_sources), 'No runtime source files found')
    for relative in frozen_sources:
        target = root / 'runtime' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK / relative, target)
    shutil.copyfile(__file__, root / 'harness.py')
    shutil.copyfile(service, root / 'jev_service_manifest.json')
    save(root / 'evaluator_only.json', truths)
    jobs = make_jobs(ids)
    settings = dict(endpoint=endpoint, sam3_url=sam3_url, dino_url=dino_url, jev_url=jev_url,
                    jev_model=JEV_MODEL, jev_min_probability=.7, thinking_budget=0,
                    direction_mode='staged', limits=LIMITS, models=list(MODELS), arms=list(ARMS),
                    observation={'AVDN': dict(observation_grounder='dino', observation_query_mode='full'),
                                 'CityNav': dict(observation_grounder='sam3', observation_query_mode='category')})
    plan = dict(schema='jev-online-paired-v1', seed=SEED, dispatch_seed=SEED+1,
                selection='Uniform random.sample on sorted eligible public IDs; no outcome stratification or replacement',
                estimand='20 per dataset from remaining existing public pools; not the entire CityNav test distribution',
                initialization='Identical cached initialization for every arm/model; zero new initialization calls',
                settings=settings, selected=selected_rows, jobs=jobs, pools=pool_metadata,
                exclusions_sources={str(FIXED): sha(FIXED), str(REFACTOR): sha(REFACTOR)},
                runtime_sources=frozen_sources, source_root=str(WORK), harness_source=str(Path(__file__).resolve()),
                harness_sha256=sha(__file__), environment=environment(), image_files=image_files,
                evaluator_sha256=sha(root / 'evaluator_only.json'),
                service_source=str(Path(service).resolve()), service_file_sha256=sha(service),
                service_content_sha256=digest(service_data),
                budget=dict(episodes=len(jobs), per_dataset=120, max_workers=2,
                            max_paid_mllm_requests=len(jobs)*48, max_generated_tokens=len(jobs)*48*2048,
                            max_jev_inference_calls=80*12*3,
                            note='Paid cap includes in-episode interface recovery; no model preflight calls. Local health/tool calls excluded.'),
                inference_policy='Each trajectory attempted once; pending/failed attempts never retried or resampled',
                metrics=dict(AVDN='primary square bbox IoU > 0.4; secondary > 0.3',
                             CityNav='final predicted center distance <=20 m',
                             actual_stop='Separate distance from actual_stop_xy_m to goal_xy_m; never substituted for localization'),
                statistics='Paired episode bootstrap 95% descriptive CI and discordant counts; n=20 is not formal equivalence evidence')
    save(root / 'plan.json', plan)
    (root / 'plan.sha256').write_text(sha(root / 'plan.json'), encoding='ascii')
    validate_frozen(root)
    return plan


def validate_frozen(root, check_images=False):
    """Public-only validation: intentionally never opens evaluator_only.json."""
    root = Path(root).resolve()
    require((root / 'plan.sha256').read_text(encoding='ascii').strip() == sha(root / 'plan.json'),
            'Frozen plan/config changed')
    plan = read(root / 'plan.json')
    require(plan['schema'] == 'jev-online-paired-v1', 'Wrong experiment schema')
    require(sources(root / 'runtime') == plan['runtime_sources'], 'Frozen runtime changed')
    require(sources(Path(plan['source_root'])) == plan['runtime_sources'], 'Source code changed; refuse resume')
    require(sha(plan['harness_source']) == plan['harness_sha256'] == sha(root / 'harness.py'), 'Harness changed; refuse resume')
    require(environment() == plan['environment'], 'Python runtime/dependencies changed; refuse resume')
    require(sha(root / 'jev_service_manifest.json') == plan['service_file_sha256'], 'Frozen service manifest changed')
    require(sha(plan['service_source']) == plan['service_file_sha256'], 'Service manifest source changed')
    require(plan['settings']['limits'] == LIMITS and plan['settings']['thinking_budget'] == 0 and
            plan['settings']['jev_min_probability'] == .7, 'Experiment limits/policy changed')
    validate_service(read(root / 'jev_service_manifest.json'))
    expected_ids = {ds: [r['episode_id'] for r in plan['selected'] if r['dataset'] == ds] for ds in POOLS}
    require(plan['jobs'] == make_jobs(expected_ids), 'Job schedule changed')
    for row in plan['selected']:
        entry = read(root / row['public_input'])
        public_only(entry)
        require(digest(entry) == row['public_sha256'], 'Public input changed')
    if check_images:
        for path, info in plan['image_files'].items():
            require(Path(path).stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Map image changed')
    run_manifest = root / 'run_manifest.json'
    if run_manifest.exists():
        require(read(run_manifest)['plan_sha256'] == sha(root / 'plan.json'), 'Plan/config changed; refuse resume')
    return plan


def job_folder(root, job):
    return Path(root) / 'episodes' / job['dataset'] / job['model'] / job['arm'] / job['episode_id']


def job_state(folder):
    folder = Path(folder)
    if (folder / 'attempt.json').exists():
        return read(folder / 'attempt.json')['status']
    return 'untracked' if folder.exists() and any(folder.iterdir()) else 'unstarted'


def claim(folder, job, plan_hash):
    folder = Path(folder)
    state = job_state(folder)
    if state != 'unstarted':
        return False
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(status='pending', job_id=job['job_id'], plan_sha256=plan_hash,
                  pid=os.getpid(), started_unix=time.time())
    try:
        with (folder / 'attempt.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream)
    except FileExistsError:
        return False
    return True


def sanitized(text, key=None):
    text = str(text)
    for secret in [key] + [os.environ.get(name) for name in SECRET_ENVS]:
        if secret:
            text = text.replace(secret, '[REDACTED]')
    return text


def process_handle(pid):
    """A Windows synchronize-only handle avoids PID reuse during parent watching."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return kernel, None
        raise OSError(error, 'Cannot inspect process lifetime')
    return kernel, handle


def process_alive(pid):
    if os.name == 'nt':
        kernel, handle = process_handle(pid)
        if handle is None:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def guard_parent(pid):
    """Exit if the owning launcher disappears; pending requests are never replayed."""
    if os.name == 'nt':
        kernel, handle = process_handle(pid)
        if handle is None:
            raise RuntimeError('Parent already exited')
        def watch():
            result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            kernel.CloseHandle(handle)
            if result == 0:
                os._exit(70)
    else:
        def watch():
            while os.getppid() == pid:
                time.sleep(.25)
            os._exit(70)
    threading.Thread(target=watch, name='launcher-lifetime', daemon=True).start()


def infrastructure_failure(root, job):
    folder = job_folder(root, job)
    attempt = read(folder / 'attempt.json') if (folder / 'attempt.json').exists() else {}
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
    return bool(attempt.get('error') or (attempt.get('status') == 'failed' and not result)
                or result.get('jev_error_count', 0)
                or any(result.get('tool_capabilities', {}).get(name) == 'unavailable' for name in ('sam3', 'dino'))
                or any(call.get('status') in ('error', 'unavailable') for call in result.get('tool_calls', []))
                or (attempt.get('status') == 'failed' and any(x in str(result.get('termination', '')).lower()
                    for x in ('http', 'timeout', 'request_error', 'connection', 'identity'))))


def dispatch_jobs(jobs, launch, on_complete, should_stop, max_workers):
    """Bound the submitted queue and leave untouched jobs unstarted on infra failure."""
    pending_jobs = iter(jobs)
    completed, stopped = [], False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        active = {}
        def fill():
            while not stopped and len(active) < max_workers:
                job = next(pending_jobs, None)
                if job is None:
                    break
                active[pool.submit(launch, job)] = job
        fill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                value = future.result()
                completed.append(value)
                on_complete(value, completed)
                stopped = stopped or should_stop(job, value)
            fill()
    return completed, stopped


def healthcheck(plan, include_grounders=True, include_jev=True):
    import requests
    settings = plan['settings']
    result = {}
    targets = [('jev', settings['jev_url'])] if include_jev else []
    if include_grounders:
        targets += [('sam3', settings['sam3_url']), ('dino', settings['dino_url'])]
    for backend, url in targets:
        response = requests.get(url.rstrip('/') + '/health', timeout=5)
        response.raise_for_status()
        info = response.json()
        if backend == 'jev':
            require(info.get('status') == 'ok' and info.get('model') == JEV_MODEL and
                    info.get('service_manifest_sha256') == plan['service_content_sha256'],
                    'Jev health does not match the frozen 4B service manifest')
        else:
            require(info.get('backend') == backend and info.get('ready'), 'Grounder not ready: ' + backend)
        result[backend] = info
    return result


def worker(root, job_id):
    root = Path(root).resolve()
    plan = validate_frozen(root)
    job = next((x for x in plan['jobs'] if x['job_id'] == job_id), None)
    require(job is not None, 'Unknown frozen job')
    folder = job_folder(root, job)
    key = os.environ.get('DASHSCOPE_API_KEY')
    require(bool(key), 'Missing process DASHSCOPE_API_KEY')
    if not claim(folder, job, sha(root / 'plan.json')):
        return dict(job_id=job_id, status=job_state(folder), skipped=True)
    attempt = read(folder / 'attempt.json')
    try:
        entry = read(root / job['public_input'])
        image_path = str(Path(entry['environment']['image_path']).resolve())
        require(sha(image_path) == plan['image_files'][image_path]['sha256'], 'Map image changed')
        # A fresh worker imports ONLY frozen navigation packages, no evaluator.
        for name in list(sys.modules):
            if name.split('.')[0] in POOLS:
                del sys.modules[name]
        sys.path.insert(0, str(root / 'runtime'))
        module = importlib.import_module(job['dataset'] + '.run')
        require(Path(module.__file__).resolve().is_relative_to(root / 'runtime'), 'Live runtime import refused')
        settings = plan['settings']
        current_health = healthcheck(plan, include_jev=job['arm'] == 'jev')
        first_health = read(root / 'tool_health.json')
        require(all(value == first_health.get(backend) for backend, value in current_health.items()),
                'A service identity/configuration changed after the run started')
        save(folder / 'service_health.json', current_health)
        result = module.run_episode(entry, folder, key=key, model=job['model'],
            url=settings['endpoint'], thinking_budget=0, max_rounds=12, max_tools=2, max_requests=48,
            direction_mode='staged', sam3_url=settings['sam3_url'], dino_url=settings['dino_url'],
            locate_anything_url=None, review_backend=job['arm'],
            jev_url=settings['jev_url'], jev_model=JEV_MODEL, jev_min_probability=.7,
            jev_service_manifest=root / 'jev_service_manifest.json', **settings['observation'][job['dataset']])
        require((folder / 'result.json').exists(), 'Runtime returned without persisted result')
        require(result.get('model_requests', 0) <= 48, 'Runtime exceeded request budget')
        journals = list((folder / 'requests').glob('*.json')) + list((folder / 'tools').glob('*-jev-review.json'))
        pending = [str(p.relative_to(folder)) for p in journals if read(p).get('status') == 'pending']
        attempt.update(status='complete' if result.get('completed') and not pending else 'failed',
                       termination=result.get('termination'), pending_journals=pending,
                       result_sha256=sha(folder / 'result.json'))
    except Exception as exc:
        attempt.update(status='failed', error=type(exc).__name__, detail=sanitized(exc, key))
    attempt['finished_unix'] = time.time()
    save(folder / 'attempt.json', attempt)
    return dict(job_id=job_id, status=attempt['status'], termination=attempt.get('termination'))


@contextmanager
def run_lock(root):
    """OS-held lock releases on crash, so no stale-file deletion or live-run restart."""
    path = Path(root) / 'run.lock'
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another run currently holds the experiment lock') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(root=ROOT, max_workers=2, ask_key=False):
    require(max_workers in (1, 2), 'max_workers must be 1 or 2')
    root = Path(root).resolve()
    plan = validate_frozen(root, check_images=True)
    with run_lock(root):
        for job in plan['jobs']:
            folder = job_folder(root, job)
            if job_state(folder) == 'pending':
                pid = read(folder / 'attempt.json').get('pid')
                require(not pid or not process_alive(pid), 'A previous worker is still alive; refuse resume')
        frozen = dict(plan_sha256=sha(root / 'plan.json'), max_workers=max_workers,
                      timeout_seconds=7200, protocol='one-attempt-parent-guard-bounded-dispatch-v2')
        if (root / 'run_manifest.json').exists():
            require(read(root / 'run_manifest.json') == frozen, 'Run settings changed; refuse resume')
        else:
            save(root / 'run_manifest.json', frozen)
        jobs = [j for j in plan['jobs'] if job_state(job_folder(root, j)) == 'unstarted']
        if not jobs:
            return dict(launched=0, note='No unstarted jobs; pending/failed jobs are never retried')
        key = next((os.environ[k] for k in SECRET_ENVS if os.environ.get(k)), None)
        if not key and ask_key:
            key = getpass.getpass('API key (hidden, child process environment only): ')
        require(bool(key), 'Set process DASHSCOPE_API_KEY or use --ask-key')
        current_health = healthcheck(plan)
        if (root / 'tool_health.json').exists():
            require(read(root / 'tool_health.json') == current_health,
                    'Service identities/configuration changed; refuse resume')
        else:
            save(root / 'tool_health.json', current_health)
        env = os.environ.copy()
        for name in SECRET_ENVS:
            env.pop(name, None)
        env.update(DASHSCOPE_API_KEY=key, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        def launch(job):
            folder = job_folder(root, job)
            try:
                completed = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'worker',
                    '--root', str(root), '--job-id', job['job_id'], '--parent-pid', str(os.getpid())], cwd=root / 'runtime', env=env,
                    capture_output=True, encoding='utf-8', errors='replace', timeout=7200)
                log = sanitized(completed.stdout + '\n' + completed.stderr, key)
                error = None if completed.returncode == 0 else 'worker_exit_' + str(completed.returncode)
            except subprocess.TimeoutExpired:
                log, error = 'Worker timeout; no retry issued.', 'worker_timeout'
            except Exception as exc:
                log, error = sanitized(exc, key), type(exc).__name__
            log_path = root / 'logs' / (job['job_id'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log, encoding='utf-8')
            if error and job_state(folder) not in TERMINAL:
                attempt = read(folder / 'attempt.json') if (folder / 'attempt.json').exists() else dict(
                    job_id=job['job_id'], plan_sha256=frozen['plan_sha256'])
                attempt.update(status='failed', error=error, finished_unix=time.time())
                save(folder / 'attempt.json', attempt)
            return dict(job_id=job['job_id'], status=job_state(folder), launcher_error=error)
        def on_complete(value, completed_jobs):
            save(root / 'progress.json', dict(current_invocation=completed_jobs,
                all_status_counts=dict(Counter(job_state(job_folder(root, j)) for j in plan['jobs']))))
            print(json.dumps(value), flush=True)
        def should_stop(job, value):
            if infrastructure_failure(root, job):
                save(root / 'infrastructure_stop.json', dict(job_id=job['job_id'],
                    time_unix=time.time(), reason='Inspect infrastructure before resuming unstarted jobs'))
                return True
            return False
        completed_jobs, stopped = dispatch_jobs(jobs, launch, on_complete, should_stop, max_workers)
        return dict(launched=len(completed_jobs), completed_this_invocation=len(completed_jobs),
                    infrastructure_stop=stopped)


def evaluate_result(dataset, result, truth):
    final = result.get('final')
    stop = result.get('actual_stop_xy_m')
    stop_distance = math.dist(stop, truth['goal_xy_m']) if stop is not None else None
    metric, primary, secondary = None, False, False
    if final:
        if dataset == 'CityNav':
            metric = math.dist(final['center_xy_m'], truth['goal_xy_m'])
            primary = secondary = metric <= 20
        else:
            from shapely.geometry import Polygon
            a = Polygon(final['evaluation_polygon_xy_m'])
            b = Polygon(truth['evaluation_polygon_xy_m'])
            # The protocol produces convex evaluation boxes. Reject malformed
            # geometry instead of silently repairing it into a different target.
            require(a.is_valid and b.is_valid, 'Invalid evaluation polygon')
            union = a.union(b).area
            metric = a.intersection(b).area / union if union else 0.
            primary, secondary = metric > .4, metric > .3
    return dict(metric=metric, success=primary, secondary_success=secondary,
                actual_stop_distance_m=stop_distance,
                actual_stop_within20m=stop_distance is not None and stop_distance <= 20)


def bootstrap_mean(values, seed=SEED, repeats=10000):
    if not values:
        return None
    rng, n = random.Random(seed), len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return [samples[int(.025 * (repeats-1))], samples[int(.975 * (repeats-1))]]


def paired_stats(left, right, repeats=10000):
    require(len(left) == len(right), 'Paired lists have unequal length')
    differences = [int(b)-int(a) for a, b in zip(left, right)]
    gains = sum(not a and b for a, b in zip(left, right))
    losses = sum(a and not b for a, b in zip(left, right))
    discordant = gains + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses)+1)) / 2**discordant) if discordant else 1.
    return dict(n=len(left), left_successes=sum(left), right_successes=sum(right),
                difference_right_minus_left=sum(differences)/len(left) if left else None,
                difference_ci95=bootstrap_mean(differences, repeats=repeats),
                gained=gains, lost=losses, both_success=sum(a and b for a, b in zip(left, right)),
                both_failure=sum(not a and not b for a, b in zip(left, right)),
                discordant=discordant, exact_mcnemar_two_sided=p,
                inference='Exploratory paired episode statistics; no equivalence or population-wide claim')


def analyze(root=ROOT):
    root = Path(root).resolve()
    plan = validate_frozen(root)
    require(sha(root / 'evaluator_only.json') == plan['evaluator_sha256'], 'Evaluation labels changed')
    truth = read(root / 'evaluator_only.json')
    rows = []
    for job in plan['jobs']:
        folder = job_folder(root, job)
        state = job_state(folder)
        item = dict(job, status=state, success=False, secondary_success=False)
        if state in TERMINAL:
            attempt = read(folder / 'attempt.json')
            require(attempt['plan_sha256'] == sha(root / 'plan.json'), 'Attempt belongs to different plan')
            result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
            if attempt.get('result_sha256'):
                require(sha(folder / 'result.json') == attempt['result_sha256'], 'Completed result changed')
            try:
                item.update(evaluate_result(job['dataset'], result, truth[job['dataset']][job['episode_id']]))
            except (ValueError, KeyError, TypeError) as exc:
                item.update(metric=None, success=False, secondary_success=False, evaluation_error=type(exc).__name__)
            # Failed execution remains a failure, even if a partial result claims a target.
            if state == 'failed':
                item.update(success=False, secondary_success=False)
            requests = [read(p) for p in sorted((folder / 'requests').glob('*.json'))]
            jev = [read(p) for p in sorted((folder / 'tools').glob('*-jev-review.json'))]
            trace = read(folder / 'trace.json') if (folder / 'trace.json').exists() else {}
            calls = result.get('jev_calls', [])
            call_count = len(calls) if isinstance(calls, list) else calls if isinstance(calls, int) else 0
            capabilities = result.get('tool_capabilities', {})
            if not capabilities and (folder / 'tool_capabilities.json').exists():
                capabilities = read(folder / 'tool_capabilities.json')
            grounder = plan['settings']['observation'][job['dataset']]['observation_grounder']
            unavailable = [name for name in ('sam3', 'dino') if capabilities.get(name) == 'unavailable']
            item.update(termination=result.get('termination', attempt.get('error')),
                model_requests=len(requests), api_errors=sum(r.get('status') != 'ok' for r in requests),
                pending_requests=sum(r.get('status') == 'pending' for r in requests + jev),
                jev_requests=max(len(jev), call_count, result.get('jev_requests', 0)),
                jev_errors=max(sum(r.get('status') != 'ok' for r in jev), result.get('jev_error_count', 0),
                    sum(e.get('kind') == 'jev_response' and e.get('status') != 'ok' for e in trace.get('events', []))),
                review_mllm_requests=sum(r.get('context', {}).get('mode') == 'review' for r in requests),
                all_image_mllm_requests=sum(bool(r.get('assets')) for r in requests),
                tool_errors=sum(e.get('kind') == 'tool' and e.get('output', {}).get('status') != 'ok'
                                for e in trace.get('events', [])),
                tool_capabilities=capabilities, unavailable_planned_grounders=unavailable,
                observation_grounder_unavailable=capabilities.get(grounder) == 'unavailable',
                unknown_planned_grounders=[name for name in ('sam3', 'dino') if name not in capabilities],
                seconds=attempt.get('finished_unix', time.time())-attempt.get('started_unix', attempt.get('finished_unix', time.time())),
                result_path=str(folder / 'result.json'))
        rows.append(item)
    summaries, comparisons, gaps, gap_changes = [], [], [], []
    for ds in POOLS:
        subset = [r for r in rows if r['dataset'] == ds]
        index = {(r['episode_id'], r['model'], r['arm']): r for r in subset}
        ids = [r['episode_id'] for r in plan['selected'] if r['dataset'] == ds]
        for model in MODELS:
            for arm in ARMS:
                arm_rows = [index[(i, model, arm)] for i in ids]
                terminal = [r for r in arm_rows if r['status'] in TERMINAL]
                successes = sum(r['success'] for r in terminal)
                unfinished = len(ids)-len(terminal)
                summaries.append(dict(dataset=ds, model=model, arm=arm, planned=len(ids),
                    statuses=dict(Counter(r['status'] for r in arm_rows)), terminal=len(terminal),
                    successes=successes, sr=successes/len(ids) if not unfinished else None,
                    observed_sr_bounds=[successes/len(ids), (successes+unfinished)/len(ids)],
                    secondary_successes=sum(r['secondary_success'] for r in terminal),
                    secondary_sr=sum(r['secondary_success'] for r in terminal)/len(ids) if not unfinished else None,
                    jev_episode_coverage=sum(r.get('jev_requests', 0) > 0 for r in terminal),
                    model_requests=sum(r.get('model_requests', 0) for r in terminal),
                    api_errors=sum(r.get('api_errors', 0) for r in terminal),
                    jev_errors=sum(r.get('jev_errors', 0) for r in terminal),
                    tool_errors=sum(r.get('tool_errors', 0) for r in terminal),
                    grounder_unavailable_episodes=sum(bool(r.get('unavailable_planned_grounders')) for r in terminal),
                    observation_grounder_unavailable_episodes=sum(r.get('observation_grounder_unavailable', False) for r in terminal),
                    grounder_capability_unknown_episodes=sum(bool(r.get('unknown_planned_grounders')) for r in terminal),
                    actual_stop_distances_m=[r.get('actual_stop_distance_m') for r in terminal]))
            for left, right in [('mllm', 'mllm_fixed'), ('mllm_fixed', 'jev'), ('mllm', 'jev')]:
                paired = [i for i in ids if all(index[(i, model, a)]['status'] in TERMINAL for a in (left, right))]
                comparisons.append(dict(dataset=ds, model=model, left=left, right=right,
                    planned_pairs=len(ids), missing_pairs=len(ids)-len(paired),
                    primary=paired_stats([index[(i, model, left)]['success'] for i in paired],
                                         [index[(i, model, right)]['success'] for i in paired]),
                    secondary=paired_stats([index[(i, model, left)]['secondary_success'] for i in paired],
                                           [index[(i, model, right)]['secondary_success'] for i in paired])))
        for arm in ARMS:
            paired = [i for i in ids if all(index[(i, model, arm)]['status'] in TERMINAL for model in MODELS)]
            gaps.append(dict(dataset=ds, arm=arm, planned_pairs=len(ids), missing_pairs=len(ids)-len(paired),
                max_minus_flash=paired_stats([index[(i, MODELS[0], arm)]['success'] for i in paired],
                                             [index[(i, MODELS[1], arm)]['success'] for i in paired])))
        complete = [i for i in ids if all(index[(i, model, arm)]['status'] in TERMINAL
                                          for model in MODELS for arm in ARMS)]
        for baseline in ('mllm', 'mllm_fixed'):
            values = [(int(index[(i, MODELS[1], 'jev')]['success'])-int(index[(i, MODELS[0], 'jev')]['success']))-
                      (int(index[(i, MODELS[1], baseline)]['success'])-int(index[(i, MODELS[0], baseline)]['success']))
                      for i in complete]
            gap_changes.append(dict(dataset=ds, baseline=baseline, n=len(values),
                signed_gap_change_jev_minus_baseline=sum(values)/len(values) if values else None,
                ci95=bootstrap_mean(values), note='Difference in signed max-minus-flash gaps; a negative value is not by itself equivalence'))
    report = dict(schema='jev-online-analysis-v1', plan_sha256=sha(root / 'plan.json'),
                  all_terminal=all(r['status'] in TERMINAL for r in rows),
                  statuses=dict(Counter(r['status'] for r in rows)), rows=rows, summaries=summaries,
                  within_model_effects=comparisons, across_model_gaps=gaps, gap_changes=gap_changes,
                  limitations=['No formal equivalence conclusion from n=20 per dataset.',
                    'CityNav inference is conditional on the existing 50-episode pool and cached initialization.',
                    'Failures remain in terminal denominators; pending/unstarted pairs are explicitly incomplete.',
                    'Episode bootstrap does not account for dependence between episodes on the same map.',
                    'Unavailable or unknown grounder capabilities are reported separately from tool-call errors; infrastructure outages may confound tool effects.',
                    'Jev replaces candidate review only; remaining main MLLM still receives navigation images.'])
    save(root / 'analysis.json', report)
    text = ['# Paired Jev navigation experiment', '',
            'Status: ' + ('all 240 attempts terminal' if report['all_terminal'] else 'INCOMPLETE: no final success-rate claim'), '',
            'Each failed attempt is retained. Unstarted/pending attempts are not silently labelled failures. ',
            'AVDN primary: square bbox IoU > 0.4; secondary > 0.3. CityNav: predicted center <= 20 m.',
            'Actual stopping distance is recorded separately in analysis.json.', '',
            '| Dataset | MLLM | Review | Terminal/planned | Successes | SR | SR bounds if incomplete | Jev errors |',
            '| --- | --- | --- | ---: | ---: | ---: | --- | ---: |']
    for row in summaries:
        sr = f"{row['sr']:.1%}" if row['sr'] is not None else 'pending'
        bounds = ', '.join(f'{x:.1%}' for x in row['observed_sr_bounds'])
        text.append(f"| {row['dataset']} | {row['model']} | {row['arm']} | {row['terminal']}/{row['planned']} | {row['successes']} | {sr} | [{bounds}] | {row['jev_errors']} |")
    text += ['', 'Within-MLLM tool effects and max-minus-flash gaps include paired bootstrap 95% CIs, ',
             'gained/lost discordant counts and exact McNemar statistics in analysis.json.', '',
             'These are exploratory comparisons, not a formal equivalence or noninferiority conclusion. ',
             'The existing CityNav pool and cached initialization constrain generalization. ',
             'Remaining navigation MLLM calls still include images.']
    (root / 'analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'worker', 'run', 'analyze'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--jev-manifest', type=Path, default=SERVICE)
    parser.add_argument('--endpoint', default=DEFAULT_URL)
    parser.add_argument('--sam3-url', default='http://127.0.0.1:8765')
    parser.add_argument('--dino-url', default='http://127.0.0.1:8766')
    parser.add_argument('--jev-url', default='http://127.0.0.1:8770')
    parser.add_argument('--max-workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--ask-key', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.root, args.jev_manifest, args.endpoint, args.sam3_url, args.dino_url, args.jev_url)
        print(json.dumps(dict(root=str(args.root.resolve()), budget=plan['budget'],
                              eligible={d: p['eligible_size'] for d, p in plan['pools'].items()})))
    elif args.command == 'worker':
        if args.parent_pid:
            guard_parent(args.parent_pid)
        print(json.dumps(worker(args.root, args.job_id)))
    elif args.command == 'run':
        guard_parent(os.getppid())
        print(json.dumps(run(args.root, args.max_workers, args.ask_key)))
    else:
        report = analyze(args.root)
        print(json.dumps(dict(statuses=report['statuses'], all_terminal=report['all_terminal'],
                              analysis=str(args.root.resolve() / 'analysis.json'))))


if __name__ == '__main__':
    main()
