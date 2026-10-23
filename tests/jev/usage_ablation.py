"""Post-hoc development-only context/pixel sensitivity; never an online SR test.

prepare/verify do no inference. run refuses until the frozen online study is terminal.
No credential or evaluator labels are read during inference. Never retry a request.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_review as fixed

WORK = fixed.WORK
STUDY = WORK/'out/jev_study_20261022'
ROOT = STUDY/'usage_ablation_4b'
SOURCE = STUDY/'fixed_review_4b'
ONLINE = STUDY/'online'
MODEL = 'OneJev1.6-4B'
MANIFEST = STUDY/'services/OneJev1.6-4B-worker.manifest.json'
ARMS = ('full', 'compact', 'substituted_images')
KEEP = ('mode', 'instruction_raw', 'turns', 'initialization', 'initial_pose',
        'position_xy_m', 'heading_deg', 'coordinate_frame', 'named_landmarks',
        'memory', 'candidate', 'views')


def compact(context):
    # Preserve verbatim turns as well as raw instruction; they need not be redundant.
    return {k: copy.deepcopy(context[k]) for k in KEEP if k in context}


def image_key(item):
    return tuple(a['sha256'] for a in item['assets'])


def donor_map(cases):
    result = {}
    for item in cases:
        eligible = sorted((c for c in cases if c['dataset'] == item['dataset']
                           and c['episode_id'] != item['episode_id']
                           and image_key(c) != image_key(item)
                           and len(c['assets']) == len(item['assets'])), key=lambda c:c['case_id'])
        if not eligible:
            raise ValueError('No distinct within-dataset image donor')
        # Fixed mapping independent of labels and model outputs. Donor reuse is allowed.
        index = int(fixed.digest(item['case_id'])[:16], 16) % len(eligible)
        result[item['case_id']] = eligible[index]['case_id']
    return result


def transformed(item, arm, donor=None):
    out = copy.deepcopy(item)
    if arm == 'compact':
        out['context'] = compact(item['context'])
    elif arm == 'substituted_images':
        if donor is None or image_key(donor) == image_key(item):
            raise ValueError('A distinct donor is required')
        # Keep recipient view IDs/metadata/text exactly; replace only pixel files.
        for asset, replacement in zip(out['assets'], donor['assets'], strict=True):
            asset.update(image_path=replacement['image_path'], sha256=replacement['sha256'])
    elif arm != 'full':
        raise ValueError('Unknown arm')
    fixed.no_truth(out)
    return out


def prepare(root=ROOT):
    design, cases = fixed.validate_public(SOURCE)
    donors = donor_map(cases)
    by_id = {c['case_id']: c for c in cases}
    jobs = []
    for item in cases:
        for arm in ARMS:
            changed = transformed(item, arm, by_id[donors[item['case_id']]])
            jobs.append(dict(case_id=item['case_id'], arm=arm,
                             input_sha256=fixed.digest(changed)))
    random.Random(20261022).shuffle(jobs)
    plan = dict(schema='jev-usage-ablation-v1', source_public_sha256=design['public_sha256'],
        online_plan_sha256=fixed.sha(ONLINE/'plan.json'), model=MODEL,
        manifest_sha256=fixed.sha(MANIFEST), client_sha256=fixed.sha(WORK/'AVDN/tools/jev.py'),
        script_sha256=fixed.sha(__file__), compact_keep=list(KEEP), donors=donors, jobs=jobs,
        threshold=.7, seed=20261022,
        interpretation=['Post-hoc development diagnostics, not held-out navigation success.',
            'Compact drops feedback/navigation/tools/budgets; useful evidence may be removed.',
            'Image substitution creates inconsistent text/images; test output sensitivity only.',
            'Original geometric labels are invalid for substituted images; never score their accuracy.',
            'Cases share source episodes; counts are not independent trajectory replicates.',
            'No threshold selection or online protocol changes.'])
    fixed.save(root/'plan.json', plan, True)
    print(json.dumps(dict(status='prepared', jobs=len(jobs), inference_started=False)))


def verify(root=ROOT):
    plan = fixed.read(root/'plan.json')
    design, cases = fixed.validate_public(SOURCE)
    checks = [(plan['source_public_sha256'], design['public_sha256']),
              (plan['online_plan_sha256'], fixed.sha(ONLINE/'plan.json')),
              (plan['manifest_sha256'], fixed.sha(MANIFEST)),
              (plan['client_sha256'], fixed.sha(WORK/'AVDN/tools/jev.py')),
              (plan['script_sha256'], fixed.sha(__file__))]
    if any(a != b for a,b in checks) or plan['compact_keep'] != list(KEEP):
        raise ValueError('Frozen ablation identity changed')
    expected = {(c['case_id'], a) for c in cases for a in ARMS}
    actual = [(j['case_id'], j['arm']) for j in plan['jobs']]
    if len(actual) != len(expected) or set(actual) != expected or plan['donors'] != donor_map(cases):
        raise ValueError('Ablation schedule/donor mismatch')
    by_id = {c['case_id']: c for c in cases}
    for job in plan['jobs']:
        item = transformed(by_id[job['case_id']], job['arm'], by_id[plan['donors'][job['case_id']]])
        if fixed.digest(item) != job['input_sha256']:
            raise ValueError('Ablation request identity changed')
    return plan, by_id


def require_terminal(online, expected_hash):
    if fixed.sha(online/'plan.json') != expected_hash:
        raise ValueError('Online plan changed')
    jobs = fixed.read(online/'plan.json')['jobs']
    if len(jobs) != 240:
        raise ValueError('Expected the complete 240-job online design')
    for job in jobs:
        folder = online/'episodes'/job['dataset']/job['model']/job['arm']/job['episode_id']
        path = folder/'attempt.json'
        if not path.exists() or fixed.read(path).get('status') not in ('complete','failed'):
            raise RuntimeError('Primary online experiment is still active; no ablation inference allowed')


def request_inputs(plan, cases, job):
    from AVDN.tools.jev import build_review_payload
    item = transformed(cases[job['case_id']], job['arm'], cases[plan['donors'][job['case_id']]])
    views = [dict(next(v for v in item['context']['views'] if v['view_id']==a['view_id']),
                  image_path=a['image_path']) for a in item['assets']]
    _, journal, assets = build_review_payload(item['context'], views, MODEL)
    if journal['state']['context'] != item['context'] or assets != item['assets']:
        raise ValueError('Shared builder changed planned inputs')
    return item, views, journal


def run(root=ROOT):
    plan, cases = verify(root)
    require_terminal(ONLINE, plan['online_plan_sha256'])
    from AVDN.tools.jev import Jev
    client = Jev(url='http://127.0.0.1:8770', expected_model=MODEL,
                 min_probability=.7, service_manifest=MANIFEST)
    for job in plan['jobs']:
        require_terminal(ONLINE, plan['online_plan_sha256'])
        item, views, _ = request_inputs(plan, cases, job)
        path = root/'responses'/job['arm']/(job['case_id']+'.json')
        # The shared client rejects pending requests and mismatching signatures.
        answer = client.verify(item['context'], views, path)
        print(json.dumps(dict(case_id=job['case_id'], arm=job['arm'],
                              status=answer['status'], verdict=answer['verdict'])), flush=True)
        if answer['status'] != 'ok':
            raise RuntimeError('Local ablation error: stop without retry')


def analyze(root=ROOT):
    plan, cases = verify(root)
    labels = fixed.read(SOURCE/'evaluator_only.json')
    if fixed.digest(labels) != fixed.read(SOURCE/'design.json')['evaluator_sha256']:
        raise ValueError('Development labels changed')
    answers, rows, response_hashes = {}, [], {}
    for job in plan['jobs']:
        path = root/'responses'/job['arm']/(job['case_id']+'.json')
        if path.exists():
            answer = fixed.read(path)
            _, _, journal = request_inputs(plan, cases, job)
            if answer.get('input') != journal or answer.get('model') != MODEL or answer.get('min_probability') != .7:
                raise ValueError('Saved response input/model/policy differs from plan')
            if answer.get('service_manifest',{}).get('sha256') != plan['manifest_sha256']:
                raise ValueError('Saved response service identity differs from plan')
            if answer.get('status') == 'ok':
                from AVDN.tools.jev import validate_answer
                parsed = validate_answer(answer['raw'], MODEL, .7)
                if any(answer.get(k) != v for k,v in parsed.items()):
                    raise ValueError('Saved parsed answer disagrees with raw response')
            answers[(job['case_id'],job['arm'])] = answer
            response_hashes[str(path.relative_to(root))] = fixed.sha(path)
    for split in sorted({c['split'] for c in cases.values()}):
        ids = [c['case_id'] for c in cases.values() if c['split']==split]
        for arm in ARMS:
            valid = [(i, answers[(i,arm)]) for i in ids if answers.get((i,arm),{}).get('status')=='ok']
            row = dict(split=split, arm=arm, planned=len(ids), ok=len(valid),
                       choices=dict(Counter(a['choice'] for _,a in valid)),
                       verdicts=dict(Counter(a['verdict'] for _,a in valid)))
            if arm != 'substituted_images':
                for field in ('choice','verdict'):
                    row[field+'_geometry_proxy'] = fixed.classify_counts([
                        (labels[i]['proposal_score']['correct'],a[field]) for i,a in valid])
            paired = [(a, answers[(i,'full')]) for i,a in valid
                      if answers.get((i,'full'),{}).get('status')=='ok']
            row['paired_with_full'] = dict(n=len(paired),
                changed_choices=sum(a['choice']!=b['choice'] for a,b in paired),
                changed_verdicts=sum(a['verdict']!=b['verdict'] for a,b in paired),
                mean_max_probability_delta=(sum(max(abs(a['probabilities'][k]-b['probabilities'][k])
                    for k in a['probabilities']) for a,b in paired)/len(paired) if paired else None))
            rows.append(row)
    result = dict(complete=len(answers)==len(plan['jobs']) and all(a['status']=='ok' for a in answers.values()),
                  plan_sha256=fixed.sha(root/'plan.json'), response_sha256=response_hashes,
                  rows=rows, limitations=plan['interpretation'])
    fixed.save(root/'analysis.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','verify','run','analyze'))
    args = parser.parse_args()
    globals()[args.command]()
