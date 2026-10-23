"""Frozen candidate-review diagnostic. This is not a navigation success-rate test.

prepare reads evaluator truth only to select the first qualifying/non-qualifying
review per source trajectory. run reads public_requests.jsonl only. analyze is
offline and joins labels after inference. No paid model API is called here.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import html
import importlib
import json
import math
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
ROOT = WORK / 'out/jev_study_20261022/fixed_review'
MECHANISM = WORK / 'out/grounding_mechanism20_20261020'
CAPACITY = WORK / 'out/avdn_mllm_capacity20_20261019'
THRESHOLD = 0.7
DECISION_POLICY = dict(min_probability=THRESHOLD, decision='argmax if max(probabilities) >= 0.7, otherwise uncertain',
                       official_confidence='record and validate only; never used as the threshold',
                       calibration='No domain calibration; do not interpret as calibrated correctness probability')
PRIVATE = {'goal_xy_m', 'goal_polygon_xy_m', 'evaluation_polygon_xy_m',
           'gt_path_corners', 'source_row', 'proposal_score', 'correct', 'label',
           'original_answer', 'evaluator_proposal_score'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def no_truth(value):
    if isinstance(value, dict):
        if PRIVATE.intersection(value):
            raise ValueError('Evaluator information in public request: ' + ','.join(PRIVATE.intersection(value)))
        for child in value.values():
            no_truth(child)
    elif isinstance(value, list):
        for child in value:
            no_truth(child)


def proposal_score(dataset, candidate, views, truth):
    """The exact existing diagnostic metrics; never used during run."""
    from AVDN.tools.geometry import project, square_bbox, world
    view = views[candidate['view_id']]
    bbox = candidate['bbox']
    if (len(bbox) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                              or not math.isfinite(x) for x in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000)):
        raise ValueError('Invalid candidate bbox')
    if dataset == 'CityNav':
        point = world(view, [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])[0]
        value = math.dist(point, truth['goal_xy_m'])
        return {'value': value, 'correct': value <= 20, 'metric': 'center_distance_m_le20'}
    from shapely import make_valid
    from shapely.geometry import Polygon
    prediction = make_valid(Polygon(project(view['pixel_to_world'], square_bbox(
        [x * (view['image_size'][0] - 1) / 1000 for x in bbox]))))
    target = make_valid(Polygon(truth['evaluation_polygon_xy_m']))
    union = prediction.union(target).area
    value = float(prediction.intersection(target).area / union) if union else 0.0
    return {'value': value, 'correct': value > .4, 'metric': 'square_bbox_iou_gt04'}


def parse_verdict(text):
    try:
        text = text.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        obj = json.loads(text)
        if obj.get('action') != 'review' or obj.get('verdict') not in ('confirm', 'reject', 'uncertain'):
            return 'invalid'
        return obj['verdict']
    except (ValueError, AttributeError, TypeError, IndexError):
        return 'invalid'


def request_case(path, dataset, episode_id, source_model, split):
    path = Path(path)
    req = read(path)
    if req.get('status') != 'ok' or req['context'].get('mode') != 'review':
        raise ValueError('Not a successful recorded review: ' + str(path))
    if req['context']['candidate'].get('target') != 'final':
        raise ValueError('Only final-target reviews are eligible')
    # IDs deliberately do not encode positive/negative evaluator classes.
    ident = 'review-' + digest({'path': str(path.resolve()), 'sha256': sha(path)})[:20]
    item = dict(case_id=ident, dataset=dataset, episode_id=episode_id, split=split,
                source_model=source_model, source_request=str(path.resolve()),
                source_sha256=sha(path), original_system=req['system_prompt'],
                context=copy.deepcopy(req['context']), assets=copy.deepcopy(req['assets']))
    no_truth(item)
    for asset in item['assets']:
        if sha(asset['image_path']) != asset['sha256']:
            raise ValueError('Original RGB bytes changed: ' + asset['image_path'])
    return item, req


def historical_answer(record, path, kind, score=None):
    result = dict(model=record.get('model'), thinking_budget=record.get('thinking', record.get('thinking_budget', 0)),
                  origin=kind, path=str(path), sha256=sha(path), status=record.get('status'),
                  verdict=parse_verdict(record.get('response', {}).get('text', '')),
                  raw_answer=record.get('response', {}).get('text', ''))
    if score is not None:
        result['refined_score'] = score
    return result


def first_reviews(path, dataset, truth):
    trace = read(Path(path) / 'trace.json')
    views = {v['view_id']: v for v in trace['views']}
    seen = set()
    for request in sorted((Path(path) / 'requests').glob('*.json')):
        req = read(request)
        candidate = req.get('context', {}).get('candidate')
        if (req.get('status') != 'ok' or req.get('context', {}).get('mode') != 'review'
                or not candidate or candidate.get('target') != 'final'):
            continue
        scored = proposal_score(dataset, candidate, views, truth)
        if scored['correct'] in seen:
            continue
        seen.add(scored['correct'])
        yield request, req, scored, views


def prepare(root=ROOT):
    root = Path(root)
    cases, labels, selection = [], {}, []
    original_city = read(MECHANISM / 'review_cases.json')
    old_truth = read(MECHANISM / 'evaluator_only.json')
    if len(original_city) != 14:
        raise ValueError('Expected the frozen 14 CityNav primary contexts')
    for original in original_city:
        item, req = request_case(original['source_request'], 'CityNav', original['episode_id'],
                                 original['source_model'], 'citynav_primary14')
        if item['source_sha256'] != original['source_sha256'] or item['context'] != original['context'] or item['assets'] != original['assets']:
            raise ValueError('Historical CityNav context changed')
        gt = old_truth[original['case_id']]
        scored = proposal_score('CityNav', item['context']['candidate'], gt['views'], gt['gt'])
        if scored['correct'] != gt['proposal_score']['correct'] or abs(scored['value'] - gt['proposal_score']['value']) > 1e-8:
            raise ValueError('Historical CityNav label no longer reproduces')
        cases.append(item)
        labels[item['case_id']] = dict(proposal_score=scored, historical=[historical_answer(req, item['source_request'], 'source_navigation')],
                                      historical_case_id=original['case_id'])
    targeted = read(MECHANISM / 'targeted_review_plan.json')
    path = Path(targeted['source_request'])
    item, req = request_case(path, 'CityNav', path.parents[1].name, read(path)['model'], 'citynav_targeted_posthoc1')
    if item['source_sha256'] != targeted['source_sha256']:
        raise ValueError('Targeted source changed')
    cases.append(item)
    labels[item['case_id']] = dict(proposal_score=dict(targeted['evaluator_proposal_score'], metric='center_distance_m_le20'),
                                  historical=[historical_answer(req, path, 'source_navigation')],
                                  historical_case_id=path.parents[1].name + '-' + path.stem)
    truth_by_id = {r['episode_id']: r for r in read(CAPACITY / 'evaluator_only.json')}
    comparison = read(CAPACITY / 'comparison.json')
    chosen = read(CAPACITY / 'final_selection.json')['artifacts']
    if len(comparison['rows']) != 60:
        raise ValueError('Expected all 20 trajectories by 3 source models')
    for row in sorted(comparison['rows'], key=lambda x: (x['episode_id'], x['model'])):
        path = Path(row['artifact_directory'])
        if Path(chosen[row['model'] + '/' + row['episode_id']]) != path:
            raise ValueError('Historical recovery selection disagrees')
        retained = []
        for request, req, scored, views in first_reviews(path, 'AVDN', truth_by_id[row['episode_id']]):
            item, _ = request_case(request, 'AVDN', row['episode_id'], row['model'], 'avdn_capacity20x3')
            cases.append(item)
            labels[item['case_id']] = dict(proposal_score=scored,
                                          historical=[historical_answer(req, request, 'source_navigation')])
            retained.append(item['case_id'])
        selection.append(dict(dataset='AVDN', episode_id=row['episode_id'], source_model=row['model'],
                              source_directory=str(path), retained=retained,
                              absence_reason=None if retained else 'No successful final-target review request'))
    # Reuse each recorded MLLM diagnostic only when all original review inputs match.
    historical_ids = {v.get('historical_case_id'): k for k, v in labels.items() if v.get('historical_case_id')}
    by_id = {r['case_id']: r for r in cases}
    for path in sorted((MECHANISM / 'requests').rglob('*.json')):
        record = read(path)
        job = record.get('input', {})
        ident = historical_ids.get(job.get('case_id'))
        if ident is None or job.get('task') not in ('review', 'targeted_review'):
            continue
        item = by_id[ident]
        if (job['system'] != item['original_system'] or job['context'] != item['context']
                or job['assets'] != item['assets']):
            raise ValueError('Prior MLLM diagnostic input mismatch: ' + str(path))
        labels[ident]['historical'].append(historical_answer(record, path, 'fixed_input_mllm'))
    for item in cases:
        # Hash the full request identity, including image order and original context.
        item['input_sha256'] = digest({k: item[k] for k in ('context', 'assets', 'original_system')})
    if len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('Duplicate review provenance')
    design = dict(schema='jev-fixed-review-v1', threshold=THRESHOLD,
        threshold_policy='Preregistered 0.7 before OneJev inference; no test-label calibration.',
        selection='CityNav original 14 contexts plus one separately reported post-hoc case. AVDN all 20 trajectories x 3 selected clean source runs: first correct and first incorrect final-target review per source model/trajectory, independent of original verdict.',
        public_sha256=digest(cases), evaluator_sha256=digest(labels),
        case_counts=dict(Counter(r['split'] for r in cases)), source_avdn_trajectories=60,
        image_policy='Exact original ordered image bytes, including original review markings; no new crop or image edit.',
        limitations=['This is candidate review, not online navigation SR or evidence of replacing MLLMs.',
                    'AVDN sample was outcome-stratified; CityNav primary set contains six development trajectories.',
                    'Geometric qualification (AVDN square IoU >0.4; CityNav center <=20m) is not exact visual identity.',
                    'Same original context/RGB is preserved, but OneJev uses its tool classification prompt rather than the original MLLM system prompt.',
                    'Repeated views and multiple source models within trajectories are correlated; no IID significance claims.',
                    'Original MLLM could refine a box. This confusion matrix evaluates its verdict on the ORIGINAL candidate only.',
                    'Uncertain and malformed/failed results are reported separately, never silently converted to reject.'])
    save(root / 'design.json', design, True)
    save(root / 'evaluator_only.json', labels, True)
    save(root / 'selection_provenance.json', selection, True)
    save(root / 'decision_policy.json', DECISION_POLICY, True)
    text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in cases)
    destination = root / 'public_requests.jsonl'
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Frozen public requests changed')
    if not destination.exists():
        destination.write_text(text, encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'root': str(root), **design['case_counts'],
                      'historical_answers': sum(len(x['historical']) for x in labels.values())}))
    return design


def validate_public(root):
    design = read(root / 'design.json')
    if read(root / 'decision_policy.json') != DECISION_POLICY:
        raise ValueError('Frozen decision policy changed')
    cases = lines(root / 'public_requests.jsonl')
    no_truth(cases)
    if digest(cases) != design['public_sha256'] or design['threshold'] != THRESHOLD:
        raise ValueError('Frozen public design changed')
    for item in cases:
        if digest({k: item[k] for k in ('context', 'assets', 'original_system')}) != item['input_sha256']:
            raise ValueError('Request identity changed')
        for asset in item['assets']:
            if sha(asset['image_path']) != asset['sha256']:
                raise ValueError('Image identity changed')
    return design, cases


def classify_counts(rows):
    counts = Counter(TP=0, FN=0, FP=0, TN=0, uncertain_positive=0,
                     uncertain_negative=0, invalid_positive=0, invalid_negative=0)
    for positive, verdict in rows:
        if verdict == 'confirm':
            counts['TP' if positive else 'FP'] += 1
        elif verdict == 'reject':
            counts['FN' if positive else 'TN'] += 1
        elif verdict == 'uncertain':
            counts['uncertain_positive' if positive else 'uncertain_negative'] += 1
        else:
            counts['invalid_positive' if positive else 'invalid_negative'] += 1
    pos = sum(bool(p) for p, _ in rows)
    neg = len(rows) - pos
    return dict(counts, n=len(rows), positive=pos, negative=neg,
                rejection_of_qualifying_rate=counts['FN'] / pos if pos else None,
                nonconfirmation_of_qualifying_rate=(pos-counts['TP']) / pos if pos else None,
                false_confirmation_rate=counts['FP'] / neg if neg else None,
                decisive_count=counts['TP']+counts['TN']+counts['FP']+counts['FN'])


def analyze(root=ROOT):
    root = Path(root)
    design, cases = validate_public(root)
    labels = read(root / 'evaluator_only.json')
    if digest(labels) != design['evaluator_sha256']:
        raise ValueError('Evaluator labels changed')
    groups, rows = defaultdict(list), []
    for item in cases:
        label = labels[item['case_id']]
        positive = label['proposal_score']['correct']
        for answer in label['historical']:
            subset = item['source_model'] if answer['origin'] == 'source_navigation' else 'all'
            key = (item['split'], answer['origin'], answer['model'], answer['thinking_budget'], subset)
            groups[key].append((positive, answer['verdict']))
        path = root / 'responses' / (item['case_id'] + '.json')
        record = read(path) if path.exists() else None
        verdict = record.get('verdict', 'invalid') if record else 'not_run'
        if record:
            if record['input_sha256'] != item['input_sha256']:
                raise ValueError('Response belongs to a different input')
            groups[(item['split'], 'OneJev', 'OneJev', 0, 'all')].append((positive, verdict))
            groups[(item['split'], 'OneJev', 'OneJev', 0, item['source_model'])].append((positive, verdict))
        rows.append(dict(case_id=item['case_id'], dataset=item['dataset'], split=item['split'],
                         episode_id=item['episode_id'], source_model=item['source_model'],
                         proposal_score=label['proposal_score'], jev_verdict=verdict,
                         jev_status=record.get('status') if record else 'not_run',
                         historical=label['historical']))
    summaries = [dict(split=k[0], method=k[1], model=k[2], thinking_budget=k[3], source_subset=k[4], **classify_counts(v))
                 for k, v in sorted(groups.items())]
    result = dict(status='complete' if all(r['jev_status'] == 'ok' for r in rows) else 'incomplete',
                  candidate_count=len(cases), summaries=summaries, rows=rows,
                  limitations=design['limitations'], threshold=THRESHOLD,
                  decision_policy=DECISION_POLICY,
                  metric='Candidate verdict versus offline geometric qualification; NOT navigation SR')
    save(root / 'comparison.json', result)
    def percent(value):
        return '-' if value is None else f'{100*value:.1f}%'
    body = ''
    for row in summaries:
        cells = [row['split'], row['method']+' / '+row['model']+' / t'+str(row['thinking_budget']),
                 row['source_subset'], row['n'], row['positive'], row['negative'],
                 row['TP'], row['FN'], row['FP'], row['TN'],
                 str(row['uncertain_positive'])+'/'+str(row['uncertain_negative']),
                 str(row['invalid_positive'])+'/'+str(row['invalid_negative']),
                 percent(row['rejection_of_qualifying_rate']), percent(row['nonconfirmation_of_qualifying_rate'])]
        body += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Jev 固定候选复核</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#203047}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf4f8}p{max-width:1200px}</style><h1>Jev 固定候选复核</h1>'
    page += '<p>状态：'+html.escape(result['status'])+'。76 个原始双图上下文；保留原图像字节及候选。此表是候选判定诊断，<strong>不是导航成功率，也不能证明可替换 MLLM</strong>。</p>'
    page += '<p>正类只表示几何指标合格：CityNav 候选中心距标注点 ≤20m；AVDN 候选方框转换后的 IoU &gt;0.4。邻近错误物体也可能通过 CityNav 距离门槛，不能把标签称为严格语义身份真值。MLLM 的框修正收益未计入这张判定表。</p>'
    page += '<p>预声明门槛：最大类别概率 ≥0.7 才采用 argmax，否则 uncertain；官方 confidence 仅记录。不使用测试集调阈值，概率未经本域校准。uncertain 与服务/格式错误单列。来源子集 all 与按源模型分组不能累加；只比较相同子集。</p>'
    headers = ['集合','方法 / 模型 / thinking','候选来源子集','N','几何正','几何负','TP','FN','FP','TN','uncertain 正/负','错误 正/负','拒绝合格率','未确认合格率']
    page += '<div style="overflow:auto"><table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+body+'</table></div>'
    page += '<p>CityNav 定向 1 例为事后机制案例，单独展示。AVDN 不同源模型产生不同候选，不能直接把各源模型小计当作同输入模型排名。所有具体候选、回答与原请求路径见 <a href="comparison.json">comparison.json</a>；设计见 <a href="design.json">design.json</a>。</p></html>'
    (root / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    return result


def run(root=ROOT, url='http://127.0.0.1:8770', limit=None, model=None, service_manifest=None):
    """Call the shared integration client. No evaluator file is read in this path."""
    root = Path(root)
    design, cases = validate_public(root)
    shared = importlib.import_module('AVDN.tools.jev')
    if not model:
        raise ValueError('Specify the exact /health model identity using --model')
    import requests
    health = requests.get(url.rstrip('/')+'/health', timeout=5)
    health.raise_for_status()
    health = health.json()
    if health.get('status') != 'ok' or health.get('model') != model:
        raise RuntimeError('OneJev service is not ready or model identity differs')
    client = shared.Jev(url=url, expected_model=model, min_probability=THRESHOLD, service_manifest=service_manifest)
    runtime = dict(public_sha256=design['public_sha256'], threshold=THRESHOLD, url=url,
                   shared_client_sha256=sha(shared.__file__), script_sha256=sha(__file__),
                   expected_model=model, service_manifest=client.manifest, decision_policy=DECISION_POLICY)
    save(root / 'run_manifest.json', runtime, True)
    for item in cases[:limit] if limit is not None else cases:
        destination = root / 'responses' / (item['case_id'] + '.json')
        if destination.exists():
            old = read(destination)
            if old['input_sha256'] != item['input_sha256']:
                raise ValueError('Resume input mismatch')
            if old['status'] == 'pending':
                raise RuntimeError('Uncertain request pending; do not automatically resend')
            continue
        record = dict(case_id=item['case_id'], input_sha256=item['input_sha256'], status='pending',
                      threshold=THRESHOLD, endpoint=url, started_unix=time.time())
        save(destination, record)
        try:
            views = []
            for asset in item['assets']:
                view = next(v for v in item['context']['views'] if v['view_id'] == asset['view_id'])
                views.append(dict(view, image_path=asset['image_path']))
            _, journal, assets = shared.build_review_payload(item['context'], views, model)
            if journal['state']['context'] != item['context']:
                raise ValueError('Shared review builder changed the original public context')
            if [(a['view_id'], a['sha256']) for a in assets] != [(a['view_id'], a['sha256']) for a in item['assets']]:
                raise ValueError('Shared review builder changed image bytes or ordering')
            no_truth(journal)
            response = client.verify(item['context'], views, root/'tool_journal'/(item['case_id']+'.json'))
            record.update(status='ok' if response.get('status') == 'ok' else 'tool_error',
                          response=response, verdict=response.get('verdict', 'invalid') if response.get('status') == 'ok' else 'invalid')
        except Exception as exc:
            record.update(status='error', error=type(exc).__name__ + ': ' + str(exc), verdict='invalid')
            # Identity/cache mismatches are not candidate outcomes. Stop the run.
            if isinstance(exc, shared.Halt):
                record['finished_unix'] = time.time()
                save(destination, record)
                raise
        record['finished_unix'] = time.time()
        save(destination, record)
        print(json.dumps({'case_id': item['case_id'], 'status': record['status'], 'verdict': record['verdict']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'analyze', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--model', help='Exact model identity returned by the local /health endpoint')
    parser.add_argument('--service-manifest', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root)
    elif args.command == 'run':
        run(args.root, args.url, args.limit, args.model, args.service_manifest)
    elif args.command == 'analyze':
        analyze(args.root)
    else:
        design, cases = validate_public(args.root)
        print(json.dumps({'status': 'verified', 'count': len(cases), 'public_sha256': design['public_sha256']}))


if __name__ == '__main__':
    main()
