"""Cross-check saved results against the project's earlier evaluator, offline only."""
import importlib.util
import json
import math
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('online_metrics_audit', Path(__file__).with_name('online_experiment.py'))
online = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(online)


def main():
    root = online.ROOT
    plan = online.validate_frozen(root)
    analysis = online.read(root/'analysis.json')
    if analysis['plan_sha256'] != online.sha(root/'plan.json'):
        raise ValueError('Analysis belongs to a different plan')
    if online.sha(root/'evaluator_only.json') != plan['evaluator_sha256']:
        raise ValueError('Evaluation labels changed')
    truth = online.read(root/'evaluator_only.json')
    legacy_root = WORK/'tests/refactor_online_compare/runtime_before'
    sys.path.insert(0, str(legacy_root))
    import AVDN.agent.evaluate as reference
    import numpy as np
    if not Path(reference.__file__).resolve().is_relative_to(legacy_root.resolve()):
        raise RuntimeError('Unexpected reference evaluator import')
    rows = []
    for item in analysis['rows']:
        if item['status'] not in online.TERMINAL:
            continue
        path = Path(item['result_path'])
        result = online.read(path) if path.exists() else {}
        final = result.get('final')
        gt = truth[item['dataset']][item['episode_id']]
        metric, primary, secondary = None, False, False
        if final is not None:
            if item['dataset'] == 'AVDN':
                metric = reference.iou(final['evaluation_polygon_xy_m'], gt['evaluation_polygon_xy_m'])
                primary, secondary = metric > .4, metric > .3
            else:
                metric = float(np.linalg.norm(np.asarray(final['center_xy_m'])-gt['goal_xy_m']))
                primary = secondary = metric <= 20
        if item['status'] == 'failed':
            primary = secondary = False
        actual = item.get('metric')
        same_metric = metric is None and actual is None or (metric is not None and actual is not None and
            math.isclose(metric, actual, rel_tol=1e-12, abs_tol=1e-10))
        if not same_metric or primary != item['success'] or secondary != item['secondary_success']:
            raise RuntimeError('Metric disagreement: '+item['job_id'])
        rows.append(dict(job_id=item['job_id'], has_prediction=final is not None,
                         metric=metric, primary_success=primary, secondary_success=secondary))
    report = dict(status='matched', checked=len(rows), planned=len(plan['jobs']),
        complete=len(rows)==len(plan['jobs']), source_analysis_sha256=online.sha(root/'analysis.json'),
        reference_evaluator=str(Path(reference.__file__).resolve()),
        reference_sha256=online.sha(reference.__file__), rows=rows,
        notes=['The earlier project evaluator uses polygon intersection/union and final predicted center distance.',
               'The frozen study uses AVDN >0.4 primary (capacity20 convention) and >0.3 secondary (earlier generic evaluator).',
               'Actual stopping distance is a separate metric and is never substituted for predicted target distance.',
               'This checks the recorded geometry and success flags, not semantic target identity.'])
    online.save(root/'metric_audit.json', report)
    print(json.dumps({k:report[k] for k in ('status','checked','planned','complete')}))


if __name__ == '__main__':
    main()
