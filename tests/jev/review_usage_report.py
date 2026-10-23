"""Read existing public Jev journals/trace only; no inference or evaluator access."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


TERMINAL = {'ok', 'error', 'identity_error'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def percentiles(values):
    """Linear interpolation between sorted observations; no third-party dependency."""
    values = sorted(float(x) for x in values if isinstance(x, (int, float))
                    and not isinstance(x, bool) and math.isfinite(x))
    if not values:
        return {'n': 0}
    result = {'n': len(values), 'min': values[0], 'max': values[-1]}
    for label, q in [('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9), ('p95', .95)]:
        at = (len(values)-1)*q
        lo, hi = math.floor(at), math.ceil(at)
        result[label] = round(values[lo] + (values[hi]-values[lo])*(at-lo), 3)
    return result


def evidence_key(record):
    """Conservative exact key; differing coordinates are not fuzzy-matched.

    Deliberately excludes round, budgets, review view ID, and full navigation state.
    Thus equal keys mean repeated pixels/proposal/task, NOT equal complete requests.
    Auxiliary identity is not explicit in the current protocol; include all public
    auxiliary hints without pretending any one is the requested auxiliary object.
    """
    context = record['input']['state']['context']
    candidate = context['candidate']
    assets = record.get('assets') or record['input']['media']
    hashes = [a['sha256'] for a in assets]
    if len(hashes) != 2 or any(not isinstance(h, str) or not h for h in hashes):
        raise ValueError('Expected two ordered image hashes')
    role = candidate['target']
    init = context.get('initialization', {})
    description = (init.get('final_target') if role == 'final' else
                   {'explicit_auxiliary_identity': None, 'public_hints': init.get('auxiliary', [])})
    return dict(ordered_image_sha256=hashes,
                candidate={k: candidate.get(k) for k in
                           ('view_id', 'bbox', 'polygon_xy_m', 'center_xy_m')},
                target_role=role, target_description=description,
                instruction_sha256=digest({'instruction_raw': context.get('instruction_raw'),
                                          'turns': context.get('turns')}))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def episode_scope(root):
    """Use only declared Jev jobs when a plan exists; never silently unbind it."""
    plan_path = root/'plan.json'
    if not plan_path.exists():
        return sorted((root/'episodes').glob('*/*/jev/*')), dict(
            binding='unbound', plan_sha256=None, planned_jev_jobs=None)
    content = plan_path.read_bytes()
    plan = json.loads(content)
    if not isinstance(plan, dict) or not isinstance(plan.get('jobs'), list):
        raise ValueError('Existing plan must declare a jobs list')
    folders = []
    for job in plan['jobs']:
        if not isinstance(job, dict):
            raise ValueError('Plan job must be an object')
        if job.get('arm') != 'jev':
            continue
        parts = [job.get(k) for k in ('dataset', 'model', 'episode_id')]
        if any(not isinstance(p, str) or not p or p in ('.', '..')
               or any(c in p for c in ('/', '\\', ':')) for p in parts):
            raise ValueError('Jev job identity must contain safe single path components')
        dataset, model, episode_id = parts
        folder = root/'episodes'/dataset/model/'jev'/episode_id
        if not folder.resolve().is_relative_to((root/'episodes').resolve()):
            raise ValueError('Jev job path leaves the episode root')
        folders.append(folder)
    if len(folders) != len(set(folders)):
        raise ValueError('Duplicate Jev episode directory in plan')
    return sorted(folders), dict(binding='plan', plan_sha256=hashlib.sha256(content).hexdigest(),
                                 planned_jev_jobs=len(folders))


def summarize(root):
    root = Path(root).resolve()
    report = dict(schema='jev-review-usage-v1', snapshot_utc=datetime.now(timezone.utc).isoformat(),
                  source_root=str(root), read_policy='Only plan.json (if present) and selected Jev episode trace.json / tools/*jev-review.json; no evaluator files',
                  snapshot_consistency='Non-transactional scan of live files between snapshot_utc and snapshot_finished_utc; files may change during the scan',
                  interpretation='Visual-evidence duplicates are not identical full requests. No accuracy or GT claims.',
                  grouping='Per episode: ordered image SHA256 + exact original candidate view/bbox/world geometry + role/description + full instruction digest. No rounding/fuzzy matching.',
                  character_measure='len(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)); Unicode characters, not tokens',
                  episodes=[], read_errors=[], skipped_pending=0, skipped_unknown_status=0)
    folders, scope = episode_scope(root)
    report.update(scope)
    field_lengths = defaultdict(list)
    tokens, context_lengths, transitions, statuses = [], [], Counter(), Counter()
    all_groups = []
    for folder in folders:
        if not folder.is_dir():
            continue
        episode = dict(path=folder.relative_to(root).as_posix(), move_count=None,
                       termination=None, terminal_review_count=0, groups=[])
        trace_path = folder/'trace.json'
        if trace_path.exists():
            try:
                trace = load(trace_path)
                episode.update(move_count=sum(e.get('kind') == 'move' for e in trace.get('events', [])),
                               termination=trace.get('termination'))
            except (OSError, ValueError, TypeError) as exc:
                report['read_errors'].append({'file': str(trace_path), 'error': type(exc).__name__})
        grouped = {}
        for path in sorted((folder/'tools').glob('*jev-review.json')):
            try:
                record = load(path)
                if record.get('status') == 'pending':
                    report['skipped_pending'] += 1
                    continue
                if record.get('status') not in TERMINAL:
                    report['skipped_unknown_status'] += 1
                    continue
                key = evidence_key(record)
                key_hash = digest(key)
                context = record['input']['state']['context']
                input_hash = digest(record['input'])
                raw = record.get('raw') or {}
                raw_choice = raw.get('answers', {}).get('review', {}).get('choice')
                # Raw choice is intentionally distinct from a client-thresholded verdict.
                raw_choice = raw_choice if raw_choice is not None else record.get('choice')
                transition = str(raw_choice or 'unavailable')+'->'+str(record.get('verdict'))
                token_count = raw.get('usage', {}).get('input_tokens')
                length = len(canonical(context))
                lengths = {k: len(canonical(v)) for k, v in context.items()}
                entry = dict(journal=path.relative_to(root).as_posix(), status=record['status'],
                             round=context.get('round'), original_signature=record.get('signature'),
                             journal_input_sha256=input_hash, raw_choice=raw_choice,
                             verdict=record.get('verdict'), max_probability=record.get('max_probability'),
                             abstained=record.get('abstained'), input_tokens=token_count,
                             context_characters=length, field_characters=lengths)
                if key_hash not in grouped:
                    grouped[key_hash] = dict(evidence_sha256=key_hash, evidence=key, reviews=[])
                grouped[key_hash]['reviews'].append(entry)
                tokens.append(token_count)
                context_lengths.append(length)
                transitions[transition] += 1
                statuses[record['status']] += 1
                for k, value in lengths.items():
                    field_lengths[k].append(value)
                episode['terminal_review_count'] += 1
            except (OSError, ValueError, TypeError, KeyError) as exc:
                report['read_errors'].append({'file': str(path), 'error': type(exc).__name__})
        for group in grouped.values():
            rows = group['reviews']
            group.update(count=len(rows), repeated_calls=max(0, len(rows)-1),
                         distinct_full_journal_inputs=len({r['journal_input_sha256'] for r in rows}),
                         distinct_request_signatures=len({r['original_signature'] for r in rows if r['original_signature']}))
            episode['groups'].append(group)
            all_groups.append(group)
        episode['repeated_calls'] = sum(g['repeated_calls'] for g in episode['groups'])
        report['episodes'].append(episode)
    report['summary'] = dict(episodes=len(report['episodes']), terminal_reviews=sum(statuses.values()),
                           statuses=dict(statuses), raw_to_verdict=dict(transitions),
                           evidence_groups=len(all_groups), repeated_evidence_groups=sum(g['count'] > 1 for g in all_groups),
                           repeated_calls=sum(g['repeated_calls'] for g in all_groups),
                           input_tokens=percentiles(tokens), context_characters=percentiles(context_lengths),
                           field_characters={k: percentiles(v) for k, v in sorted(field_lengths.items())})
    report['snapshot_finished_utc'] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('out/jev_study_20261022/online'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    output = args.output or args.root/'review_usage.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(dict(output=str(output.resolve()), snapshot_utc=report['snapshot_utc'],
                          snapshot_finished_utc=report['snapshot_finished_utc'],
                          binding=report['binding'], plan_sha256=report['plan_sha256'],
                          summary=report['summary'], read_errors=report['read_errors']), ensure_ascii=False))


if __name__ == '__main__':
    main()
