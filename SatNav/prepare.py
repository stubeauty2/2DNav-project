"""Freeze public samples before any prediction; truth stays evaluator-only."""
import random
from pathlib import Path
from AVDN.agent.io import save_json, write_jsonl, digest
from .adapter import load_episodes, raster_spec


def prepare(dataset, output, count=5, seed=20261109):
    output = Path(output)
    if count < 1:
        raise ValueError('count must be positive')
    if (output / 'sample_manifest.json').exists():
        raise ValueError('Sample already frozen; use a new output directory')
    public, truth = [], []
    for split in ('val_seen', 'val_unseen'):
        rows = list(load_episodes(dataset, split))
        # One language episode per trajectory; deterministic selection before inference.
        unique = {}
        for row in rows:
            private = row[2]
            unique.setdefault((private['scene_id'], str(private['trajectory_id'])), []).append(row)
        rng = random.Random(str(seed) + split)
        groups = rng.sample(sorted(unique), min(count, len(unique)))
        for group in groups:
            ident, agent, private, context = rng.choice(unique[group])
            public.append(dict(episode_id=ident, agent_input=agent, environment=raster_spec(**context)))
            truth.append(private)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'inputs.jsonl', public)
    write_jsonl(output / 'evaluator_only.jsonl', truth)
    save_json(output / 'sample_manifest.json', dict(seed=seed, count_per_split=count,
              selection='uniform distinct scene/trajectory pairs, then uniform instruction within pair',
              episode_ids=[p['episode_id'] for p in public], public_sha256=digest(public),
              observation_protocol='project continuous-action adapter; 100 m footprint, 768 px rendering; not native SatSim action parity'))
    return dict(count=len(public), output=str(output.resolve()))
