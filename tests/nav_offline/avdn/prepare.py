"""Offline-only preparation: splits public runtime inputs from private evaluation."""
from pathlib import Path
import numpy as np
from PIL import Image
from AVDN.tools.runtime import read_jsonl, write_jsonl, load_json, save_json, digest
from AVDN.tools.geometry import homography
from .geometry import latlon_to_xy
DATASET = 'AVDN'

def prepare(root, count=10):
    root = Path(root)
    folder = root / DATASET.lower()
    folder.mkdir(parents=True, exist_ok=True)
    base = Path('out/04_current_initialization/prompt_debug_20261013')
    selection = base / 'model_selection'
    inputs = base / 'dev/avdn/agent_inputs.jsonl'
    predictions = selection / 'qwen3.8-flash/avdn/predictions.jsonl'
    dev = base / 'dev' / DATASET.lower()
    manifest = load_json(dev / 'manifest.json')
    rows = load_json(manifest['annotation'])
    private = {r['episode_id']: r for r in read_jsonl(dev / 'evaluator_only.jsonl')}
    cached = {r['episode_id']: r for r in read_jsonl(predictions)}
    selected = read_jsonl(inputs)[:count]
    public = []
    evaluators = []
    for entry in selected:
        ident = entry['episode_id']
        safe = entry['agent_input']
        gt = private[ident]
        row = rows[gt['source_row']]
        if cached[ident]['status'] != 'ok':
            raise ValueError('Fixed episode lacks valid initialization: ' + ident)
        safe['views'] = []
        origin = np.mean(row['gt_path_corners'][0], axis=0).tolist()
        image = Path('datasets/AVDN/train_images') / (str(row['map_name']) + '.tif')
        with Image.open(image) as im:
            w, h = im.size
        low, high = (row['gps_botm_left'], row['gps_top_right'])
        en0 = latlon_to_xy([high[0], low[1]], origin)
        en1 = latlon_to_xy([low[0], high[1]], origin)
        world_to_raster = [[w / (en1[0] - en0[0]), 0, -en0[0] * w / (en1[0] - en0[0])], [0, h / (en1[1] - en0[1]), -en0[1] * h / (en1[1] - en0[1])], [0, 0, 1]]
        width = 1120 * abs(en1[0] - en0[0]) / w
        height = 1120 * abs(en1[1] - en0[1]) / h
        goal_polygon = [latlon_to_xy(p, origin) for p in row['destination']]
        eval_polygon = [latlon_to_xy(p, origin) for p in row['gt_path_corners'][-1]]
        evaluator = dict(gt, goal_polygon_xy_m=goal_polygon, evaluation_polygon_xy_m=eval_polygon)
        env = dict(image_path=str(image.resolve()), image_size=[w, h], world_to_raster=world_to_raster, main_width_m=width, main_height_m=height)
        public.append(dict(episode_id=ident, agent_input=safe, initialization=cached[ident]['parsed'], environment=env))
        evaluators.append(evaluator)
    if (folder / 'episodes.jsonl').exists():
        if read_jsonl(folder / 'episodes.jsonl') != public:
            raise ValueError('Existing frozen sample differs; use a new experiment directory')
    write_jsonl(folder / 'episodes.jsonl', public)
    write_jsonl(folder / 'evaluator_only.jsonl', evaluators)
    save_json(folder / 'sample_manifest.json', dict(dataset=DATASET, episode_ids=[x['episode_id'] for x in public], selection='first 10 existing validation inputs in source order, no performance filter', annotation=manifest['annotation'], initialization_cache=str(predictions.resolve()), initialization_cache_sha256=digest(read_jsonl(predictions)), public_sha256=digest(public), pilot_ids=[x['episode_id'] for x in public[:2]], count=count))
    return dict(dataset=DATASET, episodes=len(public), out=str(folder))
