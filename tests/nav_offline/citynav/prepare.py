"""Offline-only preparation: splits public runtime inputs from private evaluation."""
from pathlib import Path
import numpy as np
from PIL import Image
from CityNav.tools.runtime import read_jsonl, write_jsonl, load_json, save_json, digest
from CityNav.tools.geometry import homography
DATASET = 'CityNav'

def prepare(root, count=10):
    root = Path(root)
    folder = root / DATASET.lower()
    folder.mkdir(parents=True, exist_ok=True)
    base = Path('out/04_current_initialization/prompt_debug_20261013')
    selection = base / 'model_selection'
    inputs = selection / 'compact/dev/citynav/agent_inputs.jsonl'
    predictions = selection / 'qwen3.5-flash-compact/citynav/predictions.jsonl'
    dev = base / 'dev' / DATASET.lower()
    manifest = load_json(dev / 'manifest.json')
    rows = load_json(manifest['annotation'])
    private = {r['episode_id']: r for r in read_jsonl(dev / 'evaluator_only.jsonl')}
    cached = {r['episode_id']: r for r in read_jsonl(predictions)}
    selected = read_jsonl(inputs)[:count]
    import tifffile
    objects = load_json('D:/datasets/CityNav/data/refined_label/cityrefer/objects.json')
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
        name = gt['map_name']
        image = Path('D:/datasets/CityNav/data/rgbd') / (name + '.png')
        with tifffile.TiffFile(image.with_suffix('.tif')) as tif:
            tags = tif.pages[0].tags
            sx, sy, _ = tags['ModelPixelScaleTag'].value
            ix, iy, _, wx, wy, _ = tags['ModelTiepointTag'].value
        world_to_raster = [[1 / sx, 0, ix - wx / sx], [0, -1 / sy, iy + wy / sy], [0, 0, 1]]
        old = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(image) as im:
                w, h = im.size
        finally:
            Image.MAX_IMAGE_PIXELS = old
        width = height = 100.0
        obj = objects[name][str(row['object_ids'][0])]
        evaluator = dict(gt, goal_polygon_xy_m=[p[:2] for p in obj.get('contour', [])])
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
