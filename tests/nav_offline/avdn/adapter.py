import hashlib
import json
import re
from pathlib import Path
from .geometry import latlon_to_xy


def load_episodes(annotation, settings=('visual',), image_dir=None, image_output=None):
    rows = json.loads(Path(annotation).read_text(encoding='utf-8'))
    for i, row in enumerate(rows):
        first = row['gt_path_corners'][0]
        start = [sum(p[k] for p in first)/len(first) for k in (0, 1)]
        goal = [sum(p[k] for p in row['destination'])/len(row['destination']) for k in (0, 1)]
        raw = row['instructions']
        blocks = re.findall(r'\[(INS|QUE)\]\s*(.*?)(?=\[(?:INS|QUE)\]|$)', raw, re.S)
        turns = [{'turn_id': 't'+str(j), 'role': role, 'text': text.strip()}
                 for j, (role, text) in enumerate(blocks)]
        if not turns:
            turns = [{'turn_id': 't0', 'role': 'INS', 'text': raw}]
        episode_id = 'avdn-' + hashlib.sha256((Path(annotation).name+':'+str(i)).encode()).hexdigest()[:16]
        agent = {'dataset': 'AVDN', 'setting': 'visual', 'instruction_raw': raw, 'turns': turns,
                 'initial_pose': {'xy_m': [0., 0.], 'heading_deg': float(row['angle']) % 360},
                 'coordinate_frame': 'local east/north meters centered on initial pose',
                 'named_landmarks': [], 'views': []}
        private = {'episode_id': episode_id, 'dataset': 'AVDN', 'source_row': i,
                   'source_route': str(row['map_name'])+'__'+str(row['route_index']),
                   'start_xy_m': [0., 0.], 'goal_xy_m': latlon_to_xy(goal, start),
                   'goal_definition': 'destination polygon vertex mean; not final GT view center',
                   'initial_heading_deg': agent['initial_pose']['heading_deg']}
        # Lazy rendering is controlled by the CLI after sampling, avoiding a full raster scan.
        yield episode_id, agent, private, {'row': row, 'start': start}


def render_initial(context, image_dir, output):
    from PIL import Image
    from .views import crop
    row, start = context['row'], context['start']
    source = Path(image_dir)/f"{row['map_name']}.tif"
    with Image.open(source) as img:
        w, h = img.size
    low, high = row['gps_botm_left'], row['gps_top_right']
    corners = row['gt_path_corners'][0]
    raster = [[(p[1]-low[1])/(high[1]-low[1])*w,
               (high[0]-p[0])/(high[0]-low[0])*h] for p in corners]
    return crop(source, raster, [latlon_to_xy(p, start) for p in corners], output)
