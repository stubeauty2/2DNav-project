"""Offline visual-audit material. NEVER load this output into a navigation run."""
from pathlib import Path
from PIL import Image, ImageDraw
from AVDN.agent.io import read_jsonl, load_json, save_json
from .environment import Environment


def export(root):
    root = Path(root)
    public = {x['episode_id']: x for x in read_jsonl(root/'inputs.jsonl')}
    report = []
    for gt in read_jsonl(root/'evaluator_only.jsonl'):
        ident = gt['episode_id']
        path = root/'episodes'/ident/'trace.json'
        if not path.exists():
            continue
        trace = load_json(path)
        mains = [v for v in trace['views'] if v['source']=='main']
        folder = root/'offline_visual_audit'/ident
        env = Environment(public[ident]['environment'], folder)
        try:
            # North-up GT stop-area crop is context, NOT an object bounding box.
            goal = env.main(gt['goal_xy_m'], 0, 'evaluator_goal_area')
        finally:
            env.close()
        panels = [('GT stop area; north up; target NOT annotated', goal['image_path'])]
        panels += [(v['view_id']+'; heading '+str(round(v['image_up_heading_deg'],1)),v['image_path']) for v in mains]
        side, banner = 320, 32
        sheet = Image.new('RGB', (side*3, (side+banner)*((len(panels)+2)//3)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,(label,image) in enumerate(panels):
            x,y=(i%3)*side,(i//3)*(side+banner)
            draw.text((x+3,y+4),label,fill='black')
            with Image.open(image) as im:
                sheet.paste(im.convert('RGB').resize((side,side)),(x,y+banner))
        sheet.save(folder/'contact_sheet.jpg')
        record = dict(episode_id=ident,instruction=public[ident]['agent_input']['instruction_raw'],
                      contact_sheet=str((folder/'contact_sheet.jpg').resolve()),
                      target_identity='unreviewed', first_target_visible_view=None,
                      note='Human/visual audit required; GT stop area alone does not prove target identity.')
        report.append(record)
    save_json(root/'offline_visual_audit'/'index.json',report)
    return report


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print('Exported',len(export(parser.parse_args().root)))
