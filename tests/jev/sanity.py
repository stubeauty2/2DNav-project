"""Synthetic image/option sanity check, separate from navigation evaluation."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import requests
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    args = parser.parse_args()
    im = Image.new('RGB', (320, 240), 'white')
    draw = ImageDraw.Draw(im)
    draw.rectangle((20, 20, 140, 100), fill='red')
    draw.ellipse((180, 120, 300, 220), fill='blue')
    buf = BytesIO(); im.save(buf, format='PNG')
    questions = {
        'rectangle_color': dict(type='choice', instructions='What color is the rectangle?',
            criteria={'red': 'red', 'blue': 'blue', 'unknown': 'cannot tell'}),
        'circle_color': dict(type='choice', instructions='What color is the circle?',
            criteria={'red': 'red', 'blue': 'blue', 'unknown': 'cannot tell'}),
        'is_red_rectangle': dict(type='noul', instructions='There is a red rectangle in the image.'),
        'is_green_rectangle': dict(type='noul', instructions='There is a green rectangle in the image.'),
    }
    payload = dict(model=args.model, state={'image': '<image:1>'}, questions=questions,
        media=[dict(type='image', data='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())])
    response = requests.post(args.url+'/v1/systemone', json=payload, timeout=180)
    response.raise_for_status()
    raw = response.json()
    folder = Path(__file__).resolve().parents[2]/'out/jev_study_20261022/sanity'
    folder.mkdir(parents=True, exist_ok=True)
    im.save(folder/'synthetic.png')
    out = dict(model=args.model, health=requests.get(args.url+'/health',timeout=5).json(),
               questions=questions, response=raw)
    (folder/(args.model+'.json')).write_text(json.dumps(out,indent=2), encoding='utf-8')
    print(json.dumps(raw['answers']))

if __name__ == '__main__':
    main()
