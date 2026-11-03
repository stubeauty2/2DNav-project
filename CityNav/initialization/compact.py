"""Reduce public landmark catalogs using instruction names, without reading GT."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

GENERIC = set('the a an of and at in on road street lane avenue drive park hall building house church river north south east west saint st rd'.split())


def words(text):
    return re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKD', text).casefold())


def select_catalog(instruction, catalog):
    tokens = words(instruction)
    text = ' '+' '.join(tokens)+' '
    vocabulary = set(tokens)
    matches = []
    for landmark in catalog:
        name = words(landmark['name'])
        distinctive = [t for t in name if t not in GENERIC and len(t) >= 3]
        exact = bool(name) and (' '+' '.join(name)+' ') in text
        overlap = bool(distinctive) and sum(t in vocabulary for t in distinctive)/len(distinctive) >= 0.75
        if exact or overlap:
            matches.append(landmark)
    # Retain every segment with a matched name; never select by goal geometry or proximity.
    return (matches, 'instruction_name_match') if matches else (catalog, 'unmatched_keep_all')


def compact(inputs, output, audit_path):
    rows = [json.loads(s) for s in Path(inputs).read_text(encoding='utf-8').splitlines() if s.strip()]
    audit=[]
    for row in rows:
        agent=row['agent_input'];old=agent['named_landmarks']
        selected,mode=select_catalog(agent['instruction_raw'],old)
        agent['named_landmarks']=selected
        audit.append({'episode_id':row['episode_id'],'original_count':len(old),'retained_count':len(selected),
                      'method':mode,'retained_ids':[x['id'] for x in selected]})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    Path(audit_path).write_text(json.dumps({'gt_read':False,'selection':'public name lexical match; unmatched catalogs preserved',
                                         'episodes':audit},ensure_ascii=False,indent=2),encoding='utf-8')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True);parser.add_argument('--out',required=True);parser.add_argument('--audit',required=True)
    args=parser.parse_args();audit=compact(args.inputs,args.out,args.audit)
    print(json.dumps({'episodes':len(audit),'original_landmarks':sum(r['original_count'] for r in audit),
                      'retained_landmarks':sum(r['retained_count'] for r in audit)}))
