"""Replay fixed multi-candidate replies against the preserved pre-simplification core.

Both subprocesses are offline. Only paths/configuration plumbing are normalized;
requests, actual crops, candidate boxes, movement events and final geometry must match.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "tests/tmp/andh_before_simplify_20261002"
BOX = [354,354,414,414]


def proposal(source=None, invalid_slot=False):
    item = None if source is None else dict(source_view_id=source,bbox_2d=BOX,
        support=["roof and nearby road"],conflicts=[],evidence_view_ids=[source])
    bad = dict(item,bbox_2d=[10,10,768,768]) if invalid_slot else None
    return json.dumps({"candidates":[item,bad,None]})


def confirmation(decision="confirmed",round_n=1,cid="s1:c1",ids=None,bbox=BOX):
    return json.dumps(dict(decision=decision,candidate_id=cid if decision=="confirmed" else None,
        target_view_id=f"s1:r{round_n}:confirm_{cid.rsplit(':',1)[-1]}:rgb" if decision=="confirmed" else None,
        bbox_2d=bbox if decision=="confirmed" else None,support=["matches full description"],
        conflicts=[],evidence_view_ids=[],candidate_assessments=[dict(candidate_id=c,
            status="confirmed" if decision=="confirmed" and c==cid else "uncertain",
            reason="visual appearance checked",unresolved=[]) for c in (ids or [cid])]))


def cases():
    multiple=json.loads(proposal("s1:r1:current:rgb"))
    multiple["candidates"][1]=dict(multiple["candidates"][0],source_view_id="s1:r1:left:rgb",
        evidence_view_ids=["s1:r1:left:rgb"])
    return {
        "multiple_candidates": [json.dumps(multiple),confirmation(cid="s1:c2",ids=["s1:c1","s1:c2"])],
        "third_round_confirmation": [proposal("s1:r1:current:rgb"),confirmation("continue_search"),
            proposal(),confirmation("continue_search",round_n=2),proposal(),confirmation(round_n=3)],
        "no_match": [proposal()]*3,
        "invalid_slot_with_valid_candidate": [proposal("s1:r1:current:rgb",True),confirmation()],
        "invalid_round_then_recovery": [proposal("unknown"),proposal("s1:r2:current:rgb"),confirmation(round_n=2)],
        "strict_geometry_retry": [proposal("s1:r1:current:rgb"),confirmation(bbox=[10,10,50,50]),
            proposal(),confirmation(round_n=2)],
    }


def normalize(value):
    if isinstance(value,dict):
        return {k:normalize(v) for k,v in value.items() if k not in ("image_path","options")}
    if isinstance(value,list):
        return [normalize(v) for v in value]
    return value


def worker(package_parent, output_path):
    from fixtures import NavigationFixture
    sys.path.insert(0,str(package_parent))
    from andh_core.config import NavigationOptions
    previous=(package_parent/"andh_core/multiview_navigation.py").exists()
    if previous:
        from andh_core import multiview_navigation as nav
        navigate=nav.search_multiview
        legacy=dict(target_terms=["roof"],vision_tool_url="")
    else:
        from andh_core import navigation as nav
        navigate=nav.search_and_reach_destination
        legacy={}
    output={}
    for name,replies in cases().items():
        fixture=NavigationFixture()
        fixture.setUp()
        try:
            settings=dict(strict_confirmation=True,enable_trace=True,history_size=1,
                          use_relative_pose=True,model_retries=0)
            if previous: settings["multiview_mode"]="candidates"
            request_hashes=[]
            scripted=iter(replies)
            def complete(request,**kwargs):
                payload=json.dumps(request,sort_keys=True,ensure_ascii=False).encode("utf-8")
                request_hashes.append(hashlib.sha256(payload).hexdigest())
                return next(scripted)
            with patch.object(nav.ModelClient,"complete",side_effect=complete), \
                 patch("requests.sessions.Session.request",side_effect=AssertionError("network forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result=navigate(instr_id="s1",ob=fixture.ob,dest_desc="roof",via_desc="",
                    start_pos_latlng=[.05,.05],heading_deg=0,scale_factor=5,out_dir=str(fixture.out),
                    step_meters=120,max_steps=3,api_key="",url="",model="",options=NavigationOptions(**settings),**legacy)
            record=json.loads((fixture.out/"s1_search.json").read_text(encoding="utf-8"))
            output[name]=dict(result=dict(found=result[0],destination=result[1],
                trajectory=[b.tolist() for b in result[2]],zoom_index=result[3]),
                record=normalize(record),request_sha256=request_hashes)
        finally:
            fixture.tearDown()
    output_path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")


def compare(baseline=DEFAULT_BASELINE):
    if not (baseline/"andh_core/multiview_navigation.py").is_file():
        raise FileNotFoundError("Pre-simplification snapshot missing: " + str(baseline))
    temporary_root=ROOT/"tests/tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        output=[]
        for label,package in (("before",baseline),("after",ROOT/"PSC-AVDN/script")):
            path=Path(directory)/(label+".json")
            command=[sys.executable,"-B","-X","utf8",str(Path(__file__).resolve()),"--worker",str(package),"--output",str(path)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=180)
            if result.returncode:
                raise RuntimeError(label+" replay failed:\n"+result.stdout+result.stderr)
            output.append(json.loads(path.read_text(encoding="utf-8")))
        before,after=output
        for name in before:
            if before[name]!=after[name]:
                diff_root=temporary_root/"simplification_replay_failure"
                diff_root.mkdir(exist_ok=True)
                for label,data in (("before",before[name]),("after",after[name])):
                    (diff_root/(name+"_"+label+".json")).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
                raise AssertionError(name+" differs; diagnostic JSON: "+str(diff_root))
        return {name:{"matched":True,"found":data["result"]["found"],
                      "calls":len(data["request_sha256"]),"trajectory_boxes":len(data["result"]["trajectory"])}
                for name,data in after.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir",type=Path,default=DEFAULT_BASELINE)
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.output)
    else:
        result=compare(args.baseline_dir)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.output:
            args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
