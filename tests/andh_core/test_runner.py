"""Offline checks for the runner's independence from the legacy directory."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fixtures import NavigationFixture

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "PSC-AVDN/script"
sys.path[:0] = [str(SCRIPT), str(ROOT / "src")]

from andh_core import runner


class RunnerTests(NavigationFixture, unittest.TestCase):
    def run_trajectory(self, stages, replies, **settings):
        observation=dict(self.ob,route_index=0,starting_angle=0)
        env=SimpleNamespace(_get_obs=lambda t:[observation],eval_metrics=lambda *a,**k:({},{}))
        csv_path=self.root/'stages.csv'
        with csv_path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['instr_id','step_idx','instruction','move_dir','dest','via','forward'])
            writer.writeheader()
            for index,(destination,forward) in enumerate(stages,1):
                writer.writerow(dict(instr_id='map__0',step_idx=index,instruction='fixture',move_dir='12:00',
                                     dest=destination,via='',forward=forward))
        with patch.object(runner,'ANDHNavBatch',return_value=env), \
             patch.object(runner,'DataLoader',return_value=[None]), \
             patch.object(runner,'search_and_reach_destination',side_effect=replies) as search, \
             patch.object(runner,'save_traj_boxes_debug_image'), \
             contextlib.redirect_stdout(io.StringIO()):
            predictions,_=runner.run_turn_and_crop_full_trajectory(
                anno_dir=str(self.root),dataset_dir=str(self.root),results_csv=str(csv_path),
                out_dir=str(self.out),api_key='',url='',model='',
                navigation_options=runner.NavigationOptions(**settings))
        return predictions['map__0'],search

    def test_full_trajectory_failure_stops_later_stages_and_keeps_fallback(self):
        box=np.asarray(self.ob['gt_path_corners'][0])
        prediction,search=self.run_trajectory([('first roof',False),('second roof',False),('third roof',False)],
            [(True,[.052,.051],[box],0),(False,None,[],None)],max_steps=4,step_meters=90)
        self.assertEqual(search.call_count,2)
        first,second=[call.kwargs for call in search.call_args_list]
        self.assertEqual(first['max_steps'],4)
        self.assertEqual(first['step_meters'],90)
        self.assertEqual(second['start_pos_latlng'],[.052,.051])
        self.assertIs(first['trajectory_context'],second['trajectory_context'])
        expected_position=runner._move_forward_geo([.05,.05],0,meters=120)
        expected=runner.generate_view_corners_with_scale(expected_position,self.ob,scale_factor=2,angle_deg=0)
        np.testing.assert_allclose(prediction['path_corners'][-1][0],expected)
        expected_iou=float(runner.compute_iou(expected,self.ob['gt_path_corners'][-1]))
        self.assertEqual(prediction['final_iou'],expected_iou)
        self.assertEqual(prediction['success'],expected_iou>.3)
        progress=json.loads((self.out/'turn_and_crop_eval_results_full_traj.in_progress.json').read_text(encoding='utf-8'))
        self.assertEqual(progress['processed'],1)
        self.assertEqual(progress['predictions']['map__0']['final_iou'],expected_iou)

    def test_destination_special_case_bypasses_search(self):
        prediction,search=self.run_trajectory([('destination',False)],[])
        search.assert_not_called()
        np.testing.assert_allclose(prediction['path_corners'][-1][0],self.ob['gt_path_corners'][0])
        self.assertAlmostEqual(prediction['final_iou'],1)

    def test_successful_stage_uses_returned_confirmation_frame_for_final_iou(self):
        box=np.asarray(self.ob['gt_path_corners'][0])
        prediction,search=self.run_trajectory([('roof',False)],[(True,[.05,.05],[box],0)])
        self.assertEqual(search.call_count,1)
        np.testing.assert_allclose(prediction['path_corners'][-1][0],box)
        self.assertEqual(prediction['final_iou'],1)
        self.assertTrue(prediction['success'])

    def test_main_preserves_defaults_and_forwards_cli_options(self):
        with patch.object(runner.os, "makedirs") as mkdir, patch.object(
            runner, "run_turn_and_crop_full_trajectory"
        ) as run:
            runner.main([])
            mkdir.assert_called_once_with(runner.OUT_DIR, exist_ok=True)
            run.assert_called_once_with(
                anno_dir=runner.ANNO_DIR, dataset_dir=runner.DATASET_DIR,
                split=runner.SPLIT, results_csv=runner.RESULTS_CSV,
                out_dir=runner.OUT_DIR, scale_factor=runner.SCALE_FACTOR,
                api_key=runner.DEFAULT_API_KEY, url=runner.QWEN_URL,
                model=runner.QWEN_MODEL,
                navigation_options=runner.NavigationOptions(),
            )
            run.reset_mock()
            runner.main(["--qwen-url", "http://example.invalid/model",
                         "--qwen-model", "test-model",
                         "--anno-dir", "sample-annotations", "--dataset-dir", "sample-images",
                         "--results-csv", "stages.csv", "--out-dir", "experiment-output",
                         "--history-size", "1", "--relative-pose", "--trace",
                         "--max-steps", "4", "--step-meters", "90", "--model-retries", "2"])
            self.assertEqual(run.call_args.kwargs["model"], "test-model")
            self.assertEqual(run.call_args.kwargs["url"], "http://example.invalid/model")
            self.assertEqual(run.call_args.kwargs["anno_dir"], "sample-annotations")
            self.assertEqual(run.call_args.kwargs["dataset_dir"], "sample-images")
            self.assertEqual(run.call_args.kwargs["results_csv"], "stages.csv")
            self.assertEqual(run.call_args.kwargs["out_dir"], "experiment-output")
            self.assertEqual(run.call_args.kwargs["navigation_options"],
                             runner.NavigationOptions(1, True, True, max_steps=4, step_meters=90, model_retries=2))

    def test_direct_and_module_entry_without_legacy_access(self):
        # A fresh process rejects all file reads under ANDH-Full. No renaming or
        # deletion of the user's legacy files is necessary to prove independence.
        program = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1])
mode = sys.argv[2]

def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace("\\", "/").lower()
        if "/andh-full/" in path:
            raise RuntimeError("Legacy directory access: " + path)

class BlockLegacyTerms(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "category_terms":
            raise RuntimeError("Legacy absolute category_terms import")

sys.addaudithook(audit)
sys.meta_path.insert(0, BlockLegacyTerms())
sys.argv = [str(script / "andh_core/runner.py"), "--help"]
if mode == "module":
    sys.path.insert(0, str(script))
    runpy.run_module("andh_core.runner", run_name="__main__")
else:
    runpy.run_path(str(script / "andh_core/runner.py"), run_name="__main__")
'''
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("direct", "module"):
                with self.subTest(mode=mode):
                    result = subprocess.run(
                        [sys.executable, "-B", "-c", program, str(SCRIPT), mode],
                        cwd=directory, env=env, capture_output=True, text=True,
                        timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("--qwen-model", result.stdout)
                    self.assertFalse((SCRIPT / "andh_core/__main__.py").exists())


if __name__ == "__main__":
    unittest.main()
