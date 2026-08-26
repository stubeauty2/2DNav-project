import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "PSC-AVDN" / "script"
ANDH_FULL_DIR = SCRIPT_DIR / "ANDH-Full"
for item in (SCRIPT_DIR, ANDH_FULL_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from visual_grounding.backends import RawProposal
from visual_grounding.contracts import parse_concepts, parse_views
from visual_grounding.geometry import (
    bbox_from_mask,
    box_iou,
    largest_component,
    mask_iou,
    navigation_point,
)
from visual_grounding.navigation_geometry import (
    NavigationGeometryAnalyzer,
    ViewGeoTransform,
    forward_sector_min_distance_px,
    navigation_vector,
)
from visual_grounding.postprocess import filter_proposals
from visual_grounding.annotation_tool import validate_manifest
from visual_grounding.evaluate import compute_metrics
from grounding_agent import (
    GroundingAgent,
    GroundingAgentError,
    GroundingInfrastructureError,
    POINT_TOOLS,
    SELECTION_TOOLS,
)
from visual_grounding.client import VisionToolError


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


search_engine = _load_module(
    "andh_full_search_engine_for_grounding_tests",
    ANDH_FULL_DIR / "search_engine.py",
)


class _Response:
    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": self.message}]}


def _tool_message(index, name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call-{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(dict(arguments))},
        }],
    }


class _VisionClient:
    def __init__(self, candidates=None, fail=False):
        self._candidates = list(candidates or [])
        self.fail = fail
        self.candidate_payloads = []
        self.match_payloads = []

    def health(self):
        if self.fail:
            raise VisionToolError("offline")
        return {"ok": True, "prewarmed": True, "backend": "sam3"}

    def candidates(self, payload):
        if self.fail:
            raise VisionToolError("offline")
        self.candidate_payloads.append(payload)
        return {"backend": "sam3", "candidates": self._candidates, "contact_sheet_paths": []}

    def matches(self, payload):
        if self.fail:
            raise VisionToolError("offline")
        self.match_payloads.append(payload)
        return {
            "provider": "dinov3",
            "matches": [{
                "candidate_id": item["candidate_id"],
                "target_view_id": payload["target_view_ids"][0],
                "similarity": 0.82,
                "geometric_consistency": 0.9,
            } for item in payload["candidates"]],
            "visualization_paths": [],
        }


def _candidate(candidate_id="cand-001", view_id="main", role="target"):
    return {
        "candidate_id": candidate_id,
        "concept_id": "c1",
        "concept_text": "gray building",
        "concept_role": role,
        "view_id": view_id,
        "view_role": view_id,
        "bbox_2d": [10, 20, 40, 60],
        "target_point_2d": [25, 40],
        "mask_path": "",
        "score": 0.7,
        "mask_quality": 0.8,
        "area_ratio": 0.02,
        "provider": "sam3",
        "component_scores": {"sam3": 0.7},
    }


class GeometryTests(unittest.TestCase):
    @staticmethod
    def _move(position, heading_deg, distance_m):
        radians = np.radians(float(heading_deg))
        lat, lng = map(float, position)
        return [
            lat + distance_m * np.cos(radians) / 111_320.0,
            lng + distance_m * np.sin(radians) / (111_320.0 * np.cos(np.radians(lat))),
        ]

    @classmethod
    def _corners(cls, center, heading_deg, half_forward_m, half_right_m):
        points = []
        for forward, right in (
            (half_forward_m, -half_right_m),
            (half_forward_m, half_right_m),
            (-half_forward_m, half_right_m),
            (-half_forward_m, -half_right_m),
        ):
            point = cls._move(center, heading_deg, forward)
            point = cls._move(point, heading_deg + 90.0, right)
            points.append(point)
        return points

    def test_view_geo_transform_round_trip_and_cross_scale_consistency(self):
        center = [30.0, 120.0]
        target = self._move(center, 90.0, 75.0)
        recovered = []
        for half_extent in (120.0, 200.0, 280.0):
            transform = ViewGeoTransform(
                self._corners(center, 90.0, half_extent, half_extent),
                768,
                768,
            )
            pixel = transform.latlng_to_pixel(target)
            recovered.append(transform.pixel_to_latlng(pixel))
        for value in recovered:
            self.assertLess(search_engine._haversine_m(*target, *value), 0.05)
        self.assertLess(search_engine._haversine_m(*recovered[0], *recovered[2]), 0.05)

    def test_navigation_vector_signs_for_cardinal_headings(self):
        center = [30.0, 120.0]
        for heading in (0.0, 90.0, 180.0, 270.0):
            forward = navigation_vector(center, self._move(center, heading, 100.0), heading)
            right = navigation_vector(center, self._move(center, heading + 90.0, 100.0), heading)
            behind = navigation_vector(center, self._move(center, heading + 180.0, 100.0), heading)
            self.assertAlmostEqual(forward["forward_offset_m"], 100.0, delta=0.2)
            self.assertAlmostEqual(forward["right_offset_m"], 0.0, delta=0.2)
            self.assertEqual(forward["direction_label"], "forward")
            self.assertAlmostEqual(right["right_offset_m"], 100.0, delta=0.2)
            self.assertEqual(right["direction_label"], "right")
            self.assertLess(behind["forward_offset_m"], -99.0)
            self.assertEqual(behind["direction_label"], "behind")

    def test_mask_forward_corridor_entry_exit_and_footprint_overlap(self):
        center = [30.0, 120.0]
        corners = self._corners(center, 0.0, 100.0, 100.0)
        footprint = self._corners(center, 0.0, 8.0, 10.0)
        (ROOT / "out").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as temp_dir:
            mask = np.zeros((101, 101), np.uint8)
            mask[18:28, :] = 255
            mask_path = Path(temp_dir) / "road.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            analyzer = NavigationGeometryAnalyzer(
                current_position=center,
                heading_deg=0.0,
                view_corners={"main": corners},
                view_sizes={"main": [101, 101]},
                agent_footprint=footprint,
                event_type="PASS",
            )
            analysis = analyzer.analyze({
                "candidate_id": "road",
                "view_id": "main",
                "target_point_2d": [50, 27],
                "mask_path": str(mask_path),
            })
        self.assertTrue(analysis["available"])
        self.assertTrue(analysis["forward_path"]["intersects"])
        self.assertGreater(analysis["forward_path"]["entry_distance_m"], 0)
        self.assertGreater(
            analysis["forward_path"]["exit_distance_m"],
            analysis["forward_path"]["entry_distance_m"],
        )
        self.assertGreater(analysis["forward_path"]["crossing_width_m"], 0)
        self.assertFalse(analysis["mask_intersects_agent_footprint"])

    def test_forward_sector_uses_inclusive_boundaries_and_nearest_pixel(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[5, 10] = 255
        mask[5, 15] = 255
        mask[15, 10] = 255
        result = forward_sector_min_distance_px(mask, [10, 10], half_angle_deg=45.0)
        self.assertEqual(result, {"intersects": True, "minimum_distance_px": 5.0})

    def test_forward_sector_rejects_pixels_below_or_outside_sector(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[15, 10] = 255
        mask[9, 20] = 255
        self.assertEqual(
            forward_sector_min_distance_px(mask, [10, 10]),
            {"intersects": False, "minimum_distance_px": None},
        )

    def test_search_renders_only_scale_five_and_calls_qwen_once(self):
        rendered = []

        def corners(center, _observation, scale_factor, angle_deg):
            rendered.append((list(center), float(scale_factor), float(angle_deg)))
            delta = float(scale_factor) * 0.00001
            lat, lng = center
            return np.asarray([
                [lat - delta, lng - delta], [lat - delta, lng + delta],
                [lat + delta, lng + delta], [lat + delta, lng - delta],
            ])

        qwen_result = {
            "dest_present": False,
            "tool_trace": [],
            "view_paths": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(search_engine, "preflight_visual_grounding"), \
                patch.object(search_engine, "generate_view_corners_with_scale", side_effect=corners), \
                patch.object(search_engine, "create_view_image", return_value=np.zeros((32, 32, 3), dtype=np.uint8)), \
                patch.object(search_engine, "draw_pos_on_patch"), \
                patch.object(search_engine.cv2, "imwrite", return_value=True), \
                patch.object(search_engine, "qwen_locate_bbox_in_view", return_value=qwen_result) as locate:
            found, destination, _views, _zoom = search_engine.search_and_reach_destination(
                instr_id="route_e1",
                ob={},
                dest_desc="road",
                start_pos_latlng=[30.0, 120.0],
                heading_deg=90.0,
                out_dir=temp_dir,
            )
            log = json.loads((Path(temp_dir) / "route_e1_search.json").read_text(encoding="utf-8"))

        self.assertFalse(found)
        self.assertIsNone(destination)
        self.assertEqual(locate.call_count, 1)
        self.assertEqual([item[1] for item in rendered], [5.0])
        self.assertTrue(all(item[0] == [30.0, 120.0] for item in rendered))
        self.assertEqual(set(locate.call_args.kwargs["view_paths"]), {"main"})
        self.assertEqual(locate.call_args.kwargs["view_scales"], {"main": 5.0})
        self.assertEqual(log["final_pos"], [30.0, 120.0])
        self.assertEqual(len(log["steps"]), 1)
        self.assertEqual(log["steps"][0]["pos"], [30.0, 120.0])
        self.assertNotIn("attempt_scales", log)
        self.assertNotIn("max_scales", log)

    def test_single_scale_render_does_not_request_auxiliary_views(self):
        def render(_corners, _observation, save_path, out_px):
            self.assertIn("_main_", save_path)
            return np.zeros((32, 32, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(search_engine, "preflight_visual_grounding"), \
                patch.object(search_engine, "generate_view_corners_with_scale", return_value=np.zeros((4, 2))), \
                patch.object(search_engine, "create_view_image", side_effect=render), \
                patch.object(search_engine, "draw_pos_on_patch"), \
                patch.object(search_engine.cv2, "imwrite", return_value=True), \
                patch.object(search_engine, "qwen_locate_bbox_in_view", return_value={"dest_present": False}) as locate:
            search_engine.search_and_reach_destination(
                "route_e1", {}, "road", [30.0, 120.0], 0.0, temp_dir
            )
            log = json.loads((Path(temp_dir) / "route_e1_search.json").read_text(encoding="utf-8"))
        self.assertEqual(locate.call_count, 1)
        self.assertEqual(set(locate.call_args.kwargs["view_paths"]), {"main"})
        self.assertEqual(log["steps"][0]["render_errors"], {})

    def test_main_scale_render_failure_skips_qwen(self):
        def render(_corners, _observation, save_path, out_px):
            if "_main_" in save_path:
                return None
            return np.zeros((32, 32, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(search_engine, "preflight_visual_grounding"), \
                patch.object(search_engine, "generate_view_corners_with_scale", return_value=np.zeros((4, 2))), \
                patch.object(search_engine, "create_view_image", side_effect=render), \
                patch.object(search_engine, "draw_pos_on_patch"), \
                patch.object(search_engine.cv2, "imwrite", return_value=True), \
                patch.object(search_engine, "qwen_locate_bbox_in_view") as locate:
            found, destination, views, zoom = search_engine.search_and_reach_destination(
                "route_e1", {}, "road", [30.0, 120.0], 0.0, temp_dir
            )
        self.assertFalse(found)
        self.assertIsNone(destination)
        self.assertEqual(views, [])
        self.assertIsNone(zoom)
        locate.assert_not_called()

    def test_selected_target_uses_scale_five_corners_for_coordinates(self):
        corners_by_scale = {}

        def corners(center, _observation, scale_factor, angle_deg):
            delta = float(scale_factor) * 0.00001
            lat, lng = center
            value = np.asarray([
                [lat - delta, lng - delta],
                [lat - delta, lng + delta],
                [lat + delta, lng + delta],
                [lat + delta, lng - delta],
            ])
            corners_by_scale[float(scale_factor)] = value
            return value

        qwen_result = {
            "dest_present": True,
            "bbox_2d": [100, 120, 180, 200],
            "target_point_2d": [140, 160],
            "confidence": 0.8,
            "view_paths": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(search_engine, "preflight_visual_grounding"), \
                patch.object(search_engine, "generate_view_corners_with_scale", side_effect=corners), \
                patch.object(search_engine, "create_view_image", return_value=np.zeros((768, 768, 3), dtype=np.uint8)), \
                patch.object(search_engine, "draw_pos_on_patch"), \
                patch.object(search_engine.cv2, "imwrite", return_value=True), \
                patch.object(search_engine, "qwen_locate_bbox_in_view", return_value=qwen_result), \
                patch.object(search_engine, "_patch_xy_to_latlng", return_value=[30.1, 120.1]) as project:
            found, destination, views, _zoom = search_engine.search_and_reach_destination(
                "route_e1", {}, "road", [30.0, 120.0], 0.0, temp_dir
            )
        self.assertTrue(found)
        self.assertEqual(destination, [30.1, 120.1])
        np.testing.assert_allclose(views[0], corners_by_scale[5.0])
        np.testing.assert_allclose(project.call_args_list[0].args[2], corners_by_scale[5.0])

    def test_final_bbox_corners_and_qwen_point_are_converted_locally(self):
        qwen_result = {
            "dest_present": True,
            "candidate_bbox_2d": [100, 120, 180, 200],
            "bbox_2d": [100, 120, 180, 200],
            "final_bbox_2d": [10, 20, 40, 60],
            "target_point_2d": [140, 160],
            "confidence": 0.8,
            "view_paths": {},
        }

        def project(u, v, _corners, _observation):
            return [float(u), float(v)]

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(search_engine, "preflight_visual_grounding"), \
                patch.object(search_engine, "generate_view_corners_with_scale", return_value=np.zeros((4, 2))), \
                patch.object(search_engine, "create_view_image", return_value=np.zeros((768, 768, 3), dtype=np.uint8)), \
                patch.object(search_engine, "draw_pos_on_patch"), \
                patch.object(search_engine.cv2, "imwrite", return_value=True), \
                patch.object(search_engine, "qwen_locate_bbox_in_view", return_value=qwen_result), \
                patch.object(search_engine, "_patch_xy_to_latlng", side_effect=project):
            found, destination, _views, _zoom = search_engine.search_and_reach_destination(
                "route_goal", {}, "building", [30.0, 120.0], 0.0, temp_dir,
                is_final_goal=True,
            )
        self.assertTrue(found)
        self.assertEqual(destination, [140.0, 160.0])
        self.assertEqual(
            qwen_result["final_bbox_latlng"],
            [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]],
        )

    def test_largest_component_and_landmark_centroid_are_mask_internal(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[1:3, 1:3] = 1
        mask[8:16, 7:17] = 1
        cleaned = largest_component(mask)
        self.assertEqual(int(cleaned.sum()), 80)
        x, y = navigation_point(cleaned, "LANDMARK")
        self.assertTrue(cleaned[int(round(y)), int(round(x))])
        self.assertEqual(bbox_from_mask(cleaned), [7.0, 8.0, 17.0, 16.0])

    def test_region_uses_nearest_internal_point_to_view_center(self):
        mask = np.zeros((21, 21), np.uint8)
        mask[1:5, 1:5] = 1
        point = navigation_point(mask, "REGION")
        self.assertEqual(point, [4.0, 4.0])

    def test_boundary_prefers_forward_center_ray_intersection(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3:8, 9:12] = 1
        self.assertEqual(navigation_point(mask, "BOUNDARY"), [10.0, 7.0])

    def test_mask_and_box_iou(self):
        left = np.zeros((4, 4), np.uint8)
        right = np.zeros((4, 4), np.uint8)
        left[:2, :2] = 1
        right[1:3, 1:3] = 1
        self.assertAlmostEqual(mask_iou(left, right), 1 / 7)
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)

    def test_top_k_and_mask_iou_nms(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[2:6, 2:6] = 1
        other = np.zeros((10, 10), np.uint8)
        other[7:9, 7:9] = 1
        proposals = [
            RawProposal(mask, [2, 2, 6, 6], 0.9, 0.9, {}),
            RawProposal(mask.copy(), [2, 2, 6, 6], 0.8, 0.8, {}),
            RawProposal(other, [7, 7, 9, 9], 0.7, 0.7, {}),
        ]
        result = filter_proposals(proposals, max_candidates=2, nms_iou=0.5)
        self.assertEqual([item.score for item in result], [0.9, 0.7])


class ContractTests(unittest.TestCase):
    def test_contract_rejects_more_than_four_concepts(self):
        with self.assertRaises(ValueError):
            parse_concepts([{"text": str(i), "role": "target"} for i in range(5)])

    def test_contract_rejects_duplicate_concept_ids(self):
        with self.assertRaises(ValueError):
            parse_concepts([
                {"concept_id": "x", "text": "one", "role": "target"},
                {"concept_id": "x", "text": "two", "role": "anchor"},
            ])

    def test_view_contract_requires_path_and_role(self):
        with self.assertRaises(ValueError):
            parse_views([{"view_id": "main"}])

    def test_eval_metrics_cover_proposals_regions_and_negatives(self):
        labels = [
            {"id": "a", "entity_kind": "LANDMARK", "annotation": {"absent": False, "bbox_2d": [0, 0, 10, 10]}},
            {"id": "b", "entity_kind": "REGION", "annotation": {"absent": False, "polygon_2d": [[0, 0], [20, 0], [20, 20], [0, 20]]}},
            {"id": "c", "entity_kind": "LANDMARK", "annotation": {"absent": True}},
        ]
        predictions = [
            {"id": "a", "dest_present": True, "bbox_2d": [0, 0, 10, 10], "candidates": [{"bbox_2d": [0, 0, 10, 10]}], "tool_trace": [1]},
            {"id": "b", "dest_present": True, "target_point_2d": [10, 10], "candidates": [{"target_point_2d": [10, 10]}], "latency_seconds": 2},
            {"id": "c", "dest_present": False},
        ]
        metrics = compute_metrics(labels, predictions)
        self.assertEqual(metrics["proposal_recall"]["at_1"], 1.0)
        self.assertEqual(metrics["negative_correct_abandon_rate"], 1.0)
        self.assertEqual(metrics["final_grounding_success_rate"], 1.0)

    def test_manifest_split_and_negative_counts_are_enforced(self):
        records = []
        for index in range(60):
            records.append({
                "id": str(index), "image_path": "x.png", "query": "q", "entity_kind": "LANDMARK",
                "split": "calibration" if index < 20 else "test",
                "annotation": {"absent": index < 15},
            })
        self.assertEqual(validate_manifest(records), [])


class GroundingAgentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "out").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "out")
        self.root = Path(self.temp.name)
        self.image = self.root / "main.png"
        self.assertTrue(cv2.imwrite(str(self.image), np.zeros((768, 768, 3), np.uint8)))

    def tearDown(self):
        self.temp.cleanup()

    def _agent(self, client, max_calls=4):
        return GroundingAgent(
            api_key="key",
            qwen_url="https://qwen.invalid/chat",
            model="qwen3-vl-plus",
            vision_tool_url="http://127.0.0.1:8765",
            backend="sam3",
            qwen_api_timeout=300,
            max_tool_calls=max_calls,
            vision_client=client,
        )

    def _run(self, agent, messages, destination="gray building", view_paths=None, **kwargs):
        with patch("grounding_agent.requests.post", side_effect=[_Response(item) for item in messages]):
            return agent.run(
                destination=destination,
                view_paths=view_paths or {"main": str(self.image)},
                view_scales={"main": 5},
                artifact_dir=str(self.root / "artifacts"),
                skip_preflight=True,
                **kwargs,
            )

    def test_search_qwen_thinking_is_enabled(self):
        agent = self._agent(_VisionClient())
        with patch("grounding_agent.requests.post") as post:
            post.return_value = _Response(_tool_message(
                1,
                "finish_target_selection",
                {"dest_present": False, "confidence": 0.1, "reason": "unclear"},
            ))
            agent.run(
                destination="road",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "thinking-artifacts"),
                skip_preflight=True,
            )
        self.assertIs(post.call_args.kwargs["json"]["enable_thinking"], True)
        self.assertEqual(post.call_args.kwargs["json"]["tool_choice"], "auto")

    def _captured_prompt(self, event_type, destination="road"):
        agent = self._agent(_VisionClient())
        with patch("grounding_agent.requests.post") as post:
            post.return_value = _Response(_tool_message(
                1,
                "finish_target_selection",
                {"dest_present": False, "confidence": 0.1, "reason": "unclear"},
            ))
            agent.run(
                destination=destination,
                view_paths={
                    "narrow": str(self.image),
                    "main": str(self.image),
                    "wide": str(self.image),
                },
                view_scales={"narrow": 3, "main": 5, "wide": 7},
                event_type=event_type,
                relation="cross",
                side="right",
                event_context=json.dumps({
                    "active_event": {
                        "event_id": "e2",
                        "event_type": event_type,
                        "target": {"description": destination},
                        "relation": "cross",
                        "side_relative_to_image_forward": "right",
                        "spatial_constraints": ["near trees"],
                    },
                }),
                artifact_dir=str(self.root / f"{event_type.lower()}-prompt"),
                skip_preflight=True,
            )
        messages = post.call_args.kwargs["json"]["messages"]
        system = messages[0]["content"]
        user_text = " ".join(
            str(item.get("text") or "")
            for item in messages[1]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        )
        return system, user_text

    def test_reach_prompt_is_single_scale_and_image_space_only(self):
        system, user_text = self._captured_prompt("REACH")
        self.assertIn("[Target Selection - Independent Context]", system)
        for stage in range(1, 5):
            self.assertIn(f"{stage}.", system)
        self.assertIn("main/scale-5", system)
        self.assertIn("minimum_distance_px", system)
        self.assertNotIn("target_point_2d", system)
        self.assertNotIn("relation", system.casefold())
        self.assertNotIn("cross", user_text.casefold())
        self.assertNotIn("right", user_text.casefold())
        self.assertIn("有意义的前向间隔", system)
        self.assertIn("REACH通常优先后者", system)
        self.assertIn("不得仅因它距离最小", system)
        self.assertIn("reason必须写明这项额外视觉证据", system)
        self.assertIn("不得用最近、最大、最连续或最高SAM3分数打破平局", system)
        self.assertIn("不是候选评分", system)
        self.assertIn("多个候选相交时不得机械选择最近或最远者", system)
        self.assertIn("bbox中心、标签位置、缩略图中心或SAM3分数", system)
        self.assertNotIn("scale-3", system)
        self.assertNotIn("scale-7", system)
        self.assertNotIn("minimap", system)
        self.assertNotIn("analyze_navigation_geometry", system)
        self.assertIn("REACH", user_text)

    def test_reach_prompt_forbids_historical_direction_and_metric_geometry(self):
        system, user_text = self._captured_prompt("REACH")
        self.assertIn("heading", system)
        self.assertIn("left/right", system)
        self.assertIn("minimum_distance_px", system)
        self.assertNotIn("distance_m", system)
        self.assertNotIn("latitude", system)
        self.assertNotIn("cross-scale", system)
        self.assertIn("road", user_text)
        self.assertNotIn("cross", user_text)

    def test_qwen_tools_expose_no_navigation_geometry_tool(self):
        selection_names = [tool["function"]["name"] for tool in SELECTION_TOOLS]
        point_names = [tool["function"]["name"] for tool in POINT_TOOLS]
        self.assertEqual(selection_names, ["find_visual_candidates", "finish_target_selection"])
        self.assertEqual(point_names, ["finish_navigation_point"])

    def test_selection_and_point_tools_have_disjoint_fields(self):
        selection = SELECTION_TOOLS[1]["function"]["parameters"]["properties"]
        point_fields = POINT_TOOLS[0]["function"]["parameters"]["properties"]
        self.assertIn("candidate_id", selection)
        self.assertNotIn("target_point_2d", selection)
        self.assertNotIn("candidate_id", point_fields)
        point = point_fields["target_point_2d"]
        self.assertEqual(point["minItems"], 2)

    def test_plain_text_response_is_kept_before_qwen_selects_tools(self):
        client = _VisionClient([_candidate()])
        responses = [
            {
                "role": "assistant",
                "content": "I will inspect the visual candidates.",
                "reasoning_content": "The target appearance and relation need visual evidence.",
            },
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "selected after inspection",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [25, 40], "confidence": 0.7, "reason": "reachable point",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in responses],
        ) as post:
            result = self._agent(client, max_calls=2).run(
                destination="gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "auto-tool-artifacts"),
                skip_preflight=True,
            )
        self.assertTrue(result["dest_present"])
        self.assertEqual(result["tool_call_count"], 3)
        self.assertEqual(result["model_turn_count"], 4)
        second_messages = post.call_args_list[1].kwargs["json"]["messages"]
        thinking_index = next(
            index for index, message in enumerate(second_messages)
            if message.get("reasoning_content") == responses[0]["reasoning_content"]
        )
        self.assertEqual(second_messages[thinking_index]["role"], "assistant")
        self.assertEqual(second_messages[thinking_index + 1]["role"], "user")
        self.assertIn("scale-5", second_messages[thinking_index + 1]["content"])
        self.assertIn("minimum_distance_px", second_messages[thinking_index + 1]["content"])
        self.assertNotIn("target_point_2d", second_messages[thinking_index + 1]["content"])

    def test_unique_candidate_can_finish_without_match(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"concept_id": "c1", "text": "gray building", "role": "target"}],
                "view_ids": ["main"],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True, "candidate_id": "cand-001", "confidence": 0.8, "reason": "unique",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [25, 40], "confidence": 0.7, "reason": "reachable point",
            }),
        ])
        self.assertTrue(result["dest_present"])
        self.assertEqual(result["selected_candidate_id"], "cand-001")

    def test_final_goal_does_not_call_tight_bbox_in_reach_pipeline(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "selected final target",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [25, 40], "confidence": 0.85, "reason": "goal point",
            }),
        ], is_final_goal=True)
        self.assertEqual(result["bbox_2d"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(result["raw_bbox_2d"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(result["candidate_bbox_2d"], [10.0, 20.0, 40.0, 60.0])
        self.assertIsNone(result["final_bbox_2d"])
        self.assertEqual(result["bbox_grounding"]["status"], "not_requested")
        self.assertTrue(result["is_final_goal"])
        self.assertEqual(result["target_point_2d"], [25, 40])

    def test_final_bbox_is_not_requested_even_for_final_goal(self):
        client = _VisionClient([_candidate()])
        responses = [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "selected final target",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [500, 500],
                "confidence": 0.8,
                "reason": "relation-aware point may be outside the target",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in responses],
        ) as post:
            result = self._agent(client).run(
                destination="final gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "bbox-retry-artifacts"),
                is_final_goal=True,
                skip_preflight=True,
            )
        self.assertTrue(result["dest_present"])
        self.assertEqual(result["target_point_2d"], [500.0, 500.0])
        self.assertIsNone(result["final_bbox_2d"])
        self.assertEqual(result["bbox_grounding"]["status"], "not_requested")
        self.assertEqual(post.call_count, 3)

    def test_final_bbox_failure_case_preserves_selected_point_without_bbox_stage(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "selected",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [300, 320], "confidence": 0.8, "reason": "point",
            }),
        ], is_final_goal=True)
        self.assertTrue(result["dest_present"])
        self.assertEqual(result["target_point_2d"], [300.0, 320.0])
        self.assertIsNone(result["final_bbox_2d"])
        self.assertEqual(result["bbox_grounding"]["status"], "not_requested")

    def test_find_candidates_returns_only_image_space_forward_sector_facts(self):
        candidate = _candidate()
        candidate["target_point_2d"] = [32, 8]
        candidate["bbox_2d"] = [20, 4, 44, 16]
        mask = np.zeros((64, 64), np.uint8)
        mask[7:12, 15:50] = 255
        mask_path = self.root / "candidate-mask.png"
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        candidate["mask_path"] = str(mask_path)
        client = _VisionClient([candidate])
        center = [30.0, 120.0]
        corners = GeometryTests._corners(center, 0.0, 100.0, 100.0)
        responses = [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "road", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "selected from visual sector evidence",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [30, 20], "confidence": 0.8, "reason": "cross point",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in responses],
        ) as post:
            result = self._agent(client).run(
                destination="road",
                relation="cross",
                view_paths={"main": str(self.image)},
                view_scales={"main": 5},
                view_corners={"main": corners},
                view_sizes={"main": [64, 64]},
                current_position=center,
                heading_deg=0.0,
                agent_footprint=GeometryTests._corners(center, 0.0, 8.0, 10.0),
                event_type="PASS",
                artifact_dir=str(self.root / "geometry-artifacts"),
                skip_preflight=True,
            )
        second_payload = post.call_args_list[1].kwargs["json"]
        tool_results = [
            json.loads(message["content"])
            for message in second_payload["messages"]
            if message.get("role") == "tool"
        ]
        find_result = next(value for value in tool_results if "candidates" in value)
        candidate_result = find_result["candidates"][0]
        self.assertEqual(
            set(candidate_result),
            {"candidate_id", "concept_text", "concept_role", "forward_sector"},
        )
        self.assertEqual(
            set(candidate_result["forward_sector"]),
            {"intersects", "minimum_distance_px"},
        )
        self.assertNotIn("navigation_geometry", result)

    def test_out_of_bounds_pixel_point_is_rejected_before_valid_retry(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.6,
                "reason": "fixed target",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [900, 40],
                "confidence": 0.6,
                "reason": "invalid point",
            }),
            _tool_message(4, "finish_navigation_point", {
                "target_point_2d": [500, 500],
                "confidence": 0.6,
                "reason": "valid relation-aware point outside the mask",
            }),
        ])
        self.assertTrue(result["dest_present"])
        self.assertIn("outside", result["tool_trace"][2]["result"]["error"])
        self.assertEqual(result["target_point_2d"], [500.0, 500.0])

    def test_waypoint_failure_preserves_fixed_candidate_and_skips_bbox(self):
        client = _VisionClient([_candidate()])
        responses = [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.91,
                "reason": "fixed target",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [900, 40],
                "confidence": 0.6,
                "reason": "outside",
            }),
            _tool_message(4, "finish_navigation_point", {
                "target_point_2d": [float("nan"), 40],
                "confidence": 0.5,
                "reason": "non-finite",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in responses],
        ) as post:
            result = self._agent(client).run(
                destination="building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "waypoint-failure"),
                is_final_goal=True,
                skip_preflight=True,
            )
        self.assertFalse(result["dest_present"])
        self.assertEqual(result["selected_candidate_id"], "cand-001")
        self.assertEqual(result["candidate_bbox_2d"], [10.0, 20.0, 40.0, 60.0])
        self.assertIsNone(result["target_point_2d"])
        self.assertIsNone(result["final_bbox_2d"])
        self.assertEqual(
            result["bbox_grounding"]["status"],
            "not_requested_due_to_waypoint_failure",
        )
        self.assertEqual(post.call_count, 4)
        self.assertEqual(len(client.candidate_payloads), 1)

    def test_selection_and_waypoint_use_independent_contexts(self):
        client = _VisionClient([_candidate()])
        responses = [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "road", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.91,
                "reason": "selection-only evidence",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [150, 150],
                "confidence": 0.87,
                "reason": "crosses fixed target",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in responses],
        ) as post:
            result = self._agent(client).run(
                destination="road",
                relation="cross",
                side="right",
                event_type="REACH",
                event_context=json.dumps({
                    "active_event": {
                        "event_id": "e2",
                        "event_type": "REACH",
                        "target": {"description": "road"},
                        "relation": "cross",
                        "side_relative_to_image_forward": "right",
                    },
                }),
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "independent-contexts"),
                skip_preflight=True,
            )
        selection_payload = post.call_args_list[1].kwargs["json"]
        selection_text = json.dumps(selection_payload, ensure_ascii=False)
        self.assertNotIn("finish_navigation_point", selection_text)
        self.assertNotIn("target_point_2d", selection_text)
        self.assertNotIn("cross", selection_text.casefold())
        self.assertNotIn("side_relative_to_image_forward", selection_text)

        point_payload = post.call_args_list[2].kwargs["json"]
        self.assertEqual(len(point_payload["messages"]), 2)
        self.assertEqual(
            [tool["function"]["name"] for tool in point_payload["tools"]],
            ["finish_navigation_point"],
        )
        point_text = json.dumps(point_payload, ensure_ascii=False)
        self.assertIn("cross", point_text)
        self.assertIn("cand-001", point_text)
        self.assertNotIn("forward_sector", point_text)
        self.assertNotIn("selection-only evidence", point_text)
        image_items = [
            item
            for item in point_payload["messages"][1]["content"]
            if item.get("type") == "image_url"
        ]
        self.assertEqual(len(image_items), 2)
        self.assertTrue(Path(result["view_paths"]["selected_target_point_input"]).is_file())
        self.assertEqual([entry["phase"] for entry in result["tool_trace"]], [
            "selection", "selection", "waypoint",
        ])

    def test_relational_query_is_resolved_by_qwen_from_sam_candidates(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [
                    {"text": "building", "role": "target"},
                    {"text": "parking lot", "role": "anchor"},
                ],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True, "candidate_id": "cand-001", "confidence": 0.9, "reason": "Qwen resolved the relation",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [25, 40], "confidence": 0.8, "reason": "relation point",
            }),
        ], destination="building right of parking", view_paths={
            "narrow": str(self.image),
            "main": str(self.image),
            "wide": str(self.image),
        })
        self.assertTrue(result["dest_present"])
        self.assertEqual(client.match_payloads, [])
        self.assertEqual(
            [item["view_id"] for item in client.candidate_payloads[0]["views"]],
            ["main"],
        )
        self.assertEqual(client.candidate_payloads[0]["max_candidates_per_view"], 12)
        self.assertNotIn("roi", client.candidate_payloads[0])
        self.assertEqual(result["provider"], "qwen3-vl+sam3")

    def test_non_main_candidate_is_rejected(self):
        client = _VisionClient([_candidate(view_id="wide")])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}], "view_ids": ["main"],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True, "candidate_id": "cand-001", "confidence": 0.7, "reason": "wide",
            }),
            _tool_message(3, "finish_target_selection", {
                "dest_present": False, "confidence": 0.2, "reason": "wide only",
            }),
        ])
        self.assertFalse(result["dest_present"])
        self.assertIn("main view", result["tool_trace"][1]["result"]["error"])

    def test_unknown_candidate_is_rejected(self):
        client = _VisionClient([_candidate()])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}], "view_ids": ["main"],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True, "candidate_id": "invented", "confidence": 1, "reason": "invented",
            }),
            _tool_message(3, "finish_target_selection", {
                "dest_present": False, "confidence": 0, "reason": "invalid",
            }),
        ])
        self.assertFalse(result["dest_present"])
        self.assertIn("not generated", result["tool_trace"][1]["result"]["error"])

    def test_query_rewrite_limit_is_enforced(self):
        client = _VisionClient([])
        calls = [
            _tool_message(index, "find_visual_candidates", {
                "concepts": [{"text": f"building {index}", "role": "target"}], "view_ids": ["main"],
            }) for index in (1, 2, 3)
        ]
        calls.append(_tool_message(4, "finish_target_selection", {
            "dest_present": False, "confidence": 0, "reason": "none",
        }))
        result = self._run(self._agent(client), calls)
        self.assertFalse(result["dest_present"])
        self.assertIn("at most once", result["tool_trace"][2]["result"]["error"])

    def test_tool_call_budget_exhaustion_raises_recoverable_error(self):
        client = _VisionClient([])
        with self.assertRaises(GroundingAgentError):
            self._run(self._agent(client, max_calls=1), [
                _tool_message(1, "find_visual_candidates", {
                    "concepts": [{"text": "building", "role": "target"}], "view_ids": ["main"],
                }),
            ])

    def test_each_qwen_request_has_independent_five_minute_timeout(self):
        client = _VisionClient([_candidate()])
        messages = [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "building", "role": "target"}],
                "view_ids": ["main"],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True,
                "candidate_id": "cand-001",
                "confidence": 0.8,
                "reason": "done",
            }),
            _tool_message(3, "finish_navigation_point", {
                "target_point_2d": [25, 40], "confidence": 0.8, "reason": "point",
            }),
        ]
        with patch(
            "grounding_agent.requests.post",
            side_effect=[_Response(item) for item in messages],
        ) as request:
            result = self._agent(client).run(
                destination="gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "fixed-timeout"),
                skip_preflight=True,
            )
        self.assertTrue(result["dest_present"])
        self.assertEqual([call.kwargs["timeout"] for call in request.call_args_list], [300, 300, 300])

    def test_service_failure_is_infrastructure_error(self):
        client = _VisionClient(fail=True)
        with self.assertRaises(GroundingInfrastructureError):
            self._agent(client).preflight()

    def test_context_candidate_cannot_be_submitted_as_target(self):
        client = _VisionClient([_candidate(role="context")])
        result = self._run(self._agent(client), [
            _tool_message(1, "find_visual_candidates", {
                "concepts": [{"text": "gray building", "role": "target"}],
            }),
            _tool_message(2, "finish_target_selection", {
                "dest_present": True, "candidate_id": "cand-001", "confidence": 0.8, "reason": "wrong role",
            }),
            _tool_message(3, "finish_target_selection", {
                "dest_present": False, "confidence": 0.0, "reason": "no target candidate",
            }),
        ])
        self.assertFalse(result["dest_present"])
        self.assertIn("target role", result["tool_trace"][1]["result"]["error"])

    def test_backend_mismatch_is_infrastructure_error(self):
        client = _VisionClient()
        client.health = lambda: {"ok": True, "prewarmed": True, "backend": "grounded-sam2"}
        with self.assertRaises(GroundingInfrastructureError):
            self._agent(client).preflight()

    def test_agent_error_returns_tool_agent_failure(self):
        with patch.object(
            search_engine,
            "preflight_visual_grounding",
            return_value={"ok": True, "prewarmed": True, "backend": "sam3"},
        ), patch("grounding_agent.GroundingAgent") as agent_type:
            agent_type.return_value.run.side_effect = GroundingAgentError("Qwen timeout")
            result = search_engine.qwen_locate_bbox_in_view(
                "gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "agent-error"),
            )
        self.assertEqual(result["provider"], "tool-agent")
        self.assertFalse(result["dest_present"])
        self.assertNotIn("fallback_used", result)
        self.assertIn("Qwen timeout", result["reason"])

    def test_ambiguous_tool_result_is_returned_without_direct_qwen_retry(self):
        tool_result = {
            "dest_present": False,
            "reason": "ambiguous candidates",
            "provider": "tool-agent",
            "tool_trace": [{
                "tool": "find_visual_candidates",
                "result": {"candidate_count": 2, "error": ""},
            }],
            "view_paths": {"candidate_sheet": "sheet.png"},
        }
        with patch.object(
            search_engine,
            "preflight_visual_grounding",
            return_value={"ok": True, "prewarmed": True, "backend": "sam3"},
        ), patch("grounding_agent.GroundingAgent") as agent_type:
            agent_type.return_value.run.return_value = tool_result
            result = search_engine.qwen_locate_bbox_in_view(
                "gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "ambiguous"),
            )
        self.assertFalse(result["dest_present"])
        self.assertEqual(result["provider"], "tool-agent")
        self.assertNotIn("fallback_used", result)
        self.assertEqual(result["tool_trace"], tool_result["tool_trace"])

    def test_fake_qwen_and_visual_services_execute_http_tool_loop(self):
        candidate = _candidate()

        class VisionHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def send_json(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_json({"ok": True, "prewarmed": True, "backend": "sam3"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/v1/candidates":
                    self.send_json({"backend": "sam3", "candidates": [candidate], "contact_sheet_paths": []})
                else:
                    self.send_json({"provider": "dinov3", "matches": [], "visualization_paths": []})

        class QwenHandler(BaseHTTPRequestHandler):
            calls = 0

            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                json.loads(self.rfile.read(length) or b"{}")
                QwenHandler.calls += 1
                if QwenHandler.calls == 1:
                    message = _tool_message(1, "find_visual_candidates", {
                        "concepts": [{"text": "building", "role": "target"}], "view_ids": ["main"],
                    })
                elif QwenHandler.calls == 2:
                    message = _tool_message(2, "finish_target_selection", {
                        "dest_present": True, "candidate_id": "cand-001", "confidence": 0.8, "reason": "http integration",
                    })
                else:
                    message = _tool_message(3, "finish_navigation_point", {
                        "target_point_2d": [25, 40], "confidence": 0.8, "reason": "http point",
                    })
                body = json.dumps({"choices": [{"message": message}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        vision_server = ThreadingHTTPServer(("127.0.0.1", 0), VisionHandler)
        qwen_server = ThreadingHTTPServer(("127.0.0.1", 0), QwenHandler)
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (vision_server, qwen_server)
        ]
        for thread in threads:
            thread.start()
        try:
            agent = GroundingAgent(
                api_key="fake",
                qwen_url=f"http://127.0.0.1:{qwen_server.server_port}/chat",
                model="qwen3-vl-plus",
                vision_tool_url=f"http://127.0.0.1:{vision_server.server_port}",
                backend="sam3",
                qwen_api_timeout=300,
                max_tool_calls=4,
            )
            result = agent.run(
                destination="gray building",
                view_paths={"main": str(self.image)},
                artifact_dir=str(self.root / "http-artifacts"),
            )
            self.assertTrue(result["dest_present"])
            self.assertEqual(result["selected_candidate_id"], "cand-001")
        finally:
            vision_server.shutdown()
            qwen_server.shutdown()
            vision_server.server_close()
            qwen_server.server_close()


if __name__ == "__main__":
    unittest.main()
