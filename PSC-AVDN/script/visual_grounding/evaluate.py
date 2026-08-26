"""Metrics for the evaluation-only zero-shot visual-grounding set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np

from .annotation_tool import read_jsonl, validate_manifest
from .geometry import box_iou


def _point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32)
    return bool(len(contour) >= 3 and cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0)


def _candidate_hit(candidate: Dict[str, Any], annotation: Dict[str, Any], kind: str) -> bool:
    if annotation.get("bbox_2d"):
        return box_iou(candidate.get("bbox_2d") or [0, 0, 0, 0], annotation["bbox_2d"]) >= 0.5
    polygon = annotation.get("polygon_2d") or []
    point = candidate.get("target_point_2d")
    if not point and candidate.get("bbox_2d"):
        x1, y1, x2, y2 = candidate["bbox_2d"]
        point = [(x1 + x2) / 2, (y1 + y2) / 2]
    return bool(point and _point_in_polygon(point, polygon))


def compute_metrics(labels: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    predicted = {str(item.get("id")): item for item in predictions}
    positive = 0
    negatives = 0
    recalls = {1: 0, 5: 0, 10: 0}
    landmark_ious: List[float] = []
    region_hits: List[float] = []
    negative_correct = 0
    final_success = 0
    calls: List[float] = []
    latencies: List[float] = []
    missing: List[str] = []
    for label in labels:
        item_id = str(label.get("id"))
        prediction = predicted.get(item_id)
        if prediction is None:
            missing.append(item_id)
            prediction = {}
        annotation = label.get("annotation") or {}
        if annotation.get("absent"):
            negatives += 1
            if not prediction.get("dest_present"):
                negative_correct += 1
                final_success += 1
        else:
            positive += 1
            candidates = prediction.get("candidates") or []
            for k in recalls:
                if any(_candidate_hit(item, annotation, str(label.get("entity_kind"))) for item in candidates[:k]):
                    recalls[k] += 1
            selected = prediction.get("selected_candidate") or prediction
            hit = bool(prediction.get("dest_present") and _candidate_hit(selected, annotation, str(label.get("entity_kind"))))
            final_success += int(hit)
            kind = str(label.get("entity_kind") or "").upper()
            if kind == "LANDMARK" and annotation.get("bbox_2d") and selected.get("bbox_2d"):
                landmark_ious.append(box_iou(selected["bbox_2d"], annotation["bbox_2d"]))
            elif kind in {"REGION", "BOUNDARY", "CORRIDOR"}:
                region_hits.append(float(hit))
        if prediction.get("tool_call_count") is not None:
            calls.append(float(prediction["tool_call_count"]))
        elif isinstance(prediction.get("tool_trace"), list):
            calls.append(float(len(prediction["tool_trace"])))
        if prediction.get("latency_seconds") is not None:
            latencies.append(float(prediction["latency_seconds"]))
    count = max(1, len(labels))
    percentile = lambda values, q: float(np.percentile(values, q)) if values else None
    return {
        "items": len(labels),
        "positive_items": positive,
        "negative_items": negatives,
        "proposal_recall": {f"at_{k}": recalls[k] / max(1, positive) for k in sorted(recalls)},
        "landmark_mean_iou": float(np.mean(landmark_ious)) if landmark_ious else None,
        "region_center_hit_rate": float(np.mean(region_hits)) if region_hits else None,
        "negative_correct_abandon_rate": negative_correct / max(1, negatives),
        "final_grounding_success_rate": final_success / count,
        "tool_calls_mean": float(np.mean(calls)) if calls else None,
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "missing_prediction_ids": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot visual-grounding predictions")
    parser.add_argument("labels", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    labels = read_jsonl(args.labels)
    errors = validate_manifest(labels, strict_size=True)
    if errors:
        raise SystemExit("\n".join(errors))
    labels = [item for item in labels if item.get("split") == args.split]
    metrics = compute_metrics(labels, read_jsonl(args.predictions))
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

