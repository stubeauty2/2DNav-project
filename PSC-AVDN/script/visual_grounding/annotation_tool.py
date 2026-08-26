"""Small OpenCV annotation UI for the 60-item zero-shot grounding benchmark.

The manifest is evaluation-only. It is never consumed by a model-training
path. Required manifest fields are ``id``, ``image_path``, ``query``,
``entity_kind``, and ``split`` (``calibration`` or ``test``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def validate_manifest(records: Sequence[Dict[str, Any]], strict_size: bool = True) -> List[str]:
    errors: List[str] = []
    if strict_size and len(records) != 60:
        errors.append(f"manifest must contain 60 items, found {len(records)}")
    ids = [str(item.get("id") or "") for item in records]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        errors.append("manifest ids must be non-empty and unique")
    for index, item in enumerate(records):
        missing = [key for key in ("image_path", "query", "entity_kind", "split") if not item.get(key)]
        if missing:
            errors.append(f"item {ids[index] or index} missing {missing}")
        if item.get("split") not in {"calibration", "test"}:
            errors.append(f"item {ids[index] or index} has invalid split")
        if str(item.get("entity_kind") or "").upper() not in {"LANDMARK", "REGION", "BOUNDARY", "CORRIDOR"}:
            errors.append(f"item {ids[index] or index} has invalid entity_kind")
    calibration = sum(item.get("split") == "calibration" for item in records)
    test = sum(item.get("split") == "test" for item in records)
    negatives = sum(bool((item.get("annotation") or {}).get("absent")) for item in records)
    if strict_size and (calibration, test) != (20, 40):
        errors.append(f"expected calibration/test=20/40, found {calibration}/{test}")
    if strict_size and negatives != 15:
        errors.append(f"expected 15 absent targets, found {negatives}")
    return errors


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def _polygon_ui(window: str, image: np.ndarray) -> List[List[float]]:
    points: List[List[float]] = []
    display = image.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([float(x), float(y)])
            display = image.copy()
            for start, end in zip(points, points[1:]):
                cv2.line(display, tuple(map(int, start)), tuple(map(int, end)), (0, 255, 255), 2)
            for point in points:
                cv2.circle(display, tuple(map(int, point)), 4, (0, 0, 255), -1)

    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(points) >= 3:
            return points
        if key in (27, ord("q")):
            return []


def annotate(manifest_path: Path, output_path: Path) -> None:
    records = read_jsonl(manifest_path)
    prior = {str(item.get("id")): item for item in read_jsonl(output_path)} if output_path.exists() else {}
    window = "visual-grounding labels: b=bbox p=polygon n=absent s=skip q=quit"
    try:
        for source in records:
            item = dict(prior.get(str(source.get("id")), source))
            if item.get("annotation"):
                continue
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(item["image_path"])
            display = image.copy()
            cv2.putText(display, str(item["query"])[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                continue
            if key == ord("n"):
                item["annotation"] = {"absent": True}
            elif key == ord("b"):
                x, y, width, height = cv2.selectROI(window, image, showCrosshair=True)
                if width <= 0 or height <= 0:
                    continue
                item["annotation"] = {
                    "absent": False,
                    "bbox_2d": [float(x), float(y), float(x + width), float(y + height)],
                }
            elif key == ord("p"):
                polygon = _polygon_ui(window, image)
                if len(polygon) < 3:
                    continue
                item["annotation"] = {"absent": False, "polygon_2d": polygon}
            else:
                continue
            prior[str(item["id"])] = item
            ordered = [prior[str(record["id"])] for record in records if str(record["id"]) in prior]
            _write_jsonl(output_path, ordered)
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label or validate visual-grounding evaluation data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    label = subparsers.add_parser("annotate")
    label.add_argument("manifest", type=Path)
    label.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("labels", type=Path)
    validate.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "annotate":
        annotate(args.manifest, args.output)
        return
    errors = validate_manifest(read_jsonl(args.labels), strict_size=not args.allow_partial)
    if errors:
        raise SystemExit("\n".join(errors))
    print("annotation manifest is valid")


if __name__ == "__main__":
    main()

