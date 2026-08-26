"""Generate offline HTML reports for tool-agent navigation runs.

The report intentionally uses only the Python standard library.  Images stay in
their route output directories and are referenced relatively, so a report can
be opened directly without a web server or network access.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _polygon_center(polygon: Sequence[Sequence[float]]) -> Tuple[float, float] | None:
    points = [p for p in polygon if isinstance(p, Sequence) and len(p) >= 2]
    if not points:
        return None
    return (
        sum(float(p[0]) for p in points) / len(points),
        sum(float(p[1]) for p in points) / len(points),
    )


def _gt_trajectory(prediction: Mapping[str, Any]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for polygon in prediction.get("gt_path_corners", []):
        center = _polygon_center(polygon)
        if center is not None:
            result.append(center)
    return result


def _pred_trajectory(prediction: Mapping[str, Any]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for point in prediction.get("trajectory", []):
        if isinstance(point, Sequence) and len(point) >= 2:
            result.append((float(point[0]), float(point[1])))
    return result


def _prediction_polygons(prediction: Mapping[str, Any]) -> List[List[List[float]]]:
    polygons: List[List[List[float]]] = []
    for item in prediction.get("path_corners", []):
        polygon = item[0] if isinstance(item, Sequence) and len(item) == 2 else item
        if (
            isinstance(polygon, Sequence)
            and len(polygon) >= 3
            and all(isinstance(point, Sequence) and len(point) >= 2 for point in polygon)
        ):
            polygons.append([[float(point[0]), float(point[1])] for point in polygon])
    return polygons


def _route_metadata(dataset_dir: Path) -> Dict[str, Mapping[str, Any]]:
    data_path = dataset_dir / "test_unseen_full_data.json"
    records = json.loads(data_path.read_text(encoding="utf-8"))
    return {
        f"{record['map_name']}__{record['route_index']}": record
        for record in records
    }


def _parse_records(plans_path: Path) -> Dict[str, Mapping[str, Any]]:
    records: Dict[str, Mapping[str, Any]] = {}
    for line in plans_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        plan_id = str(record.get("plan_id") or "")
        if plan_id:
            records[plan_id] = record
    return records


def _trajectory_map_images(
    prediction: Mapping[str, Any],
    route_dir: Path,
    dataset_dir: Path,
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """Render GT and predicted trajectories on the same satellite-map crop."""
    route_id = str(prediction.get("instr_id") or route_dir.name)
    record = metadata.get(route_id)
    if record is None:
        return {}
    map_name = route_id.split("__", 1)[0]
    map_path = dataset_dir / "train_images" / f"{map_name}.tif"
    if not map_path.exists():
        return {}
    base = Image.open(map_path).convert("RGB")
    width, height = base.size
    lat_min, lng_min = (float(x) for x in record["gps_botm_left"])
    lat_max, lng_max = (float(x) for x in record["gps_top_right"])

    def project(point: Sequence[float]) -> Tuple[float, float]:
        lat, lng = float(point[0]), float(point[1])
        x = (lng - lng_min) / (lng_max - lng_min) * width
        y = (lat_max - lat) / (lat_max - lat_min) * height
        return x, y

    gt_polygons = [
        [[float(v) for v in point[:2]] for point in polygon]
        for polygon in prediction.get("gt_path_corners", [])
        if isinstance(polygon, Sequence) and len(polygon) >= 3
    ]
    pred_polygons = _prediction_polygons(prediction)
    final_grounding = prediction.get("final_target_grounding") or {}
    final_bbox_latlng = final_grounding.get("bbox_latlng") or []
    final_target_polygons = [final_bbox_latlng] if (
        isinstance(final_bbox_latlng, Sequence)
        and len(final_bbox_latlng) >= 3
        and isinstance(final_bbox_latlng[0], Sequence)
    ) else []
    raw_destination = record.get("destination") or []
    if (
        isinstance(raw_destination, Sequence)
        and raw_destination
        and isinstance(raw_destination[0], Sequence)
        and raw_destination[0]
        and isinstance(raw_destination[0][0], (int, float))
    ):
        destination_polygons = [raw_destination]
    else:
        destination_polygons = [
            polygon for polygon in raw_destination
            if isinstance(polygon, Sequence) and len(polygon) >= 3
        ]
    projected_all = [
        project(point)
        for polygon in gt_polygons + pred_polygons + destination_polygons + final_target_polygons
        for point in polygon
    ]
    if not projected_all:
        return {}
    xs, ys = zip(*projected_all)
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    span = max(x2 - x1, y2 - y1)
    margin = max(180.0, span * 0.18)
    crop_x1, crop_x2 = x1 - margin, x2 + margin
    crop_y1, crop_y2 = y1 - margin, y2 + margin
    min_crop = min(1400.0, float(max(width, height)))
    if crop_x2 - crop_x1 < min_crop:
        delta = (min_crop - (crop_x2 - crop_x1)) / 2
        crop_x1, crop_x2 = crop_x1 - delta, crop_x2 + delta
    if crop_y2 - crop_y1 < min_crop:
        delta = (min_crop - (crop_y2 - crop_y1)) / 2
        crop_y1, crop_y2 = crop_y1 - delta, crop_y2 + delta
    crop_x1, crop_y1 = max(0, int(crop_x1)), max(0, int(crop_y1))
    crop_x2, crop_y2 = min(width, int(math.ceil(crop_x2))), min(height, int(math.ceil(crop_y2)))
    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

    def render(show_gt: bool, show_pred: bool, output_name: str) -> Path:
        image = base.copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def draw_track(
            polygons: Sequence[Sequence[Sequence[float]]],
            line_color: Tuple[int, int, int, int],
            fill_color: Tuple[int, int, int, int],
            prefix: str,
        ) -> None:
            centers: List[Tuple[float, float]] = []
            for index, polygon in enumerate(polygons):
                points = [project(point) for point in polygon]
                center = (
                    sum(point[0] for point in points) / len(points),
                    sum(point[1] for point in points) / len(points),
                )
                centers.append(center)
                is_terminal = index == len(polygons) - 1
                draw.polygon(
                    points,
                    fill=fill_color,
                    outline=line_color,
                    width=8 if is_terminal else 3,
                )
                radius = 7 if index in (0, len(polygons) - 1) else 4
                draw.ellipse(
                    (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                    fill=line_color,
                )
                draw.text(
                    (center[0] + 7, center[1] - 12),
                    f"{prefix}{index}",
                    fill=(255, 255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )
            if len(centers) >= 2:
                draw.line(centers, fill=line_color, width=7, joint="curve")

        def draw_destination_boxes() -> None:
            for polygon in destination_polygons:
                points = [project(point) for point in polygon]
                draw.polygon(points, fill=None, outline=(250, 204, 21, 255), width=8)
                label_x = min(point[0] for point in points)
                label_y = min(point[1] for point in points)
                draw.text(
                    (label_x + 9, label_y + 8),
                    "GT destination bbox",
                    fill=(250, 204, 21, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )

        def draw_final_target_box() -> None:
            for polygon in final_target_polygons:
                points = [project(point) for point in polygon]
                draw.polygon(points, fill=None, outline=(6, 182, 212, 255), width=8)
                label_x = min(point[0] for point in points)
                label_y = min(point[1] for point in points)
                draw.text(
                    (label_x + 9, label_y + 8),
                    "Pred final target bbox",
                    fill=(6, 182, 212, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )

        def draw_turn_arrows() -> None:
            arrow_length = max(45.0, min(150.0, span * 0.10))
            for trace in prediction.get("physical_trajectory", []):
                if str(trace.get("event_type") or "") != "TURN":
                    continue
                position = trace.get("position") or []
                if len(position) < 2:
                    continue
                origin = project(position)
                heading = math.radians(float(trace.get("heading_deg") or 0.0))
                direction = (math.sin(heading), -math.cos(heading))
                end = (
                    origin[0] + arrow_length * direction[0],
                    origin[1] + arrow_length * direction[1],
                )
                draw.line((origin, end), fill=(14, 165, 233, 255), width=9)
                left = (
                    end[0] - 22 * direction[0] + 11 * direction[1],
                    end[1] - 22 * direction[1] - 11 * direction[0],
                )
                right = (
                    end[0] - 22 * direction[0] - 11 * direction[1],
                    end[1] - 22 * direction[1] + 11 * direction[0],
                )
                draw.polygon((end, left, right), fill=(14, 165, 233, 255))
                draw.ellipse(
                    (origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8),
                    fill=(14, 165, 233, 255),
                    outline=(255, 255, 255, 255),
                    width=2,
                )
                label = f"{trace.get('event_id') or 'TURN'} {float(trace.get('heading_deg') or 0.0):.0f}°"
                draw.text(
                    (end[0] + 8, end[1] - 13),
                    label,
                    fill=(255, 255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )

        # Destination boxes are contextual bounds. Draw them first without a
        # fill so route lines, points, arrows, and labels remain unobscured.
        if show_gt:
            draw_destination_boxes()
        if show_pred:
            draw_final_target_box()
        if show_gt:
            draw_track(gt_polygons, (34, 197, 94, 255), (34, 197, 94, 55), "G")
        if show_pred:
            draw_track(pred_polygons, (244, 63, 94, 255), (244, 63, 94, 55), "P")
            draw_turn_arrows()
        composed = Image.alpha_composite(image, overlay).convert("RGB").crop(crop_box)
        max_side = 1800
        if max(composed.size) > max_side:
            scale = max_side / max(composed.size)
            composed = composed.resize(
                (max(1, int(composed.width * scale)), max(1, int(composed.height * scale))),
                Image.Resampling.LANCZOS,
            )
        legend = ImageDraw.Draw(composed)
        legend.rectangle((12, 12, 470, 92), fill=(0, 0, 0, 185))
        if show_gt:
            legend.line((24, 29, 70, 29), fill=(34, 197, 94), width=6)
            legend.text((80, 20), "GT", fill=(255, 255, 255))
        if show_pred:
            offset = 105 if show_gt else 0
            legend.line((24 + offset, 47 if show_gt else 29, 70 + offset, 47 if show_gt else 29), fill=(244, 63, 94), width=6)
            legend.text((80 + offset, 38 if show_gt else 20), "Prediction", fill=(255, 255, 255))
            legend.line((245, 29, 291, 29), fill=(14, 165, 233), width=7)
            legend.text((301, 20), "TURN heading", fill=(255, 255, 255))
        if show_gt and destination_polygons:
            legend.rectangle((245, 43, 291, 65), outline=(250, 204, 21), width=5)
            legend.text((301, 44), "GT destination bbox", fill=(255, 255, 255))
        if show_pred and final_target_polygons:
            legend.rectangle((245, 66, 291, 88), outline=(6, 182, 212), width=5)
            legend.text((301, 67), "Pred final target bbox", fill=(255, 255, 255))
        output_path = route_dir / output_name
        composed.save(output_path, quality=94, optimize=True)
        return output_path.relative_to(route_dir)

    return {
        "gt": render(True, False, "trajectory_gt_map.jpg"),
        "pred": render(False, True, "trajectory_prediction_map.jpg"),
        "overlay": render(True, True, "trajectory_gt_prediction_overlay.jpg"),
    }


def _resolve_artifact(route_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path, Path.cwd() / path, route_dir / path.name]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    matches = list(route_dir.rglob(path.name))
    return matches[0].resolve() if matches else None


def _relative_artifact(route_dir: Path, value: Any) -> Path | None:
    resolved = _resolve_artifact(route_dir, value)
    if resolved is None:
        return None
    try:
        return resolved.relative_to(route_dir.resolve())
    except ValueError:
        return None


def _render_localization_overlay(
    route_dir: Path,
    event_id: str,
    run_index: int,
    attempt_index: int,
    call_kind: str,
    call: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> List[Path]:
    selected = call.get("selected_candidate") or {}
    source_value = selected.get("source_image_path")
    attempt_views = attempt.get("view_paths") or {}
    if not source_value:
        source_value = attempt_views.get("main")
    source_path = _resolve_artifact(route_dir, source_value)
    if source_path is None:
        return []
    image = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_path = _resolve_artifact(route_dir, call.get("mask_path") or selected.get("mask_path"))
    mask_bbox: Tuple[int, int, int, int] | None = None
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        mask_bbox = mask.getbbox()
        tint = Image.new("RGBA", image.size, (0, 229, 255, 105))
        tint.putalpha(mask.point(lambda value: min(120, value)))
        overlay = Image.alpha_composite(overlay, tint)
    draw = ImageDraw.Draw(overlay)
    bbox = call.get("final_bbox_2d") or call.get("bbox_2d") or selected.get("bbox_2d")
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 52, 52, 255), width=5)
        draw.text((x1 + 4, max(2, y1 - 18)), "SELECTED", fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    target = call.get("target_point_2d") or selected.get("target_point_2d")
    if isinstance(target, Sequence) and len(target) >= 2:
        x, y = float(target[0]), float(target[1])
        draw.line((x - 10, y, x + 10, y), fill=(255, 230, 0, 255), width=4)
        draw.line((x, y - 10, x, y + 10), fill=(255, 230, 0, 255), width=4)
    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    output_dir = route_dir / "event_visualizations"
    output_dir.mkdir(exist_ok=True)
    safe_event = "".join(char if char.isalnum() or char in "-_" else "_" for char in event_id)
    output_path = output_dir / f"{safe_event}_r{run_index}_a{attempt_index}_{call_kind}_localization.jpg"
    rendered.save(output_path, quality=94, optimize=True)
    results = [output_path.relative_to(route_dir)]
    focus_box: Tuple[float, float, float, float] | None = None
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        focus_box = tuple(float(value) for value in bbox)
    elif mask_bbox is not None:
        focus_box = tuple(float(value) for value in mask_bbox)
    if focus_box is not None:
        x1, y1, x2, y2 = focus_box
        box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        margin = max(48.0, 1.25 * max(box_width, box_height))
        crop = (
            max(0, int(math.floor(x1 - margin))),
            max(0, int(math.floor(y1 - margin))),
            min(rendered.width, int(math.ceil(x2 + margin))),
            min(rendered.height, int(math.ceil(y2 + margin))),
        )
        if crop[2] > crop[0] and crop[3] > crop[1]:
            zoom = rendered.crop(crop)
            scale = min(5.0, 560.0 / max(zoom.size))
            if scale > 1.0:
                zoom = zoom.resize(
                    (max(1, int(zoom.width * scale)), max(1, int(zoom.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            zoom_path = output_dir / f"{safe_event}_r{run_index}_a{attempt_index}_{call_kind}_localization_zoom.jpg"
            zoom.save(zoom_path, quality=95, optimize=True)
            results.append(zoom_path.relative_to(route_dir))
    return results


def _polyline_svg(
    gt: Sequence[Tuple[float, float]],
    pred: Sequence[Tuple[float, float]],
    mode: str,
) -> str:
    shown = gt if mode == "gt" else pred if mode == "pred" else list(gt) + list(pred)
    if not shown:
        return '<div class="empty">无轨迹坐标</div>'
    lats = [p[0] for p in shown]
    lngs = [p[1] for p in shown]
    lat0 = sum(lats) / len(lats)
    # Longitude degrees shrink by cos(latitude); compensate for a natural aspect ratio.
    lng_scale = max(0.1, math.cos(math.radians(lat0)))
    xs = [p[1] * lng_scale for p in shown]
    ys = [p[0] for p in shown]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-7)
    span_y = max(max_y - min_y, 1e-7)
    pad, width, height = 32.0, 520.0, 340.0

    def project(point: Tuple[float, float]) -> Tuple[float, float]:
        x = ((point[1] * lng_scale - min_x) / span_x) * (width - 2 * pad) + pad
        y = height - (((point[0] - min_y) / span_y) * (height - 2 * pad) + pad)
        return x, y

    def draw(points: Sequence[Tuple[float, float]], color: str, label: str) -> str:
        if not points:
            return ""
        projected = [project(p) for p in points]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in projected)
        pieces = [
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        ]
        for index, (x, y) in enumerate(projected):
            radius = 6 if index in (0, len(projected) - 1) else 3.5
            pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        sx, sy = projected[0]
        ex, ey = projected[-1]
        pieces.append(f'<text x="{sx + 8:.1f}" y="{sy - 8:.1f}" class="svg-label">{label} 起点</text>')
        pieces.append(f'<text x="{ex + 8:.1f}" y="{ey - 8:.1f}" class="svg-label">{label} 终点</text>')
        return "".join(pieces)

    body = []
    if mode in ("gt", "overlay"):
        body.append(draw(gt, "#16a34a", "GT"))
    if mode in ("pred", "overlay"):
        body.append(draw(pred, "#e11d48", "预测"))
    return (
        f'<svg class="trajectory" viewBox="0 0 {width:.0f} {height:.0f}" role="img">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        '<path d="M32 308H488 M32 32V308" stroke="#cbd5e1" stroke-width="1"/>'
        + "".join(body)
        + "</svg>"
    )


def _iter_attempts(prediction: Mapping[str, Any]) -> Iterable[Tuple[str, int, int, Mapping[str, Any]]]:
    emitted = False
    for run in prediction.get("search_runs", []):
        window_id = str(run.get("window_id", ""))
        run_index = int(run.get("run_index", 0) or 0)
        for evidence in run.get("event_evidence", []):
            for index, attempt in enumerate(evidence.get("attempts", []), start=1):
                if isinstance(attempt, Mapping):
                    emitted = True
                    yield window_id, run_index, index, attempt
    if emitted:
        return
    for ordinal, step in enumerate(prediction.get("search_steps", []), start=1):
        step_index = int(step.get("event_index", step.get("step_index", ordinal)) or ordinal)
        event_id = str(step.get("event_id") or f"event-{step_index}")
        label = f"event-{step_index} ({event_id})"
        for index, attempt in enumerate(step.get("visual_searches", []), start=1):
            if isinstance(attempt, Mapping):
                yield label, step_index, index, attempt


def _call_summary(call: Mapping[str, Any]) -> str:
    reason = html.escape(str(call.get("reason") or ""))
    bbox = html.escape(json.dumps(call.get("bbox_2d"), ensure_ascii=False))
    point = html.escape(json.dumps(call.get("target_point_2d"), ensure_ascii=False))
    geometry = call.get("navigation_geometry") or (call.get("selected_candidate") or {}).get("navigation_geometry") or {}
    return (
        '<div class="call-summary">'
        f'<span class="pill ok">{html.escape(str(call.get("provider") or "unknown"))}</span>'
        f'<span>present=<b>{_fmt(call.get("dest_present"))}</b></span>'
        f'<span>confidence=<b>{_fmt(call.get("confidence"))}</b></span>'
        f'<span>candidate=<code>{_fmt(call.get("selected_candidate_id"))}</code></span>'
        f'<span>bbox=<code>{bbox}</code></span><span>point=<code>{point}</code></span>'
        f'<p>{reason}</p>'
        + _navigation_geometry_table([{
            "candidate_id": call.get("selected_candidate_id"),
            "view_id": (call.get("selected_candidate") or {}).get("view_id"),
            "score": (call.get("selected_candidate") or {}).get("score"),
            "navigation_summary": geometry,
        }], title="最终候选导航几何")
        + "</div>"
    )


def _navigation_geometry_table(
    candidates: Sequence[Mapping[str, Any]],
    *,
    title: str = "候选导航几何",
) -> str:
    rows: List[str] = []
    for candidate in candidates:
        geometry = candidate.get("navigation_summary") or candidate.get("navigation_geometry") or {}
        if not geometry.get("available"):
            continue
        path = geometry.get("forward_path") or {}
        rows.append(
            "<tr>"
            f'<td><code>{_fmt(candidate.get("candidate_id"))}</code></td>'
            f'<td>{_fmt(candidate.get("view_id"))}</td>'
            f'<td>{_fmt(candidate.get("score"),4)}</td>'
            f'<td>{_fmt(geometry.get("distance_m"),2)}</td>'
            f'<td>{_fmt(geometry.get("relative_bearing_deg"),1)}</td>'
            f'<td>{_fmt(geometry.get("forward_offset_m"),2)}</td>'
            f'<td>{_fmt(geometry.get("right_offset_m"),2)}</td>'
            f'<td>{_fmt(geometry.get("direction_label"))}</td>'
            f'<td>{_fmt(path.get("intersects"))}</td>'
            f'<td>{_fmt(path.get("entry_distance_m"),2)} → {_fmt(path.get("exit_distance_m"),2)}</td>'
            f'<td>{_fmt(geometry.get("mask_intersects_agent_footprint"))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        f'<details open class="geometry-table"><summary>{html.escape(title)}</summary>'
        '<table><thead><tr><th>候选</th><th>视图</th><th>SAM3</th><th>距离m</th>'
        '<th>相对角°</th><th>前向m</th><th>右向m</th><th>方向</th>'
        '<th>通道相交</th><th>进入→离开m</th><th>机体重合</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></details>'
    )


def _observable_reasoning(trace: Sequence[Mapping[str, Any]]) -> str:
    """Render Qwen's persisted tool decisions without implying hidden reasoning."""
    if not trace:
        return '<p class="muted">该次调用没有保存工具决策记录。</p>'
    blocks: List[str] = []
    for ordinal, entry in enumerate(trace, start=1):
        tool = str(entry.get("tool") or "unknown")
        arguments = entry.get("arguments") or {}
        result = entry.get("result") or {}
        if tool == "find_visual_candidates":
            concepts = arguments.get("concepts") or []
            details = "".join(
                '<li><b>' + html.escape(str(item.get("role") or "concept")) + '</b> · '
                + html.escape(str(item.get("text") or "")) + '</li>'
                for item in concepts if isinstance(item, Mapping)
            )
            body = (
                '<p>QWEN 将当前目标拆分为以下 SAM3 查询概念：</p>'
                f'<ul class="concept-list">{details}</ul>'
                f'<p>scale-5 返回候选数：<b>{_fmt(result.get("candidate_count"))}</b> · '
                f'错误：<code>{_fmt(result.get("error") or "无")}</code></p>'
                '<details><summary>±15°前向范围角像素证据</summary>'
                f'<pre>{_json(result.get("candidates") or [])}</pre></details>'
            )
        elif tool == "analyze_navigation_geometry":
            analyses = result.get("analyses") or {}
            candidates = [
                {
                    "candidate_id": candidate_id,
                    "navigation_summary": value,
                }
                for candidate_id, value in analyses.items()
                if isinstance(value, Mapping)
            ]
            body = (
                '<p>QWEN 请求代码计算以下候选的详细地理关系：'
                f'<code>{html.escape(json.dumps(arguments.get("candidate_ids") or [], ensure_ascii=False))}</code></p>'
                + _navigation_geometry_table(candidates, title="详细导航几何分析")
                + '<details><summary>跨尺度最近候选与两两地理关系</summary>'
                f'<pre>{_json({"cross_scale_neighbors": result.get("cross_scale_neighbors") or {}, "pairwise_relations": result.get("pairwise_relations") or []})}</pre></details>'
            )
        elif tool == "finish_target_selection":
            body = (
                f'<p>QWEN 候选选择：present=<b>{_fmt(arguments.get("dest_present"))}</b> · '
                f'candidate=<code>{_fmt(arguments.get("candidate_id"))}</code> · '
                f'confidence=<b>{_fmt(arguments.get("confidence"))}</b></p>'
                f'<p class="decision-reason">{html.escape(str(arguments.get("reason") or "未记录理由"))}</p>'
                f'<p>工具校验结果：<code>{html.escape(json.dumps(result, ensure_ascii=False))}</code></p>'
            )
        elif tool == "finish_navigation_point":
            body = (
                f'<p>QWEN 独立航点决策：target_point='
                f'<code>{html.escape(json.dumps(arguments.get("target_point_2d"), ensure_ascii=False))}</code> · '
                f'confidence=<b>{_fmt(arguments.get("confidence"))}</b></p>'
                f'<p class="decision-reason">{html.escape(str(arguments.get("reason") or "未记录理由"))}</p>'
                f'<p>工具校验结果：<code>{html.escape(json.dumps(result, ensure_ascii=False))}</code></p>'
            )
        else:
            body = f'<pre class="inline-json">{_json(entry)}</pre>'
        blocks.append(
            '<article class="reasoning-step">'
            f'<h5>决策 {ordinal} · {html.escape(str(entry.get("phase") or "tool"))} · {html.escape(tool)}</h5>{body}</article>'
        )
    return '<div class="reasoning-flow">' + "".join(blocks) + '</div>'


def _model_calls(prediction: Mapping[str, Any]) -> Tuple[str, List[Mapping[str, Any]]]:
    blocks: List[str] = []
    traces: List[Mapping[str, Any]] = []
    for window_id, run_index, attempt_index, attempt in _iter_attempts(prediction):
        parts: List[str] = []
        for name in ("qwen",):
            call = attempt.get(name)
            if not isinstance(call, Mapping):
                continue
            trace = call.get("tool_trace") or []
            traces.extend(t for t in trace if isinstance(t, Mapping))
            parts.append(
                f'<h4>{name.upper()}</h4>{_call_summary(call)}'
                f'<details open><summary>{name} 完整输出</summary><pre>{_json(call)}</pre></details>'
            )
        if parts:
            blocks.append(
                f'<details open class="attempt"><summary>{html.escape(window_id)} · run {run_index} · '
                f'search {attempt_index}</summary>{"".join(parts)}</details>'
            )
    return "".join(blocks) or '<p class="muted">没有模型调用记录。</p>', traces


def _tool_calls(traces: Sequence[Mapping[str, Any]]) -> str:
    if not traces:
        return '<p class="muted">没有记录到工具调用。具体失败原因见对应 Search step。</p>'
    rows = []
    for index, trace in enumerate(traces, start=1):
        tool = html.escape(str(trace.get("tool") or trace.get("name") or "unknown"))
        rows.append(
            '<details open class="tool-call">'
            f'<summary>#{index} · {tool}</summary><pre>{_json(trace)}</pre></details>'
        )
    return "".join(rows)


def _trace_status(call: Mapping[str, Any]) -> str:
    trace = call.get("tool_trace") or []
    if trace:
        return f'<span class="pill ok">工具调用 {len(trace)} 次</span>'
    reason = str(call.get("reason") or "").lower()
    if "timed out" in reason or "timeout" in reason:
        label = "QWEN 请求超时，未产生工具调用"
    elif "did not select" in reason:
        label = "QWEN 未选择视觉定位工具"
    elif reason:
        label = "视觉定位失败，未产生工具调用"
    else:
        label = "未产生工具调用"
    return f'<span class="pill bad">{label}</span>'


def _figure(path: Path, caption: str, css_class: str = "") -> str:
    src = html.escape(path.as_posix(), quote=True)
    return (
        f'<figure class="figure {css_class}"><a href="{src}"><img loading="lazy" '
        f'src="{src}" alt="{html.escape(caption, quote=True)}"></a>'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
    )


def _event_lookup(prediction: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for event in prediction.get("search_steps", []):
        if event.get("event_id"):
            lookup[str(event["event_id"])] = event
    for window in prediction.get("execution_windows", []):
        for event in window.get("events", []):
            if event.get("event_id"):
                lookup[str(event["event_id"])] = event
    return lookup


def _parse_result(parse_record: Mapping[str, Any] | None) -> str:
    if not parse_record:
        return '<p class="muted">未找到该轨迹对应的 InstructionPlan。</p>'
    plan = parse_record.get("plan") or {}
    entities = plan.get("entities") or []
    events = plan.get("events") or []
    entity_lookup = {
        str(entity.get("id")): entity
        for entity in entities
        if entity.get("id") is not None
    }
    turn_rows = "".join(
        '<tr class="parse-turn-row">'
        f'<td>{_fmt(turn.get("turn"))}</td><td>{_fmt(turn.get("role"))}</td>'
        f'<td>{html.escape(str(turn.get("text") or ""))}</td></tr>'
        for turn in parse_record.get("turns", [])
    )
    entity_rows = "".join(
        '<tr class="parse-entity-row">'
        f'<td><code>{_fmt(entity.get("id"))}</code></td>'
        f'<td>{_fmt(entity.get("kind"))}</td>'
        f'<td>{html.escape(str(entity.get("description") or ""))}</td>'
        f'<td>{_fmt(entity.get("goal"))}</td>'
        f'<td><pre class="inline-json">{_json({key: value for key, value in entity.items() if key not in {"id", "kind", "description", "goal"}})}</pre></td>'
        '</tr>'
        for entity in entities
    )
    event_rows = []
    for index, event in enumerate(events, start=1):
        event_id = event.get("id") or "e{}".format(index)
        entity_id = str(event.get("entity") or "")
        entity = entity_lookup.get(entity_id, {})
        direction = event.get("direction") or {}
        remaining = {
            key: value
            for key, value in event.items()
            if key not in {"type", "turn", "entity", "direction"}
        }
        event_rows.append(
            '<tr class="parse-event-row">'
            f'<td><code>{_fmt(event_id)}</code></td><td>{_fmt(event.get("turn"))}</td>'
            f'<td><b>{_fmt(event.get("type"))}</b></td>'
            f'<td><code>{html.escape(entity_id or "—")}</code><br>'
            f'<span class="muted">{html.escape(str(entity.get("kind") or ""))} · '
            f'{html.escape(str(entity.get("description") or ""))}</span></td>'
            f'<td>{_fmt(direction.get("frame"))}<br><b>{_fmt(direction.get("angle"),1)}°</b></td>'
            f'<td><pre class="inline-json">{_json(remaining)}</pre></td>'
            '</tr>'
        )
    return (
        '<div class="parse-meta">'
        f'<span>parser_model=<code>{_fmt(parse_record.get("parser_model"))}</code></span>'
        f'<span>format_version=<b>{_fmt(parse_record.get("format_version"))}</b></span>'
        f'<span>starting_heading=<b>{_fmt(parse_record.get("starting_heading_deg"),1)}°</b></span>'
        f'<span>repaired=<b>{_fmt(parse_record.get("repaired"))}</b></span>'
        '</div>'
        '<h3>Parse 输入对话</h3><table><thead><tr><th>Turn</th><th>Role</th><th>原文</th></tr></thead>'
        f'<tbody>{turn_rows}</tbody></table>'
        '<h3>实体解析</h3><table><thead><tr><th>ID</th><th>类型</th><th>描述</th><th>Goal</th><th>其他属性</th></tr></thead>'
        f'<tbody>{entity_rows}</tbody></table>'
        '<h3>事件序列</h3><p class="muted">e1、e2…与后续事件级视觉定位卡片使用相同顺序。</p>'
        '<table><thead><tr><th>Event</th><th>Turn</th><th>类型</th><th>实体</th><th>方向</th><th>约束</th></tr></thead>'
        f'<tbody>{"".join(event_rows)}</tbody></table>'
        f'<details><summary>完整 Parse JSON</summary><pre>{_json(parse_record)}</pre></details>'
    )


def _event_visualizations(prediction: Mapping[str, Any], route_dir: Path) -> str:
    events = _event_lookup(prediction)
    sections: List[str] = []
    runs = list(prediction.get("search_runs", []))
    if not runs:
        for ordinal, step in enumerate(prediction.get("search_steps", []), start=1):
            attempts = list(step.get("visual_searches", []))
            selected_call = next(
                (
                    attempt.get("qwen")
                    for attempt in reversed(attempts)
                    if isinstance(attempt, Mapping)
                    and isinstance(attempt.get("qwen"), Mapping)
                    and attempt["qwen"].get("dest_present")
                ),
                {},
            )
            event_id = str(step.get("event_id") or f"event-{ordinal}")
            event_type = str(step.get("type") or "SEARCH")
            step_index = int(step.get("event_index", step.get("step_index", ordinal)) or ordinal)
            runs.append({
                "window_id": f"event-{step_index}",
                "run_index": step_index,
                "event_evidence": [{
                    "event_id": event_id,
                    "event_type": event_type,
                    "entity_description": step.get("target_description"),
                    "source_span": step.get("source_text"),
                    "attempts": attempts,
                    "found": step.get("status") == "located",
                    "candidate_id": selected_call.get("selected_candidate_id"),
                    "destination_position": step.get("position_after"),
                    "bbox_2d": selected_call.get("bbox_2d"),
                    "confidence": selected_call.get("confidence"),
                    "search_log_path": step.get("search_log_path"),
                    "status": step.get("status"),
                    "position_before": step.get("position_before"),
                    "position_after": step.get("position_after"),
                    "heading_before_deg": step.get("heading_before_deg"),
                    "heading_after_deg": step.get("heading_after_deg"),
                }],
            })
    for run in runs:
        window_id = str(run.get("window_id") or "")
        run_index = int(run.get("run_index", 0) or 0)
        for evidence in run.get("event_evidence", []):
            event_id = str(evidence.get("event_id") or "unknown")
            event = events.get(event_id, {})
            description = str(
                event.get("entity_description")
                or evidence.get("entity_description")
                or event.get("source_span")
                or evidence.get("source_span")
                or "未记录事件描述"
            )
            event_type = str(evidence.get("event_type") or event.get("type") or "UNKNOWN")
            attempt_blocks: List[str] = []
            for attempt_index, attempt in enumerate(evidence.get("attempts", []), start=1):
                for call_kind in ("qwen",):
                    call = attempt.get(call_kind)
                    if not isinstance(call, Mapping):
                        continue
                    qwen_visuals: List[Tuple[Path, str, str]] = []
                    sam_visuals: List[Tuple[Path, str, str]] = []
                    attempt_views = attempt.get("view_paths") or {}
                    raw_labels = {
                        "main": "尺度 5 main view（QWEN / SAM3 输入与最终坐标来源）",
                    }
                    for view_id in ("main",):
                        raw = _relative_artifact(route_dir, attempt_views.get(view_id))
                        if raw is not None:
                            qwen_visuals.append((raw, raw_labels[view_id], "source-image"))
                    overlays = _render_localization_overlay(
                        route_dir,
                        event_id,
                        run_index,
                        attempt_index,
                        call_kind,
                        call,
                        attempt,
                    )
                    if overlays:
                        overlay_label = (
                            "整条轨迹最终目标 bbox / target point"
                            if call.get("is_final_goal") and call.get("final_bbox_2d")
                            else "QWEN 事件候选与关系感知 target point"
                        )
                        qwen_visuals.append((overlays[0], overlay_label, "selected-image"))
                        if len(overlays) > 1:
                            qwen_visuals.append((overlays[1], "QWEN 定位局部放大", "selected-image"))
                    detector = _relative_artifact(route_dir, attempt_views.get("detection"))
                    if detector is not None:
                        qwen_visuals.append((detector, "搜索引擎定位框输出", "detector-image"))
                    for label, value in (call.get("view_paths") or {}).items():
                        artifact = _relative_artifact(route_dir, value)
                        if artifact is not None:
                            if "selected_target_point_input" in label.lower():
                                qwen_visuals.append((artifact, "独立航点调用的固定目标 mask/bbox 图", "selected-image"))
                                continue
                            if "selected_target_bbox_input" in label.lower():
                                qwen_visuals.append((artifact, "独立 bbox 调用的选中目标专用图", "selected-image"))
                                continue
                            if "geometry" in label.lower():
                                category = "导航几何叠加图"
                            elif "dino" in label.lower():
                                category = "DINOv3 跨视图匹配"
                            else:
                                category = "SAM3 候选 contact sheet"
                            sam_visuals.append((artifact, f"{category} · {label}", "tool-image"))
                    mask = _relative_artifact(
                        route_dir,
                        call.get("mask_path")
                        or (call.get("selected_candidate") or {}).get("mask_path"),
                    )
                    if mask is not None:
                        sam_visuals.append((mask, "SAM3 选中候选 mask", "mask-image"))
                    deduped_qwen: List[Tuple[Path, str, str]] = []
                    deduped_sam: List[Tuple[Path, str, str]] = []
                    seen = set()
                    for item in qwen_visuals:
                        identity = (item[0], item[1])
                        if identity not in seen:
                            seen.add(identity)
                            deduped_qwen.append(item)
                    seen = set()
                    for item in sam_visuals:
                        if item[0] not in seen:
                            seen.add(item[0])
                            deduped_sam.append(item)
                    trace = call.get("tool_trace") or []
                    trace_status = _trace_status(call)
                    attempt_blocks.append(
                        '<div class="event-attempt">'
                        f'<h4>Search {attempt_index} · 三尺度并行定位 · {call_kind.upper()} {trace_status}</h4>'
                        + _call_summary(call)
                        + '<h5>QWEN 输入与定位结果</h5><div class="event-images">'
                        + "".join(_figure(*item) for item in deduped_qwen)
                        + "</div>"
                        + '<h5>SAM3 候选与分割结果</h5><div class="event-images">'
                        + "".join(_figure(*item) for item in deduped_sam)
                        + "</div>"
                        + '<h5>QWEN 可观察分析与工具决策过程</h5>'
                        + _observable_reasoning(trace)
                        + (
                            f'<details><summary>展开原始工具调用 JSON</summary><pre>{_json(trace)}</pre></details>'
                            if trace else ""
                        )
                        + f'<details><summary>展开 QWEN 完整可见输出</summary><pre>{_json(call)}</pre></details>'
                        + "</div>"
                    )
            evidence_summary = {
                key: evidence.get(key)
                for key in (
                    "found", "candidate_id", "candidate_index", "destination_position",
                    "bbox_2d", "confidence", "search_log_path", "status",
                    "position_before", "position_after", "heading_before_deg",
                    "heading_after_deg",
                )
            }
            execution_note = {
                "TURN": "蓝色箭头已绘制在预测轨迹地图上，表示转向后的绝对朝向。",
                "MOVE": "该事件仅按当前朝向短距离移动，不调用视觉定位。",
                "REACH": "定位目标后按 relation 指定的目标关系完成移动；通过类 relation 会越过目标并保留安全距离。",
                "AVOID": "定位目标后停在目标前方的安全距离。",
            }.get(event_type, "")
            sections.append(
                '<article class="event-card">'
                f'<h3>{html.escape(event_id)} · {html.escape(event_type)} · {html.escape(description)}</h3>'
                f'<p class="muted">顺序步骤 {run_index} · source: '
                f'{html.escape(str(event.get("source_text") or event.get("source_span") or evidence.get("source_span") or "—"))}</p>'
                '<div class="event-meta">'
                f'<span>status=<b>{_fmt(evidence.get("status"))}</b></span>'
                f'<span>found=<b>{_fmt(evidence.get("found"))}</b></span>'
                f'<span>candidate=<code>{_fmt(evidence.get("candidate_id"))}</code></span>'
                f'<span>confidence=<b>{_fmt(evidence.get("confidence"))}</b></span>'
                f'<span>heading=<b>{_fmt(evidence.get("heading_before_deg"),1)}° → {_fmt(evidence.get("heading_after_deg"),1)}°</b></span>'
                f'<span>position=<code>{html.escape(json.dumps(evidence.get("position_before"), ensure_ascii=False))}</code> → '
                f'<code>{html.escape(json.dumps(evidence.get("position_after"), ensure_ascii=False))}</code></span>'
                f'<span>destination=<code>{html.escape(json.dumps(evidence.get("destination_position"), ensure_ascii=False))}</code></span>'
                "</div>"
                + (f'<p class="execution-note">{html.escape(execution_note)}</p>' if execution_note else "")
                + "".join(attempt_blocks)
                + f'<details><summary>事件证据摘要 JSON</summary><pre>{_json(evidence_summary)}</pre></details>'
                + "</article>"
            )
    return "".join(sections) or '<p class="muted">没有事件视觉证据。</p>'


def _referenced_images(value: Any, route_dir: Path) -> List[Path]:
    """Return only image artifacts referenced by this prediction.

    Route directories can be reused across runs.  Walking the whole directory
    would therefore mix old DINO/SAM artifacts into a new report.
    """
    found: List[Path] = []
    seen = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str) and Path(item).suffix.lower() in IMAGE_SUFFIXES:
            artifact = _relative_artifact(route_dir, item)
            if artifact is not None and artifact not in seen:
                seen.add(artifact)
                found.append(artifact)

    visit(value)
    return found


def _image_groups(
    route_dir: Path,
    prediction: Mapping[str, Any],
) -> Tuple[List[Path], Dict[str, List[Path]]]:
    model: List[Path] = []
    tools: Dict[str, List[Path]] = {}
    for rel in sorted(_referenced_images(prediction, route_dir)):
        if rel.parts[0] == "event_visualizations" or rel.name.startswith("trajectory_"):
            continue
        grounding_part = next((part for part in rel.parts if "grounding" in part.lower()), None)
        if grounding_part:
            tools.setdefault(grounding_part, []).append(rel)
        else:
            model.append(rel)
    return model, tools


def _gallery(paths: Sequence[Path], css_class: str = "") -> str:
    if not paths:
        return '<p class="muted">无图片产物。</p>'
    figures = []
    for path in paths:
        src = html.escape(path.as_posix(), quote=True)
        caption = html.escape(path.as_posix())
        figures.append(
            f'<figure class="figure {css_class}"><a href="{src}"><img loading="lazy" '
            f'src="{src}" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>'
        )
    return '<div class="images">' + "".join(figures) + "</div>"


def _tool_galleries(groups: Mapping[str, Sequence[Path]]) -> str:
    if not groups:
        return '<p class="muted">无视觉工具图片产物。</p>'
    pieces = []
    for group, paths in groups.items():
        sheets = [p for p in paths if "candidates" in p.name or "dinov3" in p.name]
        masks = [p for p in paths if p not in sheets]
        pieces.append(
            f'<details class="image-group"><summary>{html.escape(group)} · {len(paths)} 张</summary>'
            '<h4>候选图 / DINOv3 匹配图</h4>' + _gallery(sheets, "tool-image")
            + '<h4>Mask</h4>' + _gallery(masks, "mask-image") + "</details>"
        )
    return "".join(pieces)


CSS = """
*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#f1f5f9;color:#172033;font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;line-height:1.5}
main{max-width:1600px;margin:auto;padding:24px 30px 64px}h1{margin:0}h2{margin-top:34px;border-bottom:2px solid #d8e2ef;padding-bottom:7px}h3,h4{margin-bottom:8px}
a{color:#075985}.muted{color:#64748b}.nav{display:flex;gap:16px;margin-bottom:14px}.cards,.trajectory-grid,.map-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.card,.panel,details{background:white;border:1px solid #d8e2ef;border-radius:8px;box-shadow:0 1px 2px #0f172a0d}.card{padding:13px 15px}.label{font-size:12px;color:#64748b}.value{font-size:20px;font-weight:700}
.panel{padding:10px}.panel h3{margin:2px 4px 8px}.trajectory{display:block;width:100%;height:340px;border-radius:7px}.svg-label{font-size:11px;fill:#334155;font-weight:600}
.map-panel{background:white;border:1px solid #d8e2ef;border-radius:8px;padding:10px}.map-panel h3{margin:2px 4px 8px}.map-panel img{display:block;width:100%;height:430px;object-fit:contain;background:#0f172a;border-radius:7px}
.instructions{display:grid;gap:10px}.instruction-card{display:grid;grid-template-columns:90px 1fr;gap:5px 14px;background:white;border:1px solid #d8e2ef;border-radius:8px;padding:13px 15px}.instruction-index{grid-row:1/3;color:#0369a1;font-weight:750;font-size:16px}.instruction-text{font-size:16px;white-space:pre-wrap}.instruction-meta{font-size:12px;color:#64748b}
.parse-meta{display:flex;gap:18px;flex-wrap:wrap;background:white;border:1px solid #d8e2ef;border-radius:8px;padding:12px 15px}.inline-json{margin:0;max-height:130px;min-width:110px;padding:7px;font-size:11px}
.event-card{background:white;border:1px solid #cbd5e1;border-left:5px solid #0284c7;border-radius:8px;padding:14px 16px;margin:16px 0;box-shadow:0 2px 5px #0f172a12}.event-card>h3{margin-top:0}.event-meta{display:flex;gap:16px;flex-wrap:wrap;background:#f8fafc;border-radius:7px;padding:9px 11px}.event-attempt{border-top:1px solid #d8e2ef;margin-top:15px;padding-top:8px}.event-images{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.event-images .figure img{height:300px}.event-images .selected-image{border:3px solid #f43f5e}.event-images .tool-image{border:2px solid #06b6d4}
.execution-note{border-left:4px solid #0ea5e9;background:#f0f9ff;padding:8px 11px;color:#0c4a6e}.reasoning-flow{display:grid;gap:8px}.reasoning-step{border-left:4px solid #14b8a6;background:#f8fafc;padding:8px 12px}.reasoning-step h5{margin:0 0 6px}.reasoning-step p{margin:5px 0}.concept-list{margin:6px 0;padding-left:22px}.decision-reason{color:#164e63;font-weight:600}
details{padding:9px 12px;margin:8px 0}summary{cursor:pointer;color:#075985;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;border-radius:7px;padding:12px;max-height:650px;overflow:auto;font:12px/1.45 Consolas,monospace}
code{background:#e2e8f0;padding:1px 4px;border-radius:4px;overflow-wrap:anywhere;word-break:break-word}.call-summary{padding:8px 3px}.call-summary span{margin-right:12px}.pill{display:inline-block;border-radius:12px;padding:2px 8px;font-weight:650}.pill.ok{background:#dcfce7;color:#166534}.pill.bad{background:#fee2e2;color:#991b1b}
.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}.figure{margin:0;background:white;border:1px solid #d8e2ef;border-radius:8px;padding:7px}.figure img{width:100%;height:220px;object-fit:contain;background:#e2e8f0}.figure.mask-image img{image-rendering:auto}.figure figcaption{font-size:11px;color:#475569;overflow-wrap:anywhere;margin-top:5px}.image-group{margin:12px 0}.empty{height:340px;display:grid;place-items:center;color:#64748b}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:8px;border:1px solid #d8e2ef;text-align:left}th{background:#eaf1f8}@media(max-width:700px){main{padding:16px}.images{grid-template-columns:1fr}table{display:block;max-width:100%;overflow-x:auto}}
"""


def _report_html(
    prediction: Mapping[str, Any],
    route_dir: Path,
    map_images: Mapping[str, Path],
    parse_record: Mapping[str, Any] | None,
    dataset_name: str,
) -> str:
    route_id = str(prediction.get("instr_id") or route_dir.name)
    gt = _gt_trajectory(prediction)
    pred = _pred_trajectory(prediction)
    model_calls, traces = _model_calls(prediction)
    model_images, tool_images = _image_groups(route_dir, prediction)
    providers = Counter()
    call_count = 0
    for _, _, _, attempt in _iter_attempts(prediction):
        for name in ("qwen",):
            call = attempt.get(name)
            if isinstance(call, Mapping):
                call_count += 1
                providers[str(call.get("provider") or "unknown")] += 1
    provider_text = ", ".join(f"{k}: {v}" for k, v in providers.items()) or "—"
    full_outputs = {
        key: prediction.get(key)
        for key in (
            "search_steps", "event_progress", "completed_event_ids",
            "physical_trajectory", "reasoning", "grounding", "final_target_grounding",
        )
    }
    final_target = prediction.get("final_target_grounding") or {}
    final_target_html = (
        '<div class="panel"><h3>整条轨迹最终目标定位</h3>'
        f'<div class="event-meta"><span>event=<code>{html.escape(str(final_target.get("event_id") or "—"))}</code></span>'
        f'<span>candidate=<code>{html.escape(str(final_target.get("candidate_id") or "—"))}</code></span>'
        f'<span>bbox_2d=<code>{html.escape(json.dumps(final_target.get("bbox_2d"), ensure_ascii=False))}</code></span>'
        f'<span>raw_bbox_2d=<code>{html.escape(json.dumps(final_target.get("raw_bbox_2d"), ensure_ascii=False))}</code></span></div>'
        f'<p>{html.escape(str(final_target.get("reason") or ""))}</p>'
        f'<details><summary>最终目标完整结构</summary><pre>{_json(final_target)}</pre></details></div>'
        if final_target else '<p class="muted">本次运行未产生整条轨迹最终目标 bbox。</p>'
    )
    grounding = prediction.get("grounding") or {}
    execution_mode = str(prediction.get("execution_mode") or "sequential-events")
    uses_dino = any("dinov3" in name.lower() for name in providers)
    grounding_stack = (
        f"Qwen3-VL + {grounding.get('backend') or 'unknown'}"
        + (" + DINOv3" if uses_dino else "")
    )
    map_panels = "".join(
        f'<div class="map-panel"><h3>{title}</h3><a href="{html.escape(map_images[key].as_posix(), quote=True)}">'
        f'<img src="{html.escape(map_images[key].as_posix(), quote=True)}" alt="{title}"></a></div>'
        for key, title in (
            ("gt", "GT 轨迹 · 原始地图"),
            ("pred", "预测轨迹 · 原始地图"),
            ("overlay", "GT / 预测叠加 · 原始地图"),
        )
        if key in map_images
    ) or '<p class="muted">未找到该路线的 GeoTIFF 或地理边界，无法生成地图投影。</p>'
    event_visuals = _event_visualizations(prediction, route_dir)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(route_id)} · 顺序事件轨迹总结</title><style>{CSS}</style></head><body><main>
<nav class="nav"><a href="../reports_index.html">返回总索引</a><a href="prediction.json">prediction.json</a></nav>
<h1>轨迹 {html.escape(route_id)}</h1>
<p class="muted">{html.escape(dataset_name)} · {execution_mode} · {html.escape(grounding_stack)}</p>
<section class="cards">
<div class="card"><div class="label">最终 IoU</div><div class="value">{_fmt(prediction.get('final_iou'),4)}</div></div>
<div class="card"><div class="label">运行状态</div><div class="value">{_fmt(prediction.get('termination_reason'))}</div></div>
<div class="card"><div class="label">预测 / GT 点数</div><div class="value">{len(pred)} / {len(gt)}</div></div>
<div class="card"><div class="label">模型调用</div><div class="value">{call_count}</div></div>
<div class="card"><div class="label">工具调用 trace</div><div class="value">{len(traces)}</div></div>
<div class="card"><div class="label">Provider 分布</div><div class="value" style="font-size:14px">{html.escape(provider_text)}</div></div>
</section>
<h2>完整对话与 Parse 事件序列</h2><p class="muted">按原始 turn 展示完整对话，并列出实体抽取、严格执行顺序及方向/空间约束。</p>{_parse_result(parse_record)}
<h2>地图中的 GT 与预测轨迹</h2><div class="map-grid">{map_panels}</div>
<p class="muted">三幅图使用相同的 GeoTIFF 底图、经纬度投影和裁剪范围。绿色 G 点/线为 GT，红色 P 点/线为预测；半透明四边形是各步 agent footprint，金色框为数据集 GT destination bbox，青色框为整条轨迹最终预测目标 bbox，蓝色箭头表示 TURN 后的绝对朝向。</p>
<h2>整条轨迹最终目标</h2>{final_target_html}
<h2>事件级视觉定位证据</h2><p class="muted">严格按 event → Search step 排列：单张 scale-5 输入、SAM3候选、QWEN关系感知目标点，以及仅由 goal=true 事件的独立模型调用产生的最终紧致 bbox。</p>{event_visuals}
<h2>全部模型中间输出</h2><p class="muted">展示模型可见的决策理由、候选选择和按 step / attempt 保存的完整结构化输出；不包含模型不可见的内部思维链。</p><details open><summary>全部模型调用</summary>{model_calls}</details>
<h2>全部工具调用 trace</h2><p class="muted">按实际调用顺序保留函数名、文本概念、参数、候选/匹配数量、错误与最终提交理由。</p><details open><summary>全部工具调用</summary>{_tool_calls(traces)}</details>
<h2>其他导航中间图</h2><details><summary>展开未归入事件主证据链的视图</summary>{_gallery(model_images)}</details>
<h2>视觉工具产物全集</h2><details><summary>展开全部 SAM3 / DINOv3 文件</summary>{_tool_galleries(tool_images)}</details>
<h2>完整结构化中间输出</h2><details><summary>展开全部顺序事件、Search 与轨迹状态数据</summary><pre>{_json(full_outputs)}</pre></details>
</main></body></html>"""


def generate_reports(
    root: Path,
    dataset_dir: Path,
    plans_path: Path,
    limit: int | None = None,
    route_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    dataset_name = dataset_dir.name or "dataset"
    metadata = _route_metadata(dataset_dir)
    parse_plans = _parse_records(plans_path)
    prediction_paths = sorted(
        root.glob("*/prediction.json"), key=lambda p: p.stat().st_mtime
    )
    if route_ids:
        selected = set(route_ids)
        prediction_paths = [
            path for path in prediction_paths if path.parent.name in selected
        ]
    if limit is not None:
        prediction_paths = prediction_paths[:limit]
    reports: List[Dict[str, Any]] = []
    for prediction_path in prediction_paths:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        route_dir = prediction_path.parent
        report_path = route_dir / "trajectory_summary.html"
        map_images = _trajectory_map_images(
            prediction,
            route_dir,
            dataset_dir,
            metadata,
        )
        report_path.write_text(
            _report_html(
                prediction,
                route_dir,
                map_images,
                parse_plans.get(str(prediction.get("instr_id") or route_dir.name)),
                dataset_name,
            ),
            encoding="utf-8",
        )
        reports.append({
            "route_id": prediction.get("instr_id") or route_dir.name,
            "path": report_path,
            "final_iou": prediction.get("final_iou"),
            "success": prediction.get("success"),
            "termination_reason": prediction.get("termination_reason"),
        })
    rows = "".join(
        "<tr>"
        f'<td><a href="{html.escape(item["path"].relative_to(root).as_posix(), quote=True)}">{html.escape(str(item["route_id"]))}</a></td>'
        f'<td>{_fmt(item["final_iou"],4)}</td><td>{_fmt(item["success"])}</td>'
        f'<td>{_fmt(item["termination_reason"])}</td></tr>'
        for item in reports
    )
    index = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(dataset_name)} 顺序事件轨迹总结</title><style>{CSS}</style></head><body><main><h1>{html.escape(dataset_name)} 顺序事件轨迹总结</h1>
<p class="muted">每条报告包含完整对话与 Parse 事件序列、GT/预测轨迹、TURN 朝向，以及带目标 relation 的 REACH/AVOID 事件的 QWEN 定位、SAM3 候选/分割和可观察决策过程。</p>
<table><thead><tr><th>轨迹</th><th>最终 IoU</th><th>成功</th><th>终止原因</th></tr></thead><tbody>{rows}</tbody></table>
</main></body></html>"""
    (root / "reports_index.html").write_text(index, encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Search output directory")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets" / "sample2",
        help="Dataset directory containing GeoTIFF maps and route metadata",
    )
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "out" / "preds_out_full_sample2" / "instruction_plans.jsonl"
        ),
        help="Parse output JSONL containing InstructionPlans",
    )
    parser.add_argument("--limit", type=int, default=None, help="Oldest completed routes to include")
    parser.add_argument(
        "--route-id",
        action="append",
        dest="route_ids",
        help="Only include this route ID; repeat for multiple routes",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reports = generate_reports(
        root,
        args.dataset_dir.resolve(),
        args.plans_path.resolve(),
        args.limit,
        args.route_ids,
    )
    print(f"generated {len(reports)} route reports under {root}")
    for item in reports:
        print(item["path"])


if __name__ == "__main__":
    main()
