"""Loopback-only JSON service for zero-shot visual grounding.

Run in the visual-model environment, for example::

    conda run -n sam3 python -m visual_grounding.service --backend sam3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .backends import CandidateBackend, create_backend
from .contracts import parse_concepts, parse_views, require_list, require_mapping
from .matcher import DinoV3Matcher
from .postprocess import draw_contact_sheet, filter_proposals, materialize_candidates


DEFAULT_SAM3_CHECKPOINT = os.getenv(
    "SAM3_CHECKPOINT", r"E:\Levir\Graduation_Project\sam3\sam3.pt"
)
DEFAULT_DINOV3_REPO = os.getenv(
    "DINOV3_REPO", r"C:\Users\郝梓翔\.cache\torch\hub\facebookresearch_dinov3_main"
)
DEFAULT_DINOV3_CHECKPOINT = os.getenv(
    "DINOV3_CHECKPOINT",
    r"C:\Users\郝梓翔\.cache\torch\hub\checkpoints\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
)


class VisionService:
    def __init__(self, args: argparse.Namespace):
        self.started_at = time.time()
        self.args = args
        self.backend: CandidateBackend = create_backend(
            args.backend,
            device=args.device,
            score_threshold=args.score_threshold,
            sam3_checkpoint=args.sam3_checkpoint,
        )
        self.matcher = DinoV3Matcher(
            args.dinov3_repo,
            args.dinov3_checkpoint,
            device=args.device,
            image_size=args.dino_image_size,
        )
        self.prewarmed = False
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            self.backend.load()
            self.prewarmed = True

    def health(self) -> Dict[str, Any]:
        cuda: Dict[str, Any] = {"available": False}
        try:
            import torch

            cuda = {
                "available": bool(torch.cuda.is_available()),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                "reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                "total_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
            }
        except Exception as exc:
            cuda["error"] = str(exc)
        return {
            "ok": bool(self.prewarmed),
            "backend": self.backend.name,
            "weights": dict(self.backend.weights),
            "dinov3": {
                "repo": self.matcher.repo,
                "checkpoint": self.matcher.checkpoint,
                "loaded": self.matcher.model is not None,
            },
            "device": self.args.device,
            "cuda": cuda,
            "prewarmed": self.prewarmed,
            "thresholds": {
                "candidate_score": self.args.score_threshold,
                "mask_nms_iou": self.args.nms_iou,
                "top_k": self.args.top_k,
            },
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }

    def _check_backend(self, payload: Dict[str, Any]) -> None:
        requested = str(payload.get("backend") or self.backend.name)
        if requested != self.backend.name:
            raise ValueError(
                f"backend mismatch: service={self.backend.name}, request={requested}"
            )

    def candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_backend(payload)
        views = parse_views(payload.get("views"))
        concepts = parse_concepts(payload.get("concepts"))
        artifact_dir = Path(str(payload.get("artifact_dir") or "vision_tool_output")).resolve()
        entity_kind = str(payload.get("entity_kind") or "LANDMARK")
        event_type = str(payload.get("event_type") or "")
        roi = payload.get("roi")
        if roi is not None:
            roi = [float(item) for item in require_list(roi, "roi", minimum=4, maximum=4)]
        requested_top_k = min(
            self.args.top_k,
            max(1, int(payload.get("max_candidates_per_view") or self.args.top_k)),
        )
        all_candidates: List[Dict[str, Any]] = []
        contact_sheets: List[str] = []
        candidate_counts_by_view = {view.view_id: 0 for view in views}
        started = time.monotonic()
        id_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(payload.get("request_id") or "")).strip("-")[:48]
        with self._lock:
            images = {view.view_id: Image.open(view.image_path).convert("RGB") for view in views}
            for view in views:
                ranked = []
                for concept in concepts:
                    proposals = self.backend.propose(images[view.view_id], concept.text)
                    proposals = filter_proposals(
                        proposals,
                        max_candidates=requested_top_k,
                        nms_iou=self.args.nms_iou,
                        roi=roi,
                    )
                    ranked.extend((proposal.score, concept, proposal) for proposal in proposals)
                ranked.sort(key=lambda item: item[0], reverse=True)
                selected = ranked[:requested_top_k]
                view_candidates = []
                view_masks = []
                for _, concept, proposal in selected:
                    candidates, masks = materialize_candidates(
                        [proposal],
                        view,
                        concept,
                        self.backend.name,
                        artifact_dir,
                        entity_kind,
                        event_type,
                        len(all_candidates) + 1,
                        id_prefix=id_prefix,
                    )
                    all_candidates.extend(item.to_dict() for item in candidates)
                    view_candidates.extend(candidates)
                    view_masks.extend(masks)
                candidate_counts_by_view[view.view_id] = len(view_candidates)
                if view_candidates:
                    path = artifact_dir / f"{view.view_id}_candidates.png"
                    contact_sheets.append(
                        draw_contact_sheet(
                            view.image_path,
                            view_candidates,
                            view_masks,
                            path,
                            f"{view.view_id} / scale {view.scale:g} / all concepts"
                            if view.scale is not None
                            else f"{view.view_id} / all concepts",
                        )
                    )
        return {
            "backend": self.backend.name,
            "candidates": all_candidates,
            "contact_sheet_paths": contact_sheets,
            "candidate_counts_by_view": candidate_counts_by_view,
            "max_candidates_per_view": requested_top_k,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "thresholds": self.health()["thresholds"],
        }

    def matches(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_backend(payload)
        candidates = require_list(payload.get("candidates"), "candidates", minimum=1, maximum=12)
        views = parse_views(payload.get("views"))
        by_id = {view.view_id: view for view in views}
        target_ids = [str(item) for item in require_list(payload.get("target_view_ids"), "target_view_ids", minimum=1, maximum=8)]
        artifact_dir = Path(str(payload.get("artifact_dir") or "vision_tool_output")).resolve()
        expected_points = require_mapping(payload.get("expected_points") or {}, "expected_points")
        results: List[Dict[str, Any]] = []
        started = time.monotonic()
        with self._lock:
            for raw in candidates:
                candidate = require_mapping(raw, "candidate")
                source_view_id = str(candidate.get("view_id") or "")
                if source_view_id not in by_id:
                    raise ValueError(f"candidate source view is absent: {source_view_id}")
                for target_id in target_ids:
                    if target_id not in by_id:
                        raise ValueError(f"unknown target view: {target_id}")
                    expected_point = expected_points.get(target_id)
                    source_view = by_id[source_view_id]
                    target_view = by_id[target_id]
                    if (
                        expected_point is None
                        and source_view.scale is not None
                        and target_view.scale is not None
                    ):
                        source_point = candidate.get("target_point_2d")
                        if not source_point and candidate.get("bbox_2d"):
                            x1, y1, x2, y2 = [float(item) for item in candidate["bbox_2d"]]
                            source_point = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                        if source_point:
                            with Image.open(source_view.image_path) as source_image, Image.open(target_view.image_path) as target_image:
                                ratio = source_view.scale / target_view.scale
                                expected_point = [
                                    target_image.width * 0.5 + (float(source_point[0]) - source_image.width * 0.5) * ratio,
                                    target_image.height * 0.5 + (float(source_point[1]) - source_image.height * 0.5) * ratio,
                                ]
                    result = self.matcher.match(
                        candidate,
                        source_view.image_path,
                        target_id,
                        target_view.image_path,
                        artifact_dir,
                        expected_point=expected_point,
                    )
                    results.append(result.to_dict())
        return {
            "provider": "dinov3",
            "matches": results,
            "visualization_paths": [item["heatmap_path"] for item in results if item.get("heatmap_path")],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "VisualGroundingMVP/1.0"

    @property
    def app(self) -> VisionService:
        return self.server.app

    def log_message(self, format: str, *args: Any) -> None:
        print("[vision-tool] " + format % args, flush=True)

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The client may close its socket while GPU inference is finishing.
            # The result remains materialized on disk; there is no response
            # channel left to write to.
            return

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.app.health())

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid request body length")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = require_mapping(payload, "request")
            if self.path == "/v1/candidates":
                result = self.app.candidates(payload)
            elif self.path == "/v1/matches":
                result = self.app.matches(payload)
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, result)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loopback visual-grounding model service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("sam3", "grounded-sam2"), default="sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-checkpoint", default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--dinov3-repo", default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--score-threshold", type=float, default=0.08)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--dino-image-size", type=int, default=768)
    parser.add_argument("--no-prewarm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("refusing non-loopback bind; --host must be 127.0.0.1")
    app = VisionService(args)
    if not args.no_prewarm:
        print(f"[vision-tool] loading {args.backend}...", flush=True)
        app.warmup()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app
    print(f"[vision-tool] ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[vision-tool] stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
