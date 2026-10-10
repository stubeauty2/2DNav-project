"""Loopback-only JSON service for SAM3 candidate generation."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw

from .backends import Sam3Backend
from .contracts import parse_concepts, parse_views
from .postprocess import filter_proposals, materialize_candidates

DEFAULT_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", r"E:\Levir\Graduation_Project\sam3\sam3.pt")


class VisionService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = Sam3Backend(args.sam3_checkpoint, args.device, args.score_threshold)

    def warmup(self) -> None:
        self.backend.load()

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "backend": "sam3",
            "prewarmed": bool(self.backend.loaded),
            "checkpoint": self.backend.checkpoint,
            "score_threshold": self.args.score_threshold,
        }

    def candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(payload.get("backend") or "sam3") != "sam3":
            raise ValueError("backend mismatch: service only provides sam3")
        views = parse_views(payload.get("views"))
        concepts = parse_concepts(payload.get("concepts"))
        artifact_dir = str(payload.get("artifact_dir") or "sam3_outputs")
        candidates = []
        for view in views:
            image_path = Path(view.image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
            with Image.open(image_path) as image:
                for concept in concepts:
                    proposals = self.backend.propose(image, concept.text)
                    proposals = filter_proposals(proposals, top_k=int(payload.get("max_candidates_per_view") or 12))
                    candidates.extend(materialize_candidates(proposals, concept=concept, view=view, artifact_dir=artifact_dir))
        sheets = []
        if candidates:
            main_view = next((item for item in views if item.view_id == "main"), views[0])
            with Image.open(main_view.image_path) as source:
                sheet = source.convert("RGB")
            draw = ImageDraw.Draw(sheet)
            for item in candidates:
                x1, y1, x2, y2 = [int(round(v)) for v in item["bbox_2d"]]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
                draw.text((x1 + 3, max(0, y1 - 14)), item["candidate_id"], fill=(255, 0, 0))
            sheet_path = Path(artifact_dir) / "sam3_candidates.jpg"
            sheet.save(sheet_path, quality=95)
            sheets.append(str(sheet_path.resolve()))
        return {"backend": "sam3", "candidates": candidates, "contact_sheet_paths": sheets}


class Handler(BaseHTTPRequestHandler):
    server_version = "PSCAVDNSAM3/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print("[sam3-service] " + format % args, flush=True)

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.server.app.health())

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid request body length")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path != "/v1/candidates":
                self._send(404, {"error": "not found"})
                return
            self._send(200, self.server.app.candidates(payload))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loopback SAM3 candidate service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--score-threshold", type=float, default=0.08)
    parser.add_argument("--no-prewarm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("refusing non-loopback bind; --host must be 127.0.0.1")
    app = VisionService(args)
    if not args.no_prewarm:
        print("[sam3-service] loading SAM3...", flush=True)
        app.warmup()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app
    print(f"[sam3-service] ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
