"""Bounded Qwen tool loop that selects one candidate produced by SAM3."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import requests

from .client import VisionToolClient, VisionToolError


class GroundingInfrastructureError(RuntimeError):
    """The local SAM3 service is unavailable or has the wrong backend."""


class GroundingAgentError(RuntimeError):
    """Qwen returned an invalid or incomplete selection."""


SELECTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_visual_candidates",
            "description": "Query SAM3 for short target or anchor concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": {"type": "object", "properties": {
                            "concept_id": {"type": "string"},
                            "text": {"type": "string"},
                            "role": {"type": "string", "enum": ["target", "anchor", "context"]},
                        }, "required": ["concept_id", "text", "role"]},
                    },
                },
                "required": ["concepts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_target_selection",
            "description": "Select one registered target candidate or report absence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dest_present": {"type": "boolean"},
                    "candidate_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["dest_present", "confidence", "reason"],
            },
        },
    },
]


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _json(value: Any) -> Dict[str, Any]:
    try:
        result = json.loads(value if isinstance(value, str) else json.dumps(value))
    except Exception as exc:
        raise ValueError("invalid tool arguments") from exc
    if not isinstance(result, dict):
        raise ValueError("tool arguments must be an object")
    return result


class GroundingAgent:
    def __init__(self, *, api_key: str, qwen_url: str, model: str, vision_tool_url: str, backend: str = "sam3", timeout: float = 600.0, max_tool_calls: int = 4):
        self.api_key = api_key
        self.qwen_url = qwen_url
        self.model = model
        self.backend = backend
        self.timeout = float(timeout)
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.vision = VisionToolClient(vision_tool_url, timeout=120.0)

    def preflight(self) -> Dict[str, Any]:
        try:
            health = self.vision.health()
        except VisionToolError as exc:
            raise GroundingInfrastructureError(str(exc)) from exc
        if not health.get("ok") or not health.get("prewarmed"):
            raise GroundingInfrastructureError(f"vision tool is not ready: {health}")
        if health.get("backend") != self.backend:
            raise GroundingInfrastructureError(f"vision backend mismatch: requested={self.backend}, service={health.get('backend')}")
        return health

    def _chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.qwen_url:
            raise GroundingAgentError("Qwen URL is not configured")
        try:
            response = requests.post(
                self.qwen_url,
                json={"model": self.model, "temperature": 0, "stream": False, "enable_thinking": True,
                      "parallel_tool_calls": False, "messages": messages, "tools": SELECTION_TOOLS, "tool_choice": "auto"},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        except Exception as exc:
            raise GroundingAgentError(f"Qwen request failed: {exc}") from exc
        if not isinstance(message, dict):
            raise GroundingAgentError("Qwen returned an invalid message")
        return message

    def locate(self, *, destination: str, image_path: str, artifact_dir: str) -> Dict[str, Any]:
        main = str(Path(image_path).resolve())
        if not Path(main).is_file():
            raise GroundingAgentError(f"main view image is unavailable: {main}")
        messages: List[Dict[str, Any]] = [{
            "role": "system",
            "content": (
                "你是航拍目标选择智能体。只能使用当前768x768 scale-5主视图。"
                "先用find_visual_candidates查询一到四个简短目标/锚点概念，"
                "再根据SAM3返回的完整mask和bbox选择一个target候选。"
                "不得推导经纬度、米制距离或历史方向；证据不足时报告dest_present=false。"
            ),
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": f"目标描述：{destination}\n图像正上方是当前前进方向。"},
                {"type": "image_url", "image_url": {"url": _data_url(main)}},
            ],
        }]
        registry: Dict[str, Dict[str, Any]] = {}
        trace: List[Dict[str, Any]] = []
        for turn in range(self.max_tool_calls + 2):
            assistant = self._chat(messages)
            calls = assistant.get("tool_calls") or []
            if len(calls) != 1:
                messages.extend([assistant, {"role": "user", "content": "请调用find_visual_candidates或finish_target_selection，不要输出普通文本。"}])
                continue
            call = calls[0]
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                args = _json(function.get("arguments") or "{}")
                if name == "find_visual_candidates":
                    concepts = args.get("concepts")
                    if not isinstance(concepts, list) or not 1 <= len(concepts) <= 4:
                        raise ValueError("concepts must contain 1..4 items")
                    response = self.vision.candidates({
                        "backend": self.backend, "request_id": f"find-{turn + 1}",
                        "views": [{"view_id": "main", "role": "main", "image_path": main, "scale": 5}],
                        "concepts": concepts, "artifact_dir": artifact_dir, "max_candidates_per_view": 12,
                    })
                    candidates = [item for item in response.get("candidates") or [] if isinstance(item, dict) and item.get("candidate_id")]
                    for item in candidates:
                        registry[str(item["candidate_id"])] = item
                    result = {"candidates": candidates, "contact_sheet_paths": response.get("contact_sheet_paths") or []}
                elif name == "finish_target_selection":
                    present = bool(args.get("dest_present"))
                    confidence = max(0.0, min(1.0, float(args.get("confidence") or 0.0)))
                    reason = str(args.get("reason") or "")[:500]
                    if not present:
                        return {"dest_present": False, "bbox_2d": [0, 0, 0, 0], "confidence": confidence, "reason": reason, "tool_trace": trace}
                    candidate_id = str(args.get("candidate_id") or "")
                    candidate = registry.get(candidate_id)
                    if not candidate or candidate.get("concept_role") != "target" or candidate.get("view_id") != "main":
                        raise ValueError("candidate_id must be a registered main-view target")
                    box = [float(v) for v in candidate.get("bbox_2d") or []]
                    if len(box) != 4 or not (0 <= box[0] < box[2] <= 768 and 0 <= box[1] < box[3] <= 768):
                        raise ValueError("candidate bbox is outside the main view")
                    return {"dest_present": True, "bbox_2d": box, "confidence": confidence, "reason": reason,
                            "selected_candidate_id": candidate_id, "mask_path": str(candidate.get("mask_path") or ""),
                            "all_candidates": list(registry.values()), "tool_trace": trace}
                else:
                    raise ValueError(f"unknown tool: {name}")
            except VisionToolError as exc:
                raise GroundingInfrastructureError(str(exc)) from exc
            except (TypeError, ValueError) as exc:
                result = {"error": str(exc)}
            trace.append({"turn": turn + 1, "tool": name, "arguments": args, "result": result})
            messages.extend([assistant, {"role": "tool", "tool_call_id": str(call.get("id") or f"call-{turn + 1}"), "content": json.dumps({k: v for k, v in result.items() if k != "contact_sheet_paths"}, ensure_ascii=False)}])
            for sheet_path in result.get("contact_sheet_paths") or []:
                if Path(sheet_path).is_file():
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": "SAM3候选定位图；候选ID与工具结果一一对应："},
                        {"type": "image_url", "image_url": {"url": _data_url(sheet_path)}},
                    ]})
        raise GroundingAgentError("tool-call budget exhausted without a valid target selection")
