"""Bounded Qwen3-VL function-calling controller for visual grounding."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests

# Keep shared visual-grounding modules importable when ANDH-Full runs as a script.
import sys
_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from visual_grounding.client import VisionToolClient, VisionToolError
from visual_grounding.navigation_geometry import (
    ViewGeoTransform,
    forward_sector_min_distance_px,
)
from reach_contracts import (
    REACH_CONTRACT_VERSION,
    ReachMotionResult,
    TargetSelectionResult,
)


class GroundingAgentError(RuntimeError):
    """Recoverable model or control-loop failure."""


class GroundingInfrastructureError(RuntimeError):
    """Service/configuration failure that must fail before a route silently changes mode."""


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _safe_json(value: str) -> Dict[str, Any]:
    try:
        result = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments are not valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("arguments must be a JSON object")
    return result


def _empty_result(reason: str, trace: List[Dict[str, Any]], *, confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "dest_present": False,
        "bbox_2d": [0, 0, 0, 0],
        "raw_bbox_2d": [0, 0, 0, 0],
        "candidate_bbox_2d": [0, 0, 0, 0],
        "final_bbox_2d": None,
        "bbox_grounding": {"status": "not_requested"},
        "target_point_2d": None,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "reason": str(reason)[:500],
        "selected_candidate_id": None,
        "mask_path": "",
        "provider": "tool-agent",
        "component_scores": {},
        "tool_trace": trace,
        "view_paths": {},
    }


def _validated_point(value: Any, image_side: float = 768.0) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("target_point_2d must contain [x, y]")
    point = [float(item) for item in value]
    if not all(math.isfinite(item) for item in point):
        raise ValueError("target_point_2d must contain finite values")
    if not all(0.0 <= item < image_side for item in point):
        raise ValueError("target_point_2d is outside the 768x768 main image")
    return point


def _validated_bbox(value: Any, image_side: float = 768.0) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_2d must contain [x_min, y_min, x_max, y_max]")
    bbox = [float(item) for item in value]
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError("bbox_2d must contain finite values")
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= image_side and 0.0 <= y1 < y2 <= image_side):
        raise ValueError("bbox_2d must be an ordered in-frame 768x768 box")
    return bbox


def _compact_visual_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "concept_text": candidate.get("concept_text"),
        "concept_role": candidate.get("concept_role"),
        "forward_sector": candidate.get("forward_sector") or {
            "intersects": False,
            "minimum_distance_px": None,
        },
    }


def _selection_event_context(value: Any) -> Dict[str, Any]:
    """Keep only target-selection facts; movement relations belong to the point stage."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    active = value.get("active_event")
    if not isinstance(active, dict):
        return {}
    return {
        key: active[key]
        for key in (
            "event_id",
            "event_type",
            "target",
            "spatial_constraints",
            "instance_index",
            "instance_count",
        )
        if key in active
    }


def _selected_target_image(
    source_path: str,
    candidate: Dict[str, Any],
    artifact_dir: str,
    *,
    output_name: str,
    target_point: Optional[Sequence[float]] = None,
) -> str:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("scale-5 source image could not be read")
    image = image[:768, :768].copy()
    if image.shape[:2] != (768, 768):
        raise ValueError("scale-5 source image must contain a 768x768 main scene")

    mask_path = str(candidate.get("mask_path") or "")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    if mask is not None:
        if mask.shape[:2] != (768, 768):
            mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0
        overlay = image.copy()
        overlay[selected] = (255, 255, 0)
        image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 3, cv2.LINE_AA)
    bbox = candidate.get("bbox_2d") or []
    if len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(item))) for item in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    if target_point is not None:
        point = tuple(max(0, min(767, int(round(float(item))))) for item in target_point)
        cv2.drawMarker(image, point, (255, 0, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
    label = str(candidate.get("candidate_id") or "selected target")
    cv2.putText(image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    if not cv2.imwrite(str(output_path), image):
        raise ValueError("failed to write selected-target bbox image")
    return str(output_path.resolve())


SELECTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_visual_candidates",
            "description": (
                "Send one to four short target, anchor, or context concepts to SAM3 for the single scale-5 main image. "
                "The result contains only candidate identity and a code-computed image-space forward-sector cue; inspect the returned candidate sheet visually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "concept_id": {"type": "string"},
                                "text": {
                                    "type": "string",
                                    "description": "A category name or short noun phrase, preferably one or two words.",
                                },
                                "role": {"type": "string", "enum": ["target", "anchor", "context"]},
                            },
                            "required": ["text", "role"],
                        },
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
            "description": (
                "Submit one registered scale-5 target candidate, or report that the target is absent. "
                "This tool selects target identity only and does not accept a navigation point."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dest_present": {"type": "boolean"},
                    "candidate_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {
                        "type": "string",
                        "description": (
                            "A concise observable visual summary explaining the selected candidate and forward-sector evidence."
                        ),
                    },
                },
                "required": ["dest_present", "confidence", "reason"],
            },
        },
    },
]


POINT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "finish_navigation_point",
        "description": "Return only the relation-aware pixel navigation point for the already fixed target.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_point_2d": {
                    "type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number"},
                    "description": "[x, y] navigation point in the 768x768 main-image coordinate system.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["target_point_2d", "confidence", "reason"],
        },
    },
}]


BBOX_TOOLS = [{
    "type": "function",
    "function": {
        "name": "finish_final_bbox",
        "description": "Return the tight axis-aligned pixel bbox of the complete final target in the 768x768 scale-5 image.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array", "minItems": 4, "maxItems": 4,
                    "items": {"type": "number"},
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["bbox_2d", "confidence", "reason"],
        },
    },
}]


class GroundingAgent:
    def __init__(
        self,
        *,
        api_key: str,
        qwen_url: str,
        model: str,
        vision_tool_url: str,
        backend: str,
        qwen_api_timeout: float = 300.0,
        max_tool_calls: int = 4,
        vision_client: Optional[VisionToolClient] = None,
    ):
        self.api_key = api_key
        self.qwen_url = qwen_url
        self.model = model
        self.backend = backend
        self.qwen_api_timeout = float(qwen_api_timeout)
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.vision = vision_client or VisionToolClient(vision_tool_url, timeout=None)
        self.last_response_model = ""

    def preflight(self) -> Dict[str, Any]:
        try:
            health = self.vision.health()
        except VisionToolError as exc:
            raise GroundingInfrastructureError(str(exc)) from exc
        if not health.get("ok") or not health.get("prewarmed"):
            raise GroundingInfrastructureError(f"vision tool is not ready: {health}")
        if health.get("backend") != self.backend:
            raise GroundingInfrastructureError(
                f"vision backend mismatch: requested={self.backend}, service={health.get('backend')}"
            )
        return health

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "stream": False,
            "enable_thinking": True,
            "parallel_tool_calls": False,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": "auto",
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(
                self.qwen_url,
                json=payload,
                headers=headers,
                timeout=self.qwen_api_timeout,
            )
            response.raise_for_status()
            body = response.json()
            self.last_response_model = str(body.get("model") or self.model)
            message = body["choices"][0]["message"]
        except Exception as exc:
            raise GroundingAgentError(f"Qwen tool-call request failed: {exc}") from exc
        if not isinstance(message, dict):
            raise GroundingAgentError("Qwen returned an invalid assistant message")
        return message

    def _locate_final_bbox(
        self,
        *,
        destination: str,
        candidate_id: str,
        target_point_2d: Sequence[float],
        image_path: str,
    ) -> Dict[str, Any]:
        system = (
            "[Part One - Overall Task]\n"
            "你是航拍导航最终目标框选智能体。本次调用与候选选择调用完全独立。"
            "你只会收到一张768×768的scale-5目标定位图、精简的最终目标解析、已选candidate ID和关系感知像素点。"
            "青色mask/轮廓和红框表示已选候选，紫色十字表示导航点；像素点只用于消除目标歧义，不要求位于最终框内。"
            "不要推导经纬度、距离、航向或其他几何数据。\n\n"
            "[Part Three - Final Bbox Output]\n"
            "根据目标描述和图像，输出完整最终目标的紧致轴对齐像素框。"
            "坐标顺序必须是[x_min,y_min,x_max,y_max]，原点位于左上角，x向右、y向下，范围为0到768。"
            "框应覆盖目标完整可见范围，不要框住标签、导航点符号或无关邻近物体。"
            "调用finish_final_bbox；不要输出正方形扩展框，也不要输出候选选择结果。"
        )
        user = [{
            "type": "text",
            "text": (
                f"最终目标精简解析：{destination}\n"
                f"selected_candidate_id：{candidate_id}\n"
                f"relation-aware target_point_2d：{json.dumps(list(target_point_2d))}\n"
                "下面是选中目标专用scale-5定位图："
            ),
        }, {
            "type": "image_url",
            "image_url": {"url": _data_url(image_path)},
        }]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        failures: List[str] = []
        for attempt in range(2):
            assistant = self._chat(messages, tools=BBOX_TOOLS)
            calls = assistant.get("tool_calls") or []
            messages.append(assistant)
            try:
                if len(calls) != 1:
                    raise ValueError("Qwen must call finish_final_bbox exactly once")
                function = calls[0].get("function") or {}
                if str(function.get("name") or "") != "finish_final_bbox":
                    raise ValueError("Qwen selected an invalid bbox tool")
                args = _safe_json(function.get("arguments") or "{}")
                bbox = _validated_bbox(args.get("bbox_2d"))
                confidence = max(0.0, min(1.0, float(args.get("confidence") or 0.0)))
                return {
                    "status": "ok",
                    "bbox_2d": bbox,
                    "confidence": confidence,
                    "reason": str(args.get("reason") or "")[:500],
                    "attempt_count": attempt + 1,
                    "input_image_path": image_path,
                }
            except (TypeError, ValueError) as exc:
                failures.append(str(exc))
                if attempt == 0:
                    if calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": str(calls[0].get("id") or "bbox-call"),
                            "content": json.dumps({"accepted": False, "error": str(exc)}, ensure_ascii=False),
                        })
                    messages.append({
                        "role": "user",
                        "content": "修正坐标并重新调用finish_final_bbox。只输出合法的768×768紧致轴对齐框。",
                    })
        return {
            "status": "failed",
            "bbox_2d": None,
            "confidence": 0.0,
            "reason": failures[-1] if failures else "bbox output failed",
            "attempt_count": 2,
            "input_image_path": image_path,
            "validation_errors": failures,
        }

    def _locate_navigation_point(
        self,
        *,
        destination: str,
        event_type: str,
        relation: str,
        side: str,
        candidate_id: str,
        candidate_bbox: Sequence[float],
        main_path: str,
        selected_target_path: str,
    ) -> Dict[str, Any]:
        system = (
            "[Navigation Point - Independent Context]\n"
            "你是航拍视觉导航航点放置智能体。目标身份已经由上一阶段和本地程序固定；"
            "本次不得搜索、比较、否定或更换候选，也不得调用SAM3。"
            "你只会收到原始768×768 scale-5视图和固定目标专用定位图。"
            "原始图中的红点是当前位置，图像正上方是唯一前进方向；像素原点在左上角，x向右、y向下。"
            "专用定位图中的青色区域/轮廓是固定目标mask，红框是固定目标bbox。"
            "输出点是执行当前事件relation的下一时刻导航落点，不是目标中心或目标位置的再次估计。"
            "不得默认使用bbox中心、mask质心、候选标签位置或任何旧坐标。"
            "普通REACH可放在目标内部或可到达边界；pass/cross应放在从红点穿过目标后的可见远侧延续区域；"
            "before/in front of放在目标近侧；after/beyond放在目标远侧；beside/left/right放在相应可通行侧；"
            "AVOID放在避开障碍的可见安全侧。"
            "导航点可以位于mask或bbox外，但必须是图中可解释的有限像素点，并满足0≤x<768、0≤y<768。"
            "只调用finish_navigation_point，不要输出candidate_id或bbox。"
        )
        user: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"当前目标精简描述：{destination}\n"
                f"当前事件类型：{event_type or 'VISUAL_TARGET'}\n"
                f"当前relation：{relation or 'reach'}\n"
                f"当前side：{side or 'none'}\n"
                f"固定candidate_id：{candidate_id}\n"
                f"固定candidate_bbox_2d：{json.dumps(list(candidate_bbox), ensure_ascii=False)}\n"
                "Image 1是原始scale-5视图；Image 2是同一视图上的固定目标mask/bbox定位图。"
            ),
        }, {
            "type": "text",
            "text": "Image 1：原始768×768 main/scale-5视图：",
        }, {
            "type": "image_url",
            "image_url": {"url": _data_url(main_path)},
        }, {
            "type": "text",
            "text": "Image 2：固定目标专用mask/bbox定位图：",
        }, {
            "type": "image_url",
            "image_url": {"url": _data_url(selected_target_path)},
        }]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        failures: List[str] = []
        trace: List[Dict[str, Any]] = []
        for attempt in range(2):
            assistant = self._chat(messages, tools=POINT_TOOLS)
            calls = assistant.get("tool_calls") or []
            messages.append(assistant)
            entry: Dict[str, Any] = {
                "index": attempt + 1,
                "phase": "waypoint",
                "tool": "finish_navigation_point" if calls else "missing_tool_call",
                "arguments": {},
            }
            reasoning = str(assistant.get("reasoning_content") or "").strip()
            content = str(assistant.get("content") or "").strip()
            if reasoning:
                entry["reasoning_content"] = reasoning
            if content:
                entry["assistant_content"] = content
            try:
                if len(calls) != 1:
                    raise ValueError("Qwen must call finish_navigation_point exactly once")
                function = calls[0].get("function") or {}
                name = str(function.get("name") or "")
                if name != "finish_navigation_point":
                    raise ValueError("Qwen selected an invalid navigation-point tool")
                args = _safe_json(function.get("arguments") or "{}")
                entry["tool"] = name
                entry["arguments"] = args
                point = _validated_point(args.get("target_point_2d"))
                confidence = max(0.0, min(1.0, float(args.get("confidence") or 0.0)))
                reason = str(args.get("reason") or "")[:500]
                entry["result"] = {"accepted": True, "target_point_2d": point}
                trace.append(entry)
                return {
                    "status": "ok",
                    "target_point_2d": point,
                    "confidence": confidence,
                    "reason": reason,
                    "attempt_count": attempt + 1,
                    "trace": trace,
                }
            except (TypeError, ValueError) as exc:
                error = str(exc)
                failures.append(error)
                entry["result"] = {"accepted": False, "error": error}
                trace.append(entry)
                if attempt == 0:
                    if calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": str(calls[0].get("id") or "point-call"),
                            "content": json.dumps({"accepted": False, "error": error}, ensure_ascii=False),
                        })
                    messages.append({
                        "role": "user",
                        "content": (
                            "固定目标不变。仅修正target_point_2d并重新调用finish_navigation_point；"
                            "不要搜索或更换candidate_id。"
                        ),
                    })
        return {
            "status": "failed",
            "target_point_2d": None,
            "confidence": 0.0,
            "reason": failures[-1] if failures else "navigation point output failed",
            "attempt_count": 2,
            "validation_errors": failures,
            "trace": trace,
        }

    def run(
        self,
        *,
        destination: str,
        view_paths: Dict[str, str],
        view_scales: Optional[Dict[str, float]] = None,
        view_corners: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
        view_sizes: Optional[Dict[str, Sequence[int]]] = None,
        current_position: Optional[Sequence[float]] = None,
        heading_deg: float = 0.0,
        agent_footprint: Optional[Sequence[Sequence[float]]] = None,
        artifact_dir: str,
        entity_kind: str = "LANDMARK",
        event_type: str = "",
        relation: str = "",
        side: str = "",
        event_context: Any = None,
        is_final_goal: bool = False,
        skip_preflight: bool = False,
        phase: str = "all",
        event_id: str = "",
    ) -> Dict[str, Any]:
        if phase not in {"all", "selection"}:
            raise ValueError("GroundingAgent.run phase must be 'all' or 'selection'")
        if not skip_preflight:
            self.preflight()
        main_path = str(Path(view_paths.get("main") or "").resolve()) if view_paths.get("main") else ""
        if not main_path or not Path(main_path).is_file():
            raise GroundingAgentError("main view image is unavailable")
        available = {"main": main_path}
        main_size = list((view_sizes or {}).get("main") or [768, 768])
        if len(main_size) != 2:
            main_size = [768, 768]
        main_width, main_height = int(main_size[0]), int(main_size[1])
        current_pixel = [(main_width - 1.0) / 2.0, (main_height - 1.0) / 2.0]
        if current_position is not None and (view_corners or {}).get("main"):
            try:
                current_pixel = ViewGeoTransform(
                    (view_corners or {})["main"], main_width, main_height,
                ).latlng_to_pixel(current_position)
            except (TypeError, ValueError, IndexError):
                pass
        grounding_views = [{
            "view_id": "main",
            "role": "main",
            "image_path": main_path,
            "scale": 5.0,
        }]
        active_event = str(event_type or "VISUAL_TARGET").strip().upper()
        selection_context = _selection_event_context(event_context)
        multimodal: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"当前视觉事件类型：{active_event}\n"
                f"当前目标身份与外观：{destination}\n"
                f"当前选择上下文：{json.dumps(selection_context, ensure_ascii=False)}\n"
                f"输出角色：{'最终路线目标' if is_final_goal else '中间视觉目标'}\n"
                "只处理当前事件。图像已完成朝向校正，红点正上方是唯一前进方向。"
            ),
        }]
        multimodal.extend([
            {"type": "text", "text": "Image 1：唯一的768×768 main/scale-5图；红点是当前位置，图像正上方是前进方向："},
            {"type": "image_url", "image_url": {"url": _data_url(main_path)}},
        ])
        system = (
            "[Target Selection - Independent Context]\n"
            "你是航拍视觉导航目标选择智能体。本次只选择当前事件的真实目标实例，不生成导航点。"
            "你只接收一张768×768的main/scale-5原始图；调用find_visual_candidates后会收到SAM3候选定位图。"
            "红点是当前位置，图像正上方是唯一前进方向；图像左右只表示相对图像方向。"
            "不得根据原始指令中的东南西北、heading、角度、钟点方向或历史left/right重新推导朝向。"
            "不得推导经纬度、米制距离或跨尺度对应。"
            "find_visual_candidates使用一到四个简短名词调用SAM3，并返回候选定位图以及每个候选在红点正上方±15°范围角内的相交状态和最短像素距离。"
            "该距离只表示图像中的相对远近，不是越小越正确的评分。"
            "finish_target_selection只提交一个已注册候选或报告目标不存在。\n\n"
            "在thinking mode内部依次完成以下四阶段，不输出私有思维链、逐步推理或候选长篇分析；最终reason只保留简短、可观察视觉证据。\n\n"
            "1. 阶段一——锁定当前事件与搜索概念。\n\n"
            "只提取目标类别，颜色、形状、材质等外观属性，附近锚点，目标与锚点的空间关系，以及必须排除的其他目标或同类实例。\n\n"
            "原始对话、已完成事件、后续事件、heading、东南西北、角度、钟点方向和历史left/right均不得用于当前图像判断。\n\n"
            "根据目标、必要锚点和场景上下文准备一到四个简短名词概念；目标概念使用target角色，锚点和上下文只用于消除歧义，不能作为最终候选。\n\n"
            "2. 阶段二——只用scale-5建立当前视觉场景。\n\n"
            "先在原始图中确认红点，并把红点正上方视为唯一前进方向；区分红点上方的前进区域、左右侧区域、红点附近和红点下方，但不要给它们附会任何地理方向。\n\n"
            "观察目标外观、道路或通道的连续性、区域边界、建筑或植被的完整形态，以及目标与锚点的邻接、包围、连接、前后或左右拓扑。\n\n"
            "随后调用find_visual_candidates。拿到候选定位图后，用候选ID标签定位每个完整mask；不要把bbox中心、标签位置、缩略图中心或SAM3分数当作目标位置。\n\n"
            "若查询没有得到可辨认目标，可根据可见外观或上位类别改写为更短的名词再次查询，但不得查询后续目标，也不得虚构图中没有注册的candidate_id。\n\n"
            "3. 阶段三——逐候选做视觉比较并解释像素远近。\n\n"
            "只比较与当前目标有关的scale-5候选。对每个候选必须依次核对："
            "(a) concept_text和完整mask是否符合目标类别、外观及完整对象范围；"
            "(b) mask与锚点和周围场景的拓扑是否满足目标描述；"
            "(c) mask主要位于红点前方、侧方、当前位置附近还是后方；"
            "(d) forward_sector.intersects是否为真，并把minimum_distance_px解释为极近、有意义的前向间隔、较远或不相交。\n\n"
            "上述远近类别只能结合当前768×768图像相对比较，没有固定像素阈值。"
            "intersects只说明mask是否有像素进入红点正上方±15°范围角，minimum_distance_px只说明最近相交像素的图像远近；"
            "它们不是候选评分，也不能替代图像中的完整mask和场景拓扑。多个候选相交时不得机械选择最近或最远者。\n\n"
            "一般情况下，REACH事件应选择沿可见前进区域仍需到达、具有有意义前向间隔的目标实例。"
            "与红点几乎重合的极近候选通常表示当前位置所在、紧邻、已经到达或刚经过的对象；"
            "不得仅因它距离最小、进入范围角或mask看起来连续就把它选为仍需到达的目标。\n\n"
            "当一个候选极近，而另一个同类别候选在视觉类别、外观和场景拓扑上同样合理且位于前方、仍有有意义间隔时，"
            "REACH通常优先后者。只有图像中的目标外观、锚点关系、实例顺序或其他直接可观察证据明确说明当前极近对象就是目标时，"
            "才可以选择极近候选；reason必须写明这项额外视觉证据。不能仅因另一个候选更远就选择它，也不能因为极近候选是主干道、"
            "大mask或与当前位置相连就自动选择它。\n\n"
            "对AVOID，近处且确实阻挡可见前进区域的候选反而可能最重要。"
            "对侧向或范围角未覆盖的宽大目标，intersects=false不能单独否决，必须结合完整mask和可见拓扑。\n\n"
            "若类别、实例顺序、锚点关系或可见证据无法区分候选，应判定证据不足，不得用最近、最大、最连续或最高SAM3分数打破平局。\n\n"
            "4. 阶段四——只固定一个候选。\n\n"
            "选择必须完全依据阶段三的视觉比较。若有多个前向相交的同类候选，最终reason必须简短说明极近候选与仍需接近候选的差异，"
            "以及为什么所选实例符合当前事件；不得讨论航点、移动关系、侧向约束或像素落点。\n\n"
            "证据充分时调用finish_target_selection并只提交一个本次已注册且concept_role=target的candidate_id；"
            "无法可靠选出唯一目标时返回dest_present=false，不得猜测。"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": multimodal},
        ]
        registry: Dict[str, Dict[str, Any]] = {}
        find_calls = 0
        trace: List[Dict[str, Any]] = []
        artifacts: Dict[str, str] = {}
        start = time.monotonic()
        tool_call_count = 0
        plain_response_count = 0
        max_model_turns = self.max_tool_calls + 2
        selected_candidate: Optional[Dict[str, Any]] = None
        selection_confidence = 0.0
        selection_reason = ""
        selection_model_turns = 0

        for _model_turn in range(max_model_turns):
            assistant = self._chat(messages, tools=SELECTION_TOOLS)
            calls = assistant.get("tool_calls") or []
            if len(calls) > 1:
                raise GroundingAgentError("Qwen returned parallel tool calls while serial search is active")
            messages.append(assistant)
            if not calls:
                plain_response_count += 1
                if plain_response_count > 2:
                    raise GroundingAgentError("Qwen did not select a visual-grounding tool after two follow-ups")
                messages.append({
                    "role": "user",
                    "content": (
                        "继续完成单张scale-5视觉定位。使用简短目标/锚点名词调用find_visual_candidates，"
                        "观察候选图并只把±15°范围角相交状态和minimum_distance_px最短像素距离当作远近辅助。"
                        "忽略历史方向信息，不推导任何米制或地理几何。"
                        "最后调用finish_target_selection，只提交一个已注册target候选，"
                        "或在证据不足时报告dest_present=false。"
                    ),
                })
                continue
            tool_call_count += 1
            if tool_call_count > self.max_tool_calls:
                raise GroundingAgentError("tool-call budget exhausted without a valid finish")
            call = calls[0]
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                args = _safe_json(function.get("arguments") or "{}")
            except ValueError as exc:
                args = {}
                name = "invalid_arguments"
                argument_error = str(exc)
            else:
                argument_error = ""
            entry: Dict[str, Any] = {
                "index": tool_call_count,
                "phase": "selection",
                "tool": name,
                "arguments": args,
            }
            reasoning_content = str(assistant.get("reasoning_content") or "").strip()
            assistant_content = str(assistant.get("content") or "").strip()
            if reasoning_content:
                entry["reasoning_content"] = reasoning_content
            if assistant_content:
                entry["assistant_content"] = assistant_content

            try:
                if name == "find_visual_candidates":
                    find_calls += 1
                    if find_calls > 2:
                        raise ValueError("candidate query may be rewritten at most once")
                    concepts = args.get("concepts")
                    if not isinstance(concepts, list) or not 1 <= len(concepts) <= 4:
                        raise ValueError("concepts must contain 1..4 items")
                    response = self.vision.candidates({
                        "backend": self.backend,
                        "request_id": f"find-{find_calls}",
                        "views": grounding_views,
                        "concepts": concepts,
                        "artifact_dir": artifact_dir,
                        "entity_kind": entity_kind,
                        "event_type": event_type,
                        "max_candidates_per_view": 12,
                    })
                    response_candidates = []
                    for candidate in response.get("candidates") or []:
                        if isinstance(candidate, dict) and candidate.get("candidate_id"):
                            candidate = dict(candidate)
                            source_id = candidate.get("view_id")
                            candidate["source_image_path"] = available.get(source_id, "")
                            candidate["source_scale"] = 5.0
                            mask_path = str(candidate.get("mask_path") or "")
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
                            if mask is not None:
                                if mask.shape[:2] != (main_height, main_width):
                                    mask = cv2.resize(
                                        mask, (main_width, main_height), interpolation=cv2.INTER_NEAREST,
                                    )
                                candidate["forward_sector"] = forward_sector_min_distance_px(
                                    mask, current_pixel, half_angle_deg=15.0,
                                )
                            else:
                                candidate["forward_sector"] = {
                                    "intersects": False,
                                    "minimum_distance_px": None,
                                }
                            registry[str(candidate["candidate_id"])] = candidate
                            response_candidates.append(candidate)
                    sheets = [str(item) for item in response.get("contact_sheet_paths") or []]
                    for path in sheets:
                        label = Path(path).stem or str(len(artifacts) + 1)
                        artifacts[f"candidate_sheet_{label}"] = path
                    tool_result = {
                        "candidates": [_compact_visual_candidate(item) for item in response_candidates],
                        "contact_sheet_paths": sheets,
                    }
                elif name == "finish_target_selection":
                    present = bool(args.get("dest_present"))
                    confidence = max(0.0, min(1.0, float(args.get("confidence") or 0.0)))
                    reason = str(args.get("reason") or "")[:500]
                    if not present:
                        entry["result"] = {"accepted": True, "dest_present": False}
                        trace.append(entry)
                        result = _empty_result(reason or "agent reported destination absent", trace, confidence=confidence)
                        result["view_paths"] = artifacts
                        result["model_requested"] = self.model
                        result["model_version"] = self.last_response_model or self.model
                        result["tool_call_count"] = tool_call_count
                        result["model_turn_count"] = _model_turn + 1
                        result["latency_seconds"] = round(time.monotonic() - start, 3)
                        result["is_final_goal"] = bool(is_final_goal)
                        result.update({
                            "schema_version": REACH_CONTRACT_VERSION,
                            "status": "absent",
                            "event_id": event_id,
                            "all_candidates": list(registry.values()),
                            "candidate_sheet_paths": [
                                value for key, value in artifacts.items()
                                if key.startswith("candidate_sheet_")
                            ],
                        })
                        return result
                    candidate_id = str(args.get("candidate_id") or "")
                    if candidate_id not in registry:
                        raise ValueError("finish candidate_id was not generated by the service")
                    candidate = registry[candidate_id]
                    if candidate.get("view_id") != "main":
                        raise ValueError("finish candidate must originate in the main view")
                    if candidate.get("concept_role") != "target":
                        raise ValueError("finish candidate must have target role")
                    bbox = [float(item) for item in candidate.get("bbox_2d") or []]
                    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 768 and 0 <= bbox[1] < bbox[3] <= 768):
                        raise ValueError("service candidate bbox is outside the main view")
                    entry["result"] = {
                        "accepted": True,
                        "candidate_id": candidate_id,
                    }
                    trace.append(entry)
                    selected_candidate = candidate
                    selection_confidence = confidence
                    selection_reason = reason or "selected a service-generated candidate"
                    selection_model_turns = _model_turn + 1
                    break
                else:
                    raise ValueError(argument_error or f"unknown tool: {name}")
                entry["result"] = {
                    "candidate_count": len(tool_result.get("candidates") or []),
                    "candidates": tool_result.get("candidates") or [],
                    "error": "",
                }
            except VisionToolError as exc:
                message = str(exc)
                if "timed out" in message.lower() or "timeout" in message.lower():
                    raise GroundingInfrastructureError("visual tool request timed out") from exc
                if exc.status_code == 400 and "backend mismatch" not in message.lower():
                    tool_result = {"error": message}
                    entry["result"] = {"error": message}
                else:
                    raise GroundingInfrastructureError(message) from exc
            except (TypeError, ValueError) as exc:
                tool_result = {"error": str(exc)}
                entry["result"] = {"error": str(exc)}

            trace.append(entry)
            image_paths = tool_result.get("contact_sheet_paths") or []
            model_tool_result = {
                "candidates": tool_result.get("candidates") or [],
                **({"error": tool_result["error"]} if tool_result.get("error") else {}),
            }
            messages.append({
                "role": "tool",
                "tool_call_id": str(call.get("id") or f"call-{tool_call_count}"),
                "content": json.dumps(model_tool_result, ensure_ascii=False),
            })
            for path in image_paths[:1]:
                if Path(path).is_file():
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "SAM3 scale-5候选定位图；候选ID与上一工具结果一一对应："},
                            {"type": "image_url", "image_url": {"url": _data_url(path)}},
                        ],
                    })

            if tool_call_count >= self.max_tool_calls:
                break

        if selected_candidate is None:
            raise GroundingAgentError("tool-call budget exhausted without a valid target selection")

        candidate_id = str(selected_candidate.get("candidate_id") or "")
        bbox = [float(item) for item in selected_candidate.get("bbox_2d") or []]
        component_scores = dict(selected_candidate.get("component_scores") or {})
        component_scores["selection_confidence"] = selection_confidence

        if phase == "selection":
            return {
                "schema_version": REACH_CONTRACT_VERSION,
                "status": "selected",
                "event_id": event_id,
                "dest_present": True,
                "selected_candidate_id": candidate_id,
                "selected_candidate": selected_candidate,
                "all_candidates": list(registry.values()),
                "candidate_sheet_paths": [
                    value for key, value in artifacts.items()
                    if key.startswith("candidate_sheet_")
                ],
                "confidence": selection_confidence,
                "reason": selection_reason,
                "tool_trace": trace,
                "view_paths": artifacts,
                "artifacts": artifacts,
                "model_requested": self.model,
                "model_version": self.last_response_model or self.model,
                "tool_call_count": tool_call_count,
                "model_turn_count": selection_model_turns,
                "latency_seconds": round(time.monotonic() - start, 3),
                "provider": f"qwen3-vl+{selected_candidate.get('provider', self.backend)}",
                "is_final_goal": bool(is_final_goal),
            }

        try:
            point_image = _selected_target_image(
                main_path,
                selected_candidate,
                artifact_dir,
                output_name="selected_target_point_input.png",
            )
            artifacts["selected_target_point_input"] = point_image
            point_result = self._locate_navigation_point(
                destination=destination,
                event_type=active_event,
                relation=str(relation or ""),
                side=str(side or ""),
                candidate_id=candidate_id,
                candidate_bbox=bbox,
                main_path=main_path,
                selected_target_path=point_image,
            )
        except (GroundingAgentError, OSError, TypeError, ValueError) as exc:
            point_result = {
                "status": "failed",
                "target_point_2d": None,
                "confidence": 0.0,
                "reason": str(exc),
                "attempt_count": 0,
                "trace": [],
            }

        for point_entry in point_result.get("trace") or []:
            point_entry = dict(point_entry)
            point_entry["index"] = len(trace) + 1
            trace.append(point_entry)

        total_tool_calls = tool_call_count + len(point_result.get("trace") or [])
        total_model_turns = selection_model_turns + int(point_result.get("attempt_count") or 0)
        common = {
            "bbox_2d": bbox,
            "raw_bbox_2d": bbox,
            "candidate_bbox_2d": bbox,
            "is_final_goal": bool(is_final_goal),
            "selected_candidate_id": candidate_id,
            "selected_candidate": selected_candidate,
            "mask_path": selected_candidate.get("mask_path") or "",
            "provider": f"qwen3-vl+{selected_candidate.get('provider', self.backend)}",
            "component_scores": component_scores,
            "tool_trace": trace,
            "view_paths": artifacts,
            "model_requested": self.model,
            "model_version": self.last_response_model or self.model,
            "tool_call_count": total_tool_calls,
            "model_turn_count": total_model_turns,
            "latency_seconds": round(time.monotonic() - start, 3),
        }
        if point_result.get("status") != "ok":
            return {
                "dest_present": False,
                **common,
                "target_point_2d": None,
                "confidence": 0.0,
                "reason": str(point_result.get("reason") or "navigation point failed")[:500],
                "final_bbox_2d": None,
                "bbox_grounding": {"status": "not_requested_due_to_waypoint_failure"},
            }

        target_point = list(point_result["target_point_2d"])
        point_confidence = float(point_result.get("confidence") or 0.0)
        point_reason = str(point_result.get("reason") or "selected navigation point")[:500]
        component_scores["waypoint_confidence"] = point_confidence
        component_scores["agent_confidence"] = point_confidence
        return {
            "dest_present": True,
            **common,
            "target_point_2d": target_point,
            "confidence": point_confidence,
            "reason": point_reason,
            "final_bbox_2d": None,
            "bbox_grounding": {"status": "not_requested"},
        }

    def run_selection(self, **kwargs: Any) -> TargetSelectionResult:
        """Run only SAM3 candidate generation and target identity selection."""
        result = self.run(**kwargs, phase="selection")
        return TargetSelectionResult.from_dict(result)

    def run_motion(
        self,
        *,
        destination: str,
        selection: Any,
        view_paths: Dict[str, str],
        artifact_dir: str,
        event_id: str = "",
        event_type: str = "",
        relation: str = "",
        side: str = "",
    ) -> ReachMotionResult:
        """Generate a navigation point from a fixed selection snapshot.

        This method intentionally has no VisionToolClient access path: the selected
        candidate and its mask are the complete input from the previous stage.
        """
        started = time.monotonic()
        try:
            if isinstance(selection, TargetSelectionResult):
                selected = selection
            else:
                selected = TargetSelectionResult.from_dict(selection)
            if selected.status != "selected" or not selected.selected_candidate:
                return ReachMotionResult(
                    status="failed",
                    reason=f"selection status is {selected.status!r}; motion requires selected",
                    latency_seconds=round(time.monotonic() - started, 3),
                )
            candidate = dict(selected.selected_candidate)
            candidate_id = str(selected.selected_candidate_id or candidate.get("candidate_id") or "")
            main_path = str(Path(view_paths.get("main") or "").resolve())
            if not main_path or not Path(main_path).is_file():
                raise GroundingAgentError("main view image is unavailable for motion stage")
            bbox = [float(item) for item in candidate.get("bbox_2d") or []]
            if len(bbox) != 4:
                raise ValueError("selection snapshot candidate has no valid bbox")
            point_image = _selected_target_image(
                main_path,
                candidate,
                artifact_dir,
                output_name="selected_target_point_input.png",
            )
            point_result = self._locate_navigation_point(
                destination=destination,
                event_type=str(event_type or ""),
                relation=str(relation or ""),
                side=str(side or ""),
                candidate_id=candidate_id,
                candidate_bbox=bbox,
                main_path=main_path,
                selected_target_path=point_image,
            )
            artifacts = {"selected_target_point_input": point_image}
            trace = [dict(item) for item in point_result.get("trace") or []]
            if point_result.get("status") != "ok":
                return ReachMotionResult(
                    status="failed",
                    reason=str(point_result.get("reason") or "navigation point failed"),
                    tool_trace=trace,
                    artifacts=artifacts,
                    latency_seconds=round(time.monotonic() - started, 3),
                    tool_call_count=len(trace),
                )
            return ReachMotionResult(
                status="planned",
                target_point_2d=list(point_result["target_point_2d"]),
                confidence=float(point_result.get("confidence") or 0.0),
                reason=str(point_result.get("reason") or "selected navigation point"),
                tool_trace=trace,
                artifacts=artifacts,
                latency_seconds=round(time.monotonic() - started, 3),
                tool_call_count=len(trace),
            )
        except (GroundingAgentError, OSError, TypeError, ValueError) as exc:
            return ReachMotionResult(
                status="failed",
                reason=str(exc),
                latency_seconds=round(time.monotonic() - started, 3),
            )
