import os

import sys

import re

import math

import csv

import json

import base64

from pathlib import Path

from typing import Any, Dict, Iterable, Optional, Tuple, List

import numpy as np

import cv2

import requests

from tqdm import tqdm

from torch.utils.data import DataLoader

import pandas as pd

from ..config import PROJECT_ROOT

CFGPU_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

CFGPU_MODEL = "deepseek-v3"

CFGPU_API_TOKEN = os.getenv("API_TOKEN", "")

ANNO_DIR = str(PROJECT_ROOT / "datasets" / "FULL")

DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")

SPLIT = "test_unseen_full"

PRED_DIR = str(PROJECT_ROOT / "out" / "andn_core" / "parse")

MAX_STEPS = 8

SCALE_FACTOR = 3.0

_DIALOGUE_SEQUENCE_SYSTEM_PROMPT = r"""
你是一名专业的无人机导航对话解析专家。

你将收到一条轨迹的完整对话，其中包含多个 [INS] 和 [QUE]。
请把整段对话作为同一个导航任务理解，综合前后文，输出一组按实际执行顺序排列的“移动方向—搜索目标”序列。

不要机械地为每个 [INS] 输出一步。后续 [QUE]/[INS] 可能对前文进行补充、纠正、确认，
或者把前文的粗略路线细分成多个中间目标。你需要根据完整上下文生成最合理、最精简且可执行的导航步骤。

【输出格式】
只允许输出以下 JSON，不要输出解释、分析过程、Markdown 或其他内容：
{
  "steps": [
    {
      "instruction": "<本步骤简洁、独立、完整的英文导航指令>",
      "move_dir": "<时钟方向、绝对角度或 Land>",
      "dest": "<本步骤需要 Search 的具体英文视觉目标>"
    }
  ]
}

【解析规则】
1. 必须结合全部 [INS] 和 [QUE]。后续对话可以补充中间目标、修正方向、说明当前位置，
   或者给最终目标补充颜色、形状和位置属性。
2. 目标是生成完成整条路线所需的“较少且可执行的 Search 序列”，而不是复述所有移动动作、
   也不是为每个出现的地标或每个 [INS] 都生成一步。每个输出步骤都表示需要对该步 dest 执行一次 Search。
3. pass、cross、fly over、go through、after passing 等表达中的对象，既可能只是路径描述，
   也可能是后续转向或继续导航所必需的中间目标。不要根据这些动词或 road、trees、building 等类别机械判断；
   必须结合完整对话，由该对象是否仍有独立导航作用来决定是否输出。
4. 如果后续对话已经给出更清晰、更具体、可直接 Search 的目标，并且此前较泛化的途经物
   不再决定后续转向、当前位置或动作顺序，可以省略这些途经物，不要为它们生成独立步骤。
5. 如果到达某个途经物后需要改变方向、开始下一阶段，或者后续 [QUE]/[INS] 明确以它作为已到达位置，
   则它仍可作为中间 dest 保留。优先合并整段对话中对它的序号、颜色、形状及周边关系描述，使其尽量可识别。
6. [QUE] 不直接生成移动步骤，但其中对当前位置和视野状态的描述必须用作后续 [INS] 的上下文；
   如果后续 [INS] 否定了 [QUE] 对当前位置或目标的判断，以后续明确纠正为准。
7. 从完整对话中继承并合并最终目的地的具体视觉描述。当前文已经给出具体目标时，
   不要仅输出 destination、it、there 或 that building。
8. 方向来源判别：
   - 如果方向以无人机为参照，例如 your 3 o'clock、ahead、behind、left、turn left then go forward，
     输出时钟方向；
   - 如果移动方向需要根据地标或地图空间关系推断，例如 the northeastern part of the landfill、
     south side of the stadium、west of the bridge，输出 Land；
   - 如果既不以地标为参照，也不以无人机为参照，而是独立的绝对地图方向，例如
     north/east/south/west/northeast/southwest/southeast/northwest，输出对应角度：
     N=0°，NE=45°，E=90°，SE=135°，S=180°，SW=225°，W=270°，NW=315°。
   模型只负责根据文本选择输出类型和规范值，不要根据无人机初始朝向自行完成绝对角度与相对时钟之间的坐标换算；
   该换算由后续代码完成。
9. 下一步移动方向是完成必要转向后、真正开始移动前的“合成朝向”。不要把转向拆成独立步骤，
   也不要把目标相对无人机的静态方位误当作移动方向。
10. 容错：识别 oclock、o' clock、o clok、forword、lef、deirection、upposit、sic o'clock 等
    非标准拼写、口语表达和语法错误。
11. 时钟方向的最小粒度为 15°，支持 1:15、3:30、4:45 等格式。
12. dest 使用简洁、完整的英文目标描述，包括地标、具体部位或建筑等可供视觉识别的信息。
13. 当一句话同时包含“相对无人机”和“相对地标”的描述时，先判断移动方向是否确实基于地标，
    例如 the <direction> of the <landmark>；如果是则输出 Land。只有明确以无人机为参照时才输出时钟方向。
14. 连续动作合成规则，这是核心规则：
    如果同一个移动动作中同时出现多个方向表达，额外采用以下优先级：
    明确绝对移动方向 > 明确时钟方向 > turn right/left/backward 等转向方式。
    例如 go south at 7 o'clock 输出 180°；turn right and head east 输出 90°；
    turn right to 4 o'clock 输出 4:00。不要把多个方向角度叠加。
    上述优先级仅适用于同一动作。若两个方向之间存在实际移动、到达条件或 then/after/once，
    应将它们理解为不同路线阶段；但仍需结合规则 2 至 5 判断是否存在值得独立 Search 的中间目标，
    不要仅因出现多个方向或途经物就机械拆步。
15. north/south/east/west 如果只描述目标或地标的部位、朝向或相对位置，必须保留在 dest 中，
    不能用它替换移动方向。例如 building on the south side at your 7 o'clock 应输出 7:00。
16. 对话指代：结合完整上下文解析 the last building、there、it 等指代，并将其还原为具体目标描述。
17. 时钟映射参考：N=12:00，NE=1:30，E=3:00，SE=4:30，S=6:00，SW=7:30，W=9:00，NW=10:30；
    slight left/right 约为 ±15°，sharp left/right 约为 ±90°，back/behind/turn backwards 为 +180°；
    其他轻微转向指令也可以输出相应的 15° 粒度时钟方向。
18. 如果无法确定清晰的移动朝向，但可以确认移动方向以地标为参照，则输出 Land；
    如果既没有地标参照，也无法确定朝向，则保守输出 12:00。
19. right in front of you、right ahead、straight ahead、just ahead、in front of you、ahead 均输出 12:00；
    这些表达中的 right 是强调词，不表示向右。
20. dest 必须使用完整英文，可以包含途经点作为定位参考；如果目标通过途经点或参照物描述，
    应保留后续视觉定位所需的完整关系描述。

【数据集示例 1：后文已有清晰目标，省略泛化途经物】
完整对话：
[INS] head forward towards your 11 o'clock direction. After passing the ground and a road. The destination is a gray color building.
正确输出：
{
  "steps": [
    {
      "instruction": "Head towards 11 o'clock to the gray building",
      "move_dir": "11:00",
      "dest": "gray building"
    }
  ]
}
说明：ground 和 road 只是途中经过，后文已有更清晰的灰色建筑目标，因此不为它们生成 Search 步骤。

【数据集示例 2：泛化对象承担阶段切换作用，因此保留】
完整对话：
[INS] Hi drone, head forward towards 12 o'clock direction. After passing 2nd road the destination is building blocks.
[QUE] I'm at the 2nd road. Am I near the destination? In what direction is our destination?.
[INS] Proceed forward direction and turn right in nearest by brown colour building in your detination.
正确输出：
{
  "steps": [
    {
      "instruction": "Move forward to the second road",
      "move_dir": "12:00",
      "dest": "second road"
    },
    {
      "instruction": "Turn right and move to the nearest brown building",
      "move_dir": "3:00",
      "dest": "nearest brown building"
    }
  ]
}
说明：second road 虽然较泛化，但后续问答明确以它作为已到达位置，并从这里开始下一阶段，所以需要保留。

22. 只输出 JSON 对象，不要输出代码围栏或任何其他文字。
""".strip()

_CLOCK_OUTPUT_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{1,2}))?\s*(?:o\s*'?\s*clock)?\s*$",
    re.IGNORECASE,
)

_DEGREE_OUTPUT_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:°|º|˚|deg(?:ree)?s?)\s*$",
    re.IGNORECASE,
)

_BARE_NUMBER_OUTPUT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$")

def analyze_dialogue_with_prompt(
    dialogue: str,
    starting_heading: float,
    model: str = CFGPU_MODEL,
    api_token: str = CFGPU_API_TOKEN,
    base_url: str = CFGPU_URL,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "stream": False,
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": _DIALOGUE_SEQUENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"完整对话：\n{dialogue}",
            },
        ],
    }
    try:
        resp = requests.post(base_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Request to cfgpu failed: {e}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict):
            for key in ("content", "text", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key]
        raise RuntimeError(f"Unexpected response schema: {data}")

def _extract_json_object(output: str) -> Dict[str, Any]:
    text = (output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型输出中没有可解析的 JSON 对象")
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型输出 JSON 解析失败: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("模型输出的顶层必须是 JSON 对象")
    return parsed

def _normalize_move_dir(value: Any) -> str:
    def _format_degree(number: float) -> str:
        if not math.isfinite(number):
            raise ValueError(f"非法角度数值: {number}")
        degree = number % 360.0
        if degree.is_integer():
            return f"{int(degree)}°"
        return f"{degree:g}°"

    if isinstance(value, bool):
        raise ValueError("move_dir 必须是方向字符串或数值")
    if isinstance(value, (int, float)):
        return _format_degree(float(value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("move_dir 必须是非空字符串")
    move_dir = value.strip()
    if move_dir.lower() == "land":
        return "Land"

    degree_match = _DEGREE_OUTPUT_RE.fullmatch(move_dir)
    if degree_match:
        return _format_degree(float(degree_match.group(1)))

    clock_match = _CLOCK_OUTPUT_RE.fullmatch(move_dir)
    if clock_match:
        hour = int(clock_match.group(1))
        minute = int(clock_match.group(2) or 0)
        if 1 <= hour <= 12 and 0 <= minute <= 59:
            return f"{hour}:{minute:02d}"
        # A colon or an explicit "o'clock" marker unambiguously denotes an
        # invalid clock value. A bare value such as "45" may instead be an
        # absolute degree angle emitted without its degree symbol.
        if ":" in move_dir or re.search(r"o\s*'?\s*clock", move_dir, re.IGNORECASE):
            raise ValueError(f"非法时钟方向: {move_dir}")

    bare_number_match = _BARE_NUMBER_OUTPUT_RE.fullmatch(move_dir)
    if bare_number_match:
        return _format_degree(float(bare_number_match.group(1)))
    raise ValueError(f"非法方向格式: {move_dir}")

def parse_dialogue_sequence(output: str, max_steps: int = MAX_STEPS) -> List[Dict[str, str]]:
    payload = _extract_json_object(output)
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("steps 必须是 JSON 数组")
    if not 1 <= len(raw_steps) <= int(max_steps):
        raise ValueError(f"steps 数量必须在 1 至 {int(max_steps)} 之间")

    steps: List[Dict[str, str]] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"第 {index} 步必须是 JSON 对象")
        instruction = raw_step.get("instruction")
        dest = raw_step.get("dest")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"第 {index} 步 instruction 必须是非空字符串")
        if not isinstance(dest, str) or not dest.strip():
            raise ValueError(f"第 {index} 步 dest 必须是非空字符串")
        steps.append({
            "instruction": instruction.strip(),
            "move_dir": _normalize_move_dir(raw_step.get("move_dir")),
            "dest": dest.strip(),
        })
    return steps

def request_dialogue_sequence(
    dialogue: str,
    starting_heading: float,
    max_steps: int = MAX_STEPS,
    attempts: int = 2,
) -> Tuple[List[Dict[str, str]], str]:
    """请求并校验一条完整对话的导航序列。失败时只重试整段对话，不回退到逐 INS。"""
    from .termination import stop_only_instruction, stop_step
    if stop_only_instruction(dialogue):
        steps = [stop_step(dialogue)]
        return steps, json.dumps(dict(steps=steps), ensure_ascii=False)
    last_error: Optional[Exception] = None
    last_output = ""
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            last_output = analyze_dialogue_with_prompt(
                dialogue,
                starting_heading=starting_heading,
            )
            return parse_dialogue_sequence(last_output, max_steps=max_steps), last_output
        except Exception as exc:
            last_error = exc
            print(f"[WARN] full-dialogue parse attempt {attempt} failed: {exc}")
    error = RuntimeError(
        f"完整对话解析在 {max(1, int(attempts))} 次尝试后仍失败: {last_error}; "
        f"last_output={last_output[:500]!r}"
    )
    error.last_output = last_output
    raise error from last_error

def append_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    """Append one complete trajectory result and force it to disk for live monitoring."""
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())

CLOCK_ALIAS = {
    "forward": "12",
    "front": "12",
    "ahead": "12",
    "straight": "12",
    "right": "3",
    "left": "9",
    "back": "6",
    "backward": "6",
    "front-right": "1:30",
    "front-left": "10:30",
    "back-right": "4:30",
    "back-left": "7:30",
}

CLOCK_DEG_MAP = {
    0: "12:00",
    45: "1:30",
    90: "3:00",
    135: "4:30",
    180: "6:00",
    225: "7:30",
    270: "9:00",
    315: "10:30",
}

DEGREE_MARK = re.compile(r"^(\d+(?:\.\d+)?)\s*°$")

CLOCK_MARK = re.compile(r"^(\d{1,2})(?::(\d{1,2}))?$")

def clock_to_angle_deg(clock_str: str) -> float:
    s = str(clock_str).strip().replace("点", "").replace("方向", "")
    m = CLOCK_MARK.fullmatch(s)
    if not m:
        alias = CLOCK_ALIAS.get(s.lower())
        if alias:
            m = CLOCK_MARK.fullmatch(alias)
        if not m:
            return 0.0
    hour = int(m.group(1)) % 12
    minute = int(m.group(2) or 0)
    return (hour * 30.0 + minute * 0.5) % 360.0

def quantize_deg_to_clock(angle: float) -> str:
    nearest = int((((float(angle) % 360.0) + 22.5) // 45) * 45) % 360
    return CLOCK_DEG_MAP.get(nearest, CLOCK_DEG_MAP[0])

def generate_view_corners_with_scale(center_point, ob, scale_factor=1.0, angle_deg=None):
    center_point = np.array(center_point, dtype=float).reshape(2,)
    lat_min, lng_min = ob['gps_botm_left']
    lat_max, lng_max = ob['gps_top_right']
    h, w = ob['map_size'][:2]

    lat_per_px = (lat_max - lat_min) / h
    lng_per_px = (lng_max - lng_min) / w

    base_pixels = 224 * float(scale_factor)
    half_lat = (base_pixels / 2) * lat_per_px
    half_lng = (base_pixels / 2) * lng_per_px

    if angle_deg is None:
        return np.array([
            [center_point[0] + half_lat, center_point[1] - half_lng],
            [center_point[0] + half_lat, center_point[1] + half_lng],
            [center_point[0] - half_lat, center_point[1] + half_lng],
            [center_point[0] - half_lat, center_point[1] - half_lng],
        ], dtype=float)

    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    local = np.array([
        [+half_lat, -half_lng],
        [+half_lat, +half_lng],
        [-half_lat, +half_lng],
        [-half_lat, -half_lng],
    ], dtype=float)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    rot = local @ R.T
    return rot + center_point

def create_view_image(view_corners, ob, save_path=None):
    try:
        map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
        if not os.path.exists(map_path):
            print(f"[WARN] map file not found: {map_path}")
            return None

        sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if sat_map is None:
            print(f"[WARN] failed to read map: {map_path}")
            return None

        h, w = sat_map.shape[:2]
        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']

        src = []
        for lat, lng in view_corners:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        src[:, 0] = np.clip(src[:, 0], 0, w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, h - 1)

        if cv2.contourArea(src.astype(np.float32)) < 1.0:
            print("[WARN] view area too small")
            return None

        dst = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        patch = cv2.warpPerspective(sat_map, M, (224, 224))
        if patch is None or patch.size == 0:
            print("[WARN] perspective warp returned empty")
            return None

        if save_path:
            cv2.imwrite(save_path, patch)
        return patch
    except Exception as e:
        print(f"[ERROR] crop failed: {e}")
        return None

def safe_read_csv(path: Path, encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin1")) -> Tuple[pd.DataFrame, str]:
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to read CSV with provided encodings.")

CLOCK_MAP = {
    0: "12:00",
    45: "1:30",
    90: "3:00",
    135: "4:30",
    180: "6:00",
    225: "7:30",
    270: "9:00",
    315: "10:30",
}

DEGREE_MARK = re.compile(r"^(\d+(?:\.\d+)?)\s*°$")

CLOCK_MARK  = re.compile(r"^\s*(\d{1,2})(?::(\d{1,2}))?\s*$", re.I)

def degree_to_clock(angle: float) -> str:
    a = float(angle) % 360.0
    nearest = (int(((a + 22.5) // 45) * 45)) % 360
    return CLOCK_MAP.get(nearest, CLOCK_MAP[0])

def _pick_angle_baseline(row) -> float:
    for key in ("angle_before", "angle", "starting_angle"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass
    return 0.0

def convert_move_dir_row(move_dir_value, angle_baseline_value) -> str:
    move_dir = str(move_dir_value).strip()
    try:
        base = float(angle_baseline_value)
    except Exception:
        base = 0.0

    m = DEGREE_MARK.match(move_dir)
    if m:
        abs_angle = float(m.group(1)) % 360
        rel_angle = (abs_angle - base) % 360
        return degree_to_clock(rel_angle)
    else:
        return move_dir

INS_BLOCK = re.compile(r"\[\s*INS\s*\](.*?)(?:\[\s*/\s*INS\s*\]|$)", flags=re.IGNORECASE | re.DOTALL)

FORWARD_PATTERN = re.compile(
    r"(?:\b(?:go|move|fly|proceed|continue)\s*(?:straight|forward|forword)\b)"
    r"|(?:\b(?:straight(?:\s*ahead)?|forward|forword)\b)"
    r"|(?:直(?:走|行)|forward|proceed forward|笔直)",
    flags=re.IGNORECASE
)

def extract_ins(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = INS_BLOCK.search(text)
    return (m.group(1) if m else text).strip()

def has_forward(ins_text: str) -> bool:
    return bool(FORWARD_PATTERN.search(ins_text or ""))

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def clean_text(x):
    if not isinstance(x, str):
        return x
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = CONTROL_CHARS.sub("", x)
    x = re.sub(r"[ \t]+", " ", x)
    return x.strip()

