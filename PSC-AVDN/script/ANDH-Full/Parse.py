"""Parse each complete ANDH dialog into one chronological trajectory plan.

QWEN reads all INS and QUE turns together and emits entities plus only four
execution event types.  This module validates that JSON and permits one repair
request for an invalid response; it has no rule-based semantic fallback.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for import_path in (SCRIPT_DIR, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from instruction_schema import (  # noqa: E402
    FORMAT_VERSION,
    InstructionPlan,
    event_summary_rows,
    normalize_instruction_plan,
)


MODEL_URL = os.getenv(
    "PARSER_MODEL_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
MODEL_NAME = os.getenv("PARSER_MODEL_NAME", "qwen3.6-max-preview")
MODEL_API_TOKEN = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
MODEL_CONNECT_TIMEOUT = float(os.getenv("PARSER_CONNECT_TIMEOUT", "20"))
MODEL_READ_TIMEOUT = float(os.getenv("PARSER_READ_TIMEOUT", "300"))

ANNO_DIR = PROJECT_ROOT / "datasets" / "sample3"
DATASET_DIR = PROJECT_ROOT / "datasets" / "sample3"
SPLIT = "test_unseen_full"
PRED_DIR = PROJECT_ROOT / "out" / "preds_out_full_sample3_no_thinking"
MAX_DIALOGS = int(os.getenv("PARSER_MAX_DIALOGS", "0"))

TAG_RE = re.compile(r"\[(INS|QUE)\]", re.IGNORECASE)


SYSTEM_PROMPT = """
You are an expert in aerial dialogue navigation. You convert one COMPLETE
aerial-navigation dialog into one trajectory plan.
Read every INS and QUE together first, reconstruct the route from start to
finish, and only then emit JSON. Do not parse turns independently. Later QUE
and INS text can prove that an earlier local "destination" was only an
intermediate waypoint, correct an earlier interpretation, or add the visual
description needed to understand the route.

Return one JSON object with exactly these top-level keys:
- trajectory_id: copy the supplied trajectory id
- starting_heading: {"angle": number}, copied from the supplied metadata
- entities: every visual object or region used by an event
- events: the complete chronological execution sequence

Each entity has id, description, kind and goal. kind is LANDMARK, REGION,
CORRIDOR or BOUNDARY. Exactly one entity has goal=true: the final destination
of the whole dialog. Repeated uses of the same physical object share one id;
different successive waypoints are different entities.

There are exactly four event types:
- TURN: change travel direction without moving
- MOVE: move a limited, unspecified distance in the current direction with no
  visual target
- REACH: move relative to a visual target. This includes both arriving at the
  target and passing, flying over, crossing, going through, or travelling along
  it; encode the exact target-relative motion in relation
- AVOID: avoid a visual reference

PASS is not an event type. Never emit an event whose type is PASS. Convert
every pass/fly-over/cross/go-through/travel-along instruction into a REACH
event bound to that visual target, and preserve the action explicitly in its
relation field.

Every event has id (e1, e2, ...), turn and type. turn is the INS turn that
supplies the executable instruction. QUE text helps interpret the complete
route but never creates an event of its own.

Optional event fields:
- entity: required for REACH and AVOID; forbidden for TURN and MOVE
- direction: for TURN, with frame, angle, and optionally clock/description
- relation: concise relation to the event entity. For former pass-style
  actions, begin with the explicit motion phrase: "pass", "fly over", "cross",
  "go through", or "travel along". Do not weaken these to only "over" or
  "along". For ordinary arrival, retain relations such as "reach the north
  side", "behind", or "on the right side"
- distance: use "limited_unspecified" for an explicit untargeted MOVE
- spatial_constraints: list of concrete geographic relations needed to locate
  the target
- count and side when explicitly stated

Direction rules:
- Express every direction change as a separate TURN before the motion it
  controls, even when the text says "move/head/go toward 6 o'clock".
- Do not attach a direction directly to MOVE, REACH or AVOID. They use
  the current direction established by the preceding TURN.
- A direction stays active until another TURN.
- Angles are clockwise degrees. Absolute: north=0, east=90, south=180,
  west=270. Relative: forward/12=0, 3=90, 6=180, 9=270, 1=30, 7=210,
  8:30=255.
- Preserve multiple direction operations in one instruction. For example,
  "move toward 6 o'clock and turn 3 o'clock" must not lose the 6 o'clock leg.

Route rules:
- A destination description with a direction can imply TURN + REACH even if
  the motion verb is omitted, when the complete dialog later shows that the
  waypoint was reached.
- Split an explicit free-flight leg before a target into TURN, MOVE, then any
  later TURN and target event.
- If motion ends at a visual object, use REACH rather than MOVE.
- If an instruction says to pass, fly over, cross, go through, or travel along
  a visual object, also use REACH, set that object as entity, and put the exact
  action in relation. Such a REACH is complete only after satisfying the
  relation to the target, not merely after arriving at its center.
- Only emit MOVE when no visual object determines where that leg ends.
- Local uses of "destination" may be intermediate waypoints. Only the last
  destination of the complete route is goal=true.
- Keep spatial descriptions such as "north of the long side", "on the right
  side", and "behind" in relation or spatial_constraints.
- Do not output observations, queries, progress/state events, entity updates,
  explanations, or any event type outside the four listed above.

Before returning JSON, THINK THROUGH the complete dialog once more and check
the integrity of the entire event chain from start to finish:
1. Re-read all INS and QUE in chronological order and summarize the route
   internally; do not expose this reasoning in the response.
2. Ensure every stated direction change appears exactly once as TURN and no
   movement leg or landmark relation is omitted.
3. Use later QUE/INS statements to remove or revise any earlier REACH that the
   complete dialog shows was not actually reached, not visible, or only a
   mistaken local destination.
4. Ensure an untargeted MOVE has its own free-flight leg. Never emit MOVE
   immediately before REACH when both describe one continuous movement to the
   same visual target.
5. Ensure each intermediate waypoint is reached at the correct chronological
   point, each event entity is declared, and every declared entity is used by
   an event or one of its spatial constraints.
6. Ensure the final goal's REACH is the final displacement event. If TURN,
   MOVE, AVOID, or another REACH follows it, the earlier goal REACH is
   premature and must be corrected.
7. Verify the final JSON contains the complete executable trajectory rather
   than independent per-turn interpretations. Return JSON only.

Example input metadata:
trajectory_id: 1070__3
starting_heading: 90
Complete dialog:
1 INS: Destination is a building not so far from you to the southwest.
2 QUE: I am on top of the building. Is the destination in my view?
3 INS: Move towards the 6 o'clock direction and turn 3 o'clock direction and
grey color building is your destination.
4 QUE: I move to the grey building at back. What is the destination?
5 INS: Destination is a ruined courtyard that is brown and white. Go to your
three o'clock.

Example output:
{"trajectory_id":"1070__3","starting_heading":{"angle":90},"entities":[{"id":"nearby_building","description":"building not far to the southwest","kind":"LANDMARK","goal":false},{"id":"grey_building","description":"grey color building","kind":"LANDMARK","goal":false},{"id":"goal_courtyard","description":"ruined courtyard that is brown and white","kind":"REGION","goal":true}],"events":[{"id":"e1","turn":1,"type":"TURN","direction":{"frame":"absolute","angle":225,"description":"southwest"}},{"id":"e2","turn":1,"type":"REACH","entity":"nearby_building","relation":"arrive above the building"},{"id":"e3","turn":3,"type":"TURN","direction":{"frame":"relative","angle":180,"clock":"6:00"}},{"id":"e4","turn":3,"type":"MOVE","distance":"limited_unspecified"},{"id":"e5","turn":3,"type":"TURN","direction":{"frame":"relative","angle":90,"clock":"3:00"}},{"id":"e6","turn":3,"type":"REACH","entity":"grey_building","relation":"behind"},{"id":"e7","turn":5,"type":"TURN","direction":{"frame":"relative","angle":90,"clock":"3:00"}},{"id":"e8","turn":5,"type":"REACH","entity":"goal_courtyard"}]}

Second example input metadata:
trajectory_id: 579__2
starting_heading: 156
Complete dialog:
1 INS: Destination is a parking lot on the north side of a pavement lot to
your seven o'clock.
2 QUE: I see a parking lot. Am I near the destination? Where should I go now?
3 INS: No, you are not near the destination. You have to turn to 8:30 o'clock
and move forward to reach the destination.
4 QUE: I see the parking lot. Can I see the destination? How does the
destination look? Where should I go now?
5 INS: No. Head in the 1:00 direction. Fly over the off-white building with
the V-shaped roof. North of the long side of that building will be a parking
lot. Go to the north side of that lot.

Second example output:
{"trajectory_id":"579__2","starting_heading":{"angle":156},"entities":[{"id":"pavement_lot","description":"pavement lot","kind":"REGION","goal":false},{"id":"v_roof_building","description":"off-white building with a V-shaped roof","kind":"LANDMARK","goal":false},{"id":"goal_parking_lot","description":"parking lot north of the pavement lot and the V-roof building","kind":"REGION","goal":true}],"events":[{"id":"e1","turn":3,"type":"TURN","direction":{"frame":"relative","angle":255,"clock":"8:30"}},{"id":"e2","turn":3,"type":"MOVE","distance":"limited_unspecified"},{"id":"e3","turn":5,"type":"TURN","direction":{"frame":"relative","angle":30,"clock":"1:00"}},{"id":"e4","turn":5,"type":"REACH","entity":"v_roof_building","relation":"fly over"},{"id":"e5","turn":5,"type":"REACH","entity":"goal_parking_lot","relation":"reach the north side","spatial_constraints":["north of the pavement lot","north of the long side of the off-white V-roof building","initially described at relative 7:00 from the start"]}]}

Why the second example is not parsed turn by turn: Turn 1 describes the final
goal and its initial location but contains no executed movement. The parking
lot observed in QUE is explicitly rejected by the following INS. Turn 3 is
therefore an untargeted movement leg, not an early REACH. The only final REACH
is created after Turn 5 supplies the V-roof-building landmark and the complete
north-side spatial constraints.
""".strip()


def parse_tagged_turns(dialog: str) -> List[Dict[str, Any]]:
    """Return every INS/QUE span with a single 1-based chronological index."""
    text = str(dialog or "")
    matches = list(TAG_RE.finditer(text))
    turns: List[Dict[str, Any]] = []
    for index, match in enumerate(matches, start=1):
        end = matches[index].start() if index < len(matches) else len(text)
        turns.append({
            "turn": index,
            "role": match.group(1).upper(),
            "text": text[match.end():end].strip(),
        })
    return turns


def _numbered_dialog(turns: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{int(item['turn'])} {item['role']}: {item['text']}" for item in turns
    )


def _response_content(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    content = str(content or "").strip()
    if not content:
        raise ValueError("model response content is empty")
    return content


def _decode_json_object(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            raise ValueError("model content does not contain a JSON object")
    if not isinstance(value, dict):
        raise ValueError("model JSON must be an object")
    return value


def _request_chat_completion(
    messages: Sequence[Dict[str, str]],
    *,
    api_token: str,
    url: str,
    model: str,
) -> str:
    if not api_token:
        raise ValueError("QWEN_API_KEY or DASHSCOPE_API_KEY is required")
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": list(messages),
            "temperature": 0,
            "enable_thinking": False,
        },
        timeout=(MODEL_CONNECT_TIMEOUT, MODEL_READ_TIMEOUT),
    )
    response.raise_for_status()
    return _response_content(response.json())


def _parse_request_text(
    turns: Sequence[Dict[str, Any]],
    *,
    plan_id: str,
    starting_heading_deg: Optional[float],
) -> str:
    heading = "unknown" if starting_heading_deg is None else str(float(starting_heading_deg))
    return (
        f"trajectory_id: {plan_id}\n"
        f"starting_heading: {heading}\n"
        "Complete dialog:\n"
        + _numbered_dialog(turns)
    )


def _initial_messages(
    turns: Sequence[Dict[str, Any]],
    *,
    plan_id: str,
    starting_heading_deg: Optional[float],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _parse_request_text(
                turns,
                plan_id=plan_id,
                starting_heading_deg=starting_heading_deg,
            ),
        },
    ]


def _repair_messages(
    turns: Sequence[Dict[str, Any]],
    previous_content: str,
    issues: Sequence[str],
    *,
    plan_id: str,
    starting_heading_deg: Optional[float],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                _parse_request_text(
                    turns,
                    plan_id=plan_id,
                    starting_heading_deg=starting_heading_deg,
                )
                + "\n\nYour previous JSON was structurally unusable:\n"
                + previous_content
                + "\n\nFix only these structural issues:\n- "
                + "\n- ".join(str(item) for item in issues)
                + "\nReturn the complete entities/events JSON object only."
            ),
        },
    ]


def _write_failure_debug(
    path: Optional[Path],
    *,
    plan_id: str,
    attempts: Sequence[Dict[str, Any]],
) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"plan_id": plan_id, "attempts": list(attempts)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def request_model_plan(
    dialog: str,
    *,
    plan_id: str,
    starting_heading_deg: Optional[float] = None,
    api_token: str = MODEL_API_TOKEN,
    url: str = MODEL_URL,
    model: str = MODEL_NAME,
    failure_debug_path: Optional[Path] = None,
) -> Tuple[InstructionPlan, bool]:
    """Generate, normalize and minimally validate one flat plan."""
    turns = parse_tagged_turns(dialog)
    if not turns:
        raise ValueError("dialog contains no [INS] or [QUE] turns")
    messages = _initial_messages(
        turns,
        plan_id=plan_id,
        starting_heading_deg=starting_heading_deg,
    )
    attempts: List[Dict[str, Any]] = []
    previous_content = ""
    previous_issues: List[str] = []
    for attempt_index in range(2):
        if attempt_index:
            messages = _repair_messages(
                turns,
                previous_content,
                previous_issues,
                plan_id=plan_id,
                starting_heading_deg=starting_heading_deg,
            )
        content = _request_chat_completion(
            messages,
            api_token=api_token,
            url=url,
            model=model,
        )
        try:
            raw = _decode_json_object(content)
            plan, issues = normalize_instruction_plan(
                raw,
                plan_id=plan_id,
                turns=turns,
                starting_heading_deg=starting_heading_deg,
            )
        except ValueError as exc:
            plan = InstructionPlan(
                plan_id=plan_id,
                starting_heading_deg=starting_heading_deg,
                turns=turns,
            )
            issues = [str(exc)]
        attempts.append({
            "attempt": attempt_index + 1,
            "content": content,
            "issues": issues,
        })
        if not issues:
            return plan, bool(attempt_index)
        previous_content = content
        previous_issues = list(issues)
    _write_failure_debug(
        failure_debug_path,
        plan_id=plan_id,
        attempts=attempts,
    )
    raise ValueError(
        "instruction plan remained invalid after one repair: "
        + "; ".join(previous_issues)
    )


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def _write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "plan_id",
        "map_name",
        "route_idx",
        "parser_model",
        "repaired",
        "event_index",
        "turn",
        "type",
        "entity",
        "direction_frame",
        "direction_angle",
        "direction_clock",
        "relation",
        "distance",
        "spatial_constraints",
        "count",
        "side",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compile_dataset_instruction_plans(
    anno_dir: Path = ANNO_DIR,
    dataset_dir: Path = DATASET_DIR,
    split: str = SPLIT,
    pred_dir: Path = PRED_DIR,
    max_dialogs: int = MAX_DIALOGS,
    *,
    api_token: str = MODEL_API_TOKEN,
    model: str = MODEL_NAME,
    url: str = MODEL_URL,
) -> List[Dict[str, Any]]:
    """Parse a split, preserve successful plans, and report every failure."""
    if not api_token:
        raise RuntimeError("QWEN_API_KEY or DASHSCOPE_API_KEY is required for Parse.py")
    try:
        from torch.utils.data import DataLoader
        from env import ANDHNavBatch
    except ImportError as exc:
        raise RuntimeError("dataset compilation requires torch and src/env.py") from exc

    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    environment = ANDHNavBatch(
        anno_dir=str(anno_dir),
        dataset_dir=str(Path(dataset_dir) / "train_images"),
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False,
    )
    loader = DataLoader(environment, batch_size=1)
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    total = min(environment.size(), max_dialogs) if max_dialogs else environment.size()
    print(f"[parser] dialogs={total} model={model} format={FORMAT_VERSION}", flush=True)

    for dataset_index, _ in enumerate(loader):
        if max_dialogs and dataset_index >= max_dialogs:
            break
        observations = environment._get_obs(t=0)
        if not observations:
            failures.append({"dataset_index": dataset_index, "error": "no observation"})
            continue
        observation = observations[0]
        map_name = str(observation.get("map_name") or "")
        route_idx = observation.get("route_index", dataset_index)
        plan_id = f"{map_name}__{route_idx}"
        dialog = str(observation.get("instructions") or "")
        turns = parse_tagged_turns(dialog)
        started = time.perf_counter()
        try:
            plan, repaired = request_model_plan(
                dialog,
                plan_id=plan_id,
                starting_heading_deg=float(observation.get("starting_angle") or 0.0),
                api_token=api_token,
                url=url,
                model=model,
                failure_debug_path=pred_dir / "model_debug" / f"{plan_id}.json",
            )
        except (requests.RequestException, ValueError) as exc:
            failures.append({
                "plan_id": plan_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "error": str(exc),
            })
            print(f"[parser] {plan_id} FAILED: {exc}", flush=True)
            continue
        record = {
            "format_version": FORMAT_VERSION,
            "plan_id": plan_id,
            "map_name": map_name,
            "route_idx": route_idx,
            "starting_heading_deg": float(observation.get("starting_angle") or 0.0),
            "turns": turns,
            "parser_model": model,
            "repaired": repaired,
            "plan": plan.to_dict(),
        }
        records.append(record)
        for row in event_summary_rows(plan):
            row.update({
                "map_name": map_name,
                "route_idx": route_idx,
                "parser_model": model,
                "repaired": repaired,
            })
            summary_rows.append(row)
        print(
            f"[parser] {plan_id} events={len(plan.events)} entities={len(plan.entities)} "
            f"repaired={repaired} elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    _write_jsonl(pred_dir / "instruction_plans.jsonl", records)
    _write_jsonl(pred_dir / "instruction_plan_failures.jsonl", failures)
    _write_summary(pred_dir / "parsing_results_full.csv", summary_rows)
    if failures:
        raise RuntimeError(
            f"{len(failures)} instruction plans failed; see "
            f"{pred_dir / 'instruction_plan_failures.jsonl'}"
        )
    return records


def main() -> None:
    records = compile_dataset_instruction_plans()
    print(f"compiled {len(records)} instruction plans into {PRED_DIR}", flush=True)


if __name__ == "__main__":
    main()
