"""Parse complete ANDH dialogs into a flat ``InstructionPlan``.

The language model emits two lists only: entities and ordered events.  This
module performs tolerant structural normalization and at most one repair for
malformed output.  It deliberately has no rule-based semantic parser and no
second-model reviewer.
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
MODEL_READ_TIMEOUT = float(os.getenv("PARSER_READ_TIMEOUT", "180"))

ANNO_DIR = PROJECT_ROOT / "datasets" / "sample2"
DATASET_DIR = PROJECT_ROOT / "datasets" / "sample2"
SPLIT = "test_unseen_full"
PRED_DIR = PROJECT_ROOT / "out" / "preds_out_full_sample2"
MAX_DIALOGS = int(os.getenv("PARSER_MAX_DIALOGS", "0"))

TAG_RE = re.compile(r"\[(INS|QUE)\]", re.IGNORECASE)


SYSTEM_PROMPT = """
You parse a complete multi-turn aerial-navigation dialog. Return one JSON
object with exactly two top-level keys: entities and events. Do not return
markdown or explanations.

entities is a list. Each entity has:
- id: short stable identifier
- description: visual description known when the entity is first introduced;
  put later details in chronological UPDATE_ENTITY events
- kind: LANDMARK, REGION, CORRIDOR, or BOUNDARY
- goal: true only for the final destination

events is one chronological list extracted from INS turns only. Keep the
original dialogue turn number, so event turns may be 1, 3, 5, etc. Each event
has type and turn. Navigation types are MOVE, TURN, REACH, STOP_AT,
PASS, APPROACH, CROSS, GO_THROUGH, ENTER, EXIT, FOLLOW, AVOID. Dialogue types
are UPDATE_ENTITY and PROGRESS.

Optional event fields:
- entity: entity id, or a concrete description when no id was declared
- direction: {"frame":"absolute|relative", "angle": number}
- mode: FORWARD or BACKWARD
- count, side
- ref and completed for PROGRESS; ref is the 1-based index of an earlier event
- description for UPDATE_ENTITY

QUE turns are context only for understanding the following INS answer. Never
emit an event from a QUE turn, never output QUERY, and never extract entities,
progress claims, observations, or movements solely from QUE text.

Angles are clockwise degrees. Absolute 0 is north, 90 east, 180 south and 270
west. Relative 0 is forward; clock bearings are relative (12=0, 3=90, 6=180,
7=210, 9=270). Preserve the stated event order. A direction remains active
until a later event changes it, so it need not be repeated. Use PROGRESS only
when an INS turn explicitly states that an earlier event has already happened;
visibility or proximity alone is completed=false. Later descriptions from INS
turns use UPDATE_ENTITY. The final destination
must be goal=true and must have a REACH event. Do not invent
segments, motion policies, geometry, source spans, predicates or event ids.

Example 1 input:
1 INS: Go southwest at seven o'clock to the large blue-roof building.
Example 1 output:
{"entities":[{"id":"goal","description":"large blue-roof building","kind":"LANDMARK","goal":true}],"events":[{"type":"REACH","turn":1,"entity":"goal","direction":{"frame":"relative","angle":210}}]}

Example 2 input:
1 INS: Fly north, cross two roads and pass the parking lot.
2 QUE: I crossed the roads. What does the destination look like?
3 INS: It is the yellow warehouse. Continue east until you reach it.
Example 2 output:
{"entities":[{"id":"roads","description":"roads","kind":"BOUNDARY","goal":false},{"id":"parking","description":"parking lot","kind":"REGION","goal":false},{"id":"goal","description":"destination","kind":"LANDMARK","goal":true}],"events":[{"type":"CROSS","turn":1,"entity":"roads","direction":{"frame":"absolute","angle":0},"count":2},{"type":"PASS","turn":1,"entity":"parking"},{"type":"UPDATE_ENTITY","turn":3,"entity":"goal","description":"yellow warehouse"},{"type":"REACH","turn":3,"entity":"goal","direction":{"frame":"absolute","angle":90}}]}
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


def _initial_messages(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Parse these numbered turns:\n" + _numbered_dialog(turns),
        },
    ]


def _repair_messages(
    turns: Sequence[Dict[str, Any]],
    previous_content: str,
    issues: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Parse these numbered turns:\n"
                + _numbered_dialog(turns)
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
    api_token: str = MODEL_API_TOKEN,
    url: str = MODEL_URL,
    model: str = MODEL_NAME,
    failure_debug_path: Optional[Path] = None,
) -> Tuple[InstructionPlan, bool]:
    """Generate, normalize and minimally validate one flat plan."""
    turns = parse_tagged_turns(dialog)
    if not turns:
        raise ValueError("dialog contains no [INS] or [QUE] turns")
    messages = _initial_messages(turns)
    attempts: List[Dict[str, Any]] = []
    previous_content = ""
    previous_issues: List[str] = []
    for attempt_index in range(2):
        if attempt_index:
            messages = _repair_messages(turns, previous_content, previous_issues)
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
            )
        except ValueError as exc:
            plan = InstructionPlan(plan_id=plan_id, turns=turns)
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
        "mode",
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
