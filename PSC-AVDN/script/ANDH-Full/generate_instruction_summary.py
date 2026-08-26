#!/usr/bin/env python3
"""Generate a self-contained HTML summary for parsed instruction plans."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STYLE = r"""
:root{--bg:#f4f7fb;--paper:#fff;--ink:#17243a;--muted:#65758a;--line:#d9e3ee;--blue:#256fd1;--blue-soft:#e8f1ff;--green:#087d5b;--green-soft:#e2f6ef;--orange:#b45b08;--orange-soft:#fff0dc;--purple:#6844b5;--purple-soft:#f0eaff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.62 "Segoe UI","Microsoft YaHei",sans-serif}.hero{padding:38px max(20px,calc((100vw - 1500px)/2));background:linear-gradient(135deg,#0a1728,#164d7a);color:#fff}.hero h1{font-size:40px;margin:4px 0}.hero p{max-width:980px;color:#d3dfed;margin:4px 0}.stats{display:flex;gap:24px;flex-wrap:wrap;margin-top:20px}.stats div{min-width:100px}.stats b{display:block;font-size:25px}.stats span{color:#bcd0e5;font-size:12px}main{max-width:1500px;margin:auto;padding:24px 20px 70px}.controls{display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:10px;align-items:end;margin-bottom:14px}.controls label{display:grid;gap:5px;color:var(--muted)}input,select,button{font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:9px 10px}button{cursor:pointer}.distribution{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 22px}.distribution span{background:var(--paper);border:1px solid var(--line);border-radius:99px;padding:4px 9px;color:var(--muted)}.distribution b{color:var(--ink)}.visible-count{color:var(--muted);margin-bottom:8px}.route{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin:10px 0;overflow:hidden}.route[hidden]{display:none}.route>summary{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:13px 16px;cursor:pointer}.route>summary strong{font-size:17px;margin-right:auto}.route>summary>span:not(.route-number):not(.repair){color:var(--muted)}.route-number{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--blue-soft);color:var(--blue);font-weight:600}.repair{padding:3px 8px;border-radius:99px}.repair.no{background:var(--green-soft);color:var(--green)}.repair.yes{background:var(--orange-soft);color:var(--orange)}.route-body{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line)}.route-body>section{padding:17px;min-width:0}.route-body>section+section{border-left:1px solid var(--line)}h3{font-size:16px;margin:0 0 10px}.turns,.entities,.events{list-style:none;margin:0;padding:0}.turns li{display:grid;grid-template-columns:auto auto 1fr;gap:8px;align-items:start;margin:0 0 10px}.turns p,.entities p,.events p{margin:0}.role,.turn-index,.kind,.goal,.event-index,.event-type,.event-turn{display:inline-block;font-size:11px;border-radius:5px;padding:2px 6px;white-space:nowrap}.role.ins{background:var(--blue-soft);color:var(--blue)}.role.que{background:var(--purple-soft);color:var(--purple)}.turn-index,.event-turn{color:var(--muted);background:var(--bg)}.entities li{border-left:3px solid var(--green);padding:7px 10px;margin-bottom:8px;background:var(--green-soft)}.entities li>div{display:flex;gap:7px;align-items:center}.kind{background:var(--paper);color:var(--muted)}.goal{background:var(--orange-soft);color:var(--orange)}.events li{display:grid;grid-template-columns:auto auto auto 1fr;gap:7px;align-items:start;padding:7px 0;border-bottom:1px solid var(--line)}.event-index{background:var(--blue-soft);color:var(--blue)}.event-type{background:var(--purple-soft);color:var(--purple)}.events p{color:var(--muted);overflow-wrap:anywhere}.raw{margin-top:14px}.raw summary{cursor:pointer;color:var(--blue)}pre{background:#111d2c;color:#e8eef6;padding:13px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto}.empty{padding:35px;text-align:center;color:var(--muted)}footer{margin-top:28px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}@media(max-width:900px){.controls{grid-template-columns:1fr 1fr}.route-body{grid-template-columns:1fr}.route-body>section+section{border-left:0;border-top:1px solid var(--line)}}@media(max-width:560px){.hero h1{font-size:30px}.controls{grid-template-columns:1fr}.route>summary strong{width:calc(100% - 50px)}.events li{grid-template-columns:auto auto auto}.events p{grid-column:1/-1}}
"""


SCRIPT = r"""
(() => {
  const search = document.getElementById('plan-search');
  const repair = document.getElementById('repair-filter');
  const type = document.getElementById('type-filter');
  const count = document.getElementById('visible-count');
  const routes = [...document.querySelectorAll('.route')];
  function apply() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    routes.forEach(route => {
      const okQuery = !query || route.dataset.search.includes(query);
      const okRepair = !repair.value || route.dataset.repaired === repair.value;
      const okType = !type.value || route.dataset.types.split(' ').includes(type.value);
      route.hidden = !(okQuery && okRepair && okType);
      if (!route.hidden) visible += 1;
    });
    count.textContent = `显示 ${visible} / ${routes.length} 条轨迹`;
  }
  search.addEventListener('input', apply);
  repair.addEventListener('change', apply);
  type.addEventListener('change', apply);
  document.getElementById('expand-visible').addEventListener('click', () => {
    const shouldOpen = routes.some(route => !route.hidden && !route.open);
    routes.forEach(route => { if (!route.hidden) route.open = shouldOpen; });
  });
  apply();
})();
"""


def esc(value: Any, *, quote: bool = True) -> str:
    return html.escape(str(value), quote=quote)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def parse_dialogue(text: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"\[(INS|QUE)\]\s*", text, flags=re.IGNORECASE))
    turns: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turns.append(
            {
                "turn": index + 1,
                "role": match.group(1).upper(),
                "text": text[match.end() : end].strip(),
            }
        )
    return turns


def dataset_dialogues(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["map_name"]), int(row["route_index"]))
        result[key] = parse_dialogue(row.get("instructions", ""))
    return result


def render_turns(turns: Iterable[dict[str, Any]]) -> str:
    items = []
    for turn in turns:
        role = str(turn.get("role", "")).upper()
        items.append(
            "<li>"
            f"<span class='role {esc(role.lower())}'>{esc(role)}</span>"
            f"<span class='turn-index'>T{esc(turn.get('turn', ''))}</span>"
            f"<p>{esc(turn.get('text', ''))}</p>"
            "</li>"
        )
    return "".join(items) or "<li class='empty'>无对话内容</li>"


def render_entities(entities: list[dict[str, Any]]) -> str:
    items = []
    for entity in entities:
        goal = "<span class='goal'>GOAL</span>" if entity.get("goal") else ""
        items.append(
            "<li><div>"
            f"<code>{esc(entity.get('id', ''))}</code>"
            f"<span class='kind'>{esc(entity.get('kind', ''))}</span>{goal}"
            f"</div><p>{esc(entity.get('description', ''))}</p></li>"
        )
    return "".join(items) or "<li class='empty'>无实体</li>"


def event_description(event: dict[str, Any], entity_lookup: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    entity_id = event.get("entity")
    if entity_id is not None:
        entity = entity_lookup.get(str(entity_id), {})
        description = entity.get("description")
        value = f"entity={entity_id}"
        if description:
            value += f" ({description})"
        parts.append(value)
    direction = event.get("direction")
    if isinstance(direction, dict):
        direction_bits = []
        frame = direction.get("frame")
        angle = direction.get("angle")
        if frame is not None or angle is not None:
            direction_bits.append(f"{frame or ''} {angle if angle is not None else ''}°".strip())
        direction_bits.extend(
            str(direction[key]) for key in ("clock", "description") if direction.get(key)
        )
        if direction_bits:
            parts.append("direction=" + " / ".join(direction_bits))
    for key, label in (("distance", "distance"), ("relation", "relation")):
        if event.get(key) is not None:
            parts.append(f"{label}={event[key]}")
    spatial = event.get("spatial_constraints")
    if spatial:
        if isinstance(spatial, list):
            parts.append("spatial=" + "; ".join(map(str, spatial)))
        else:
            parts.append(f"spatial={spatial}")
    rendered_keys = {
        "id", "turn", "type", "entity", "direction", "distance", "relation", "spatial_constraints"
    }
    for key, value in event.items():
        if key not in rendered_keys and value is not None:
            parts.append(f"{key}={value}")
    return " · ".join(parts) or "无附加参数"


def render_events(events: list[dict[str, Any]], entities: list[dict[str, Any]]) -> str:
    lookup = {str(entity.get("id")): entity for entity in entities}
    items = []
    for index, event in enumerate(events, 1):
        event_id = event.get("id") or f"e{index}"
        items.append(
            "<li>"
            f"<span class='event-index'>{esc(event_id)}</span>"
            f"<span class='event-type'>{esc(event.get('type', ''))}</span>"
            f"<span class='event-turn'>T{esc(event.get('turn', ''))}</span>"
            f"<p>{esc(event_description(event, lookup))}</p>"
            "</li>"
        )
    return "".join(items) or "<li class='empty'>无事件</li>"


def render_route(
    record: dict[str, Any],
    index: int,
    original_turns: dict[tuple[str, int], list[dict[str, Any]]],
) -> str:
    plan = record.get("plan") or {}
    entities = plan.get("entities") or []
    events = plan.get("events") or []
    key = (str(record.get("map_name")), int(record.get("route_idx", 0)))
    turns = original_turns.get(key) or record.get("turns") or []
    repaired = bool(record.get("repaired"))
    repaired_text = str(repaired).lower()
    event_types = sorted({str(event.get("type", "")) for event in events if event.get("type")})
    searchable = " ".join(
        [str(record.get("plan_id", ""))]
        + [f"[{turn.get('role', '')}] {turn.get('text', '')}" for turn in turns]
        + [str(entity.get("description", "")) for entity in entities]
        + event_types
    ).lower()
    raw_plan = esc(json.dumps(plan, ensure_ascii=False, indent=2))
    open_attr = " open" if index <= 2 else ""
    heading = record.get("starting_heading_deg")
    heading_text = "—" if heading is None else f"{heading}°"
    return f"""
<details class="route" data-search="{esc(searchable)}" data-repaired="{repaired_text}" data-types="{esc(' '.join(event_types))}"{open_attr}>
  <summary>
    <span class="route-number">{index:02d}</span>
    <strong>{esc(record.get('plan_id', ''))}</strong>
    <span>{len(turns)} turns</span><span>{len(entities)} entities</span><span>{len(events)} events</span>
    <span>start heading={esc(heading_text)}</span>
    <span class="repair {'yes' if repaired else 'no'}">repaired={repaired_text}</span>
  </summary>
  <div class="route-body">
    <section><h3>原始指令</h3><ol class="turns">{render_turns(turns)}</ol></section>
    <section>
      <h3>Entities</h3><ul class="entities">{render_entities(entities)}</ul>
      <h3>Ordered events</h3><ol class="events">{render_events(events, entities)}</ol>
      <details class="raw"><summary>查看当前规范化 plan JSON</summary><pre>{raw_plan}</pre></details>
    </section>
  </div>
</details>"""


def count_nonempty_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def build_html(
    records: list[dict[str, Any]],
    dialogues: dict[tuple[str, int], list[dict[str, Any]]],
    title: str,
    subtitle: str,
    plan_source: Path,
    dataset_source: Path,
    failure_count: int,
) -> str:
    entity_count = sum(len((record.get("plan") or {}).get("entities") or []) for record in records)
    event_count = sum(len((record.get("plan") or {}).get("events") or []) for record in records)
    repaired_count = sum(bool(record.get("repaired")) for record in records)
    type_counts = Counter(
        str(event.get("type"))
        for record in records
        for event in ((record.get("plan") or {}).get("events") or [])
        if event.get("type")
    )
    type_options = "".join(
        f"<option value='{esc(event_type)}'>{esc(event_type)} ({count})</option>"
        for event_type, count in sorted(type_counts.items())
    )
    distribution = "".join(
        f"<span><b>{esc(event_type)}</b> {count}</span>"
        for event_type, count in sorted(type_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    )
    routes = "".join(render_route(record, index, dialogues) for index, record in enumerate(records, 1))
    format_versions = sorted({str(record.get("format_version", "?")) for record in records})
    version_text = ", ".join(format_versions)
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{STYLE}</style></head><body>
<header class='hero'><small>PSC-AVDN · INSTRUCTION PLAN · FORMAT {esc(version_text)}</small><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><div class='stats'><div><b>{len(records)}</b><span>解析计划</span></div><div><b>{entity_count}</b><span>实体</span></div><div><b>{event_count}</b><span>事件</span></div><div><b>{repaired_count}</b><span>结构修复后接受</span></div><div><b>{failure_count}</b><span>解析失败</span></div></div></header>
<main><section class='controls' aria-label='筛选'><label>搜索轨迹、指令或实体<input id='plan-search' type='search' placeholder='例如 2618__5、parking、REACH'></label><label>结构修复<select id='repair-filter'><option value=''>全部</option><option value='false'>repaired=false</option><option value='true'>repaired=true</option></select></label><label>事件类型<select id='type-filter'><option value=''>全部事件</option>{type_options}</select></label><button id='expand-visible' type='button'>展开 / 折叠可见项</button></section>
<div class='distribution' aria-label='事件类型分布'>{distribution}</div><div id='visible-count' class='visible-count'></div><section id='routes'>{routes}</section>
<footer>计划来源：{esc(plan_source)} · 原指令来源：{esc(dataset_source)} · format_version={esc(version_text)}</footer></main><script>{SCRIPT}</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True, type=Path, help="instruction_plans.jsonl")
    parser.add_argument("--dataset", required=True, type=Path, help="original dialogue JSON")
    parser.add_argument("--output", required=True, type=Path, help="output HTML path")
    parser.add_argument("--failures", type=Path, help="optional failures JSONL")
    parser.add_argument("--title", default="指令解析总结")
    parser.add_argument(
        "--subtitle",
        default="逐条整合原始完整 INS/QUE 对话与解析模型输出的实体、时序事件、方向、地标关系及空间约束。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.plans)
    dialogues = dataset_dialogues(args.dataset)
    content = build_html(
        records=records,
        dialogues=dialogues,
        title=args.title,
        subtitle=args.subtitle,
        plan_source=args.plans,
        dataset_source=args.dataset,
        failure_count=count_nonempty_lines(args.failures),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output} ({len(records)} plans)")


if __name__ == "__main__":
    main()
