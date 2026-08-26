"""Minimal contract for a trajectory parsed from one complete dialog."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


FORMAT_VERSION = 3


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DirectionFrame(StringEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class EntityKind(StringEnum):
    LANDMARK = "LANDMARK"
    REGION = "REGION"
    CORRIDOR = "CORRIDOR"
    BOUNDARY = "BOUNDARY"


class EventType(StringEnum):
    MOVE = "MOVE"
    TURN = "TURN"
    REACH = "REACH"
    AVOID = "AVOID"


NAVIGATION_EVENT_TYPES = frozenset(EventType)
TARGET_EVENT_TYPES = frozenset({EventType.REACH, EventType.AVOID})


def finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_angle(value: float) -> float:
    return float(value) % 360.0


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or fallback


def _entity_kind(value: Any) -> EntityKind:
    name = str(value or "LANDMARK").strip().upper()
    aliases = {
        "OBJECT": "LANDMARK",
        "BUILDING": "LANDMARK",
        "AREA": "REGION",
        "ROAD": "CORRIDOR",
        "LINE": "BOUNDARY",
    }
    try:
        return EntityKind(aliases.get(name, name))
    except ValueError:
        return EntityKind.LANDMARK


@dataclass(frozen=True)
class Direction:
    frame: DirectionFrame
    angle: float
    clock: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "Direction":
        if not isinstance(raw, dict):
            raise ValueError("direction must be an object")
        try:
            frame = DirectionFrame(str(raw.get("frame") or "").strip().lower())
        except ValueError as exc:
            raise ValueError("direction.frame must be absolute or relative") from exc
        angle = finite_float(raw.get("angle"))
        if angle is None:
            raise ValueError("direction.angle must be a finite number")
        return cls(
            frame=frame,
            angle=normalize_angle(angle),
            clock=str(raw.get("clock") or "").strip(),
            description=str(raw.get("description") or "").strip(),
        )

    def resolve(self, body_heading_deg: float) -> float:
        if self.frame == DirectionFrame.ABSOLUTE:
            return normalize_angle(self.angle)
        return normalize_angle(float(body_heading_deg) + self.angle)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"frame": self.frame.value, "angle": self.angle}
        if self.clock:
            result["clock"] = self.clock
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class Entity:
    entity_id: str
    description: str
    kind: EntityKind = EntityKind.LANDMARK
    goal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entity_id,
            "description": self.description,
            "kind": self.kind.value,
            "goal": bool(self.goal),
        }


@dataclass
class InstructionEvent:
    event_id: str
    event_type: EventType
    turn: int
    entity_ref: str = ""
    direction: Optional[Direction] = None
    relation: str = ""
    distance: str = ""
    spatial_constraints: List[str] = field(default_factory=list)
    count: int = 1
    side: str = ""
    source_span: str = ""

    @property
    def turn_id(self) -> int:
        return self.turn

    @property
    def required(self) -> bool:
        return True

    @property
    def requires_visual_search(self) -> bool:
        return bool(self.entity_ref)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": self.event_id,
            "turn": self.turn,
            "type": self.event_type.value,
        }
        if self.entity_ref:
            result["entity"] = self.entity_ref
        if self.direction is not None:
            result["direction"] = self.direction.to_dict()
        if self.relation:
            result["relation"] = self.relation
        if self.distance:
            result["distance"] = self.distance
        if self.spatial_constraints:
            result["spatial_constraints"] = list(self.spatial_constraints)
        if self.count != 1:
            result["count"] = self.count
        if self.side:
            result["side"] = self.side
        return result


@dataclass
class InstructionPlan:
    plan_id: str
    starting_heading_deg: Optional[float] = None
    entities: Dict[str, Entity] = field(default_factory=dict)
    events: List[InstructionEvent] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        *,
        plan_id: str = "",
        turns: Optional[List[Dict[str, Any]]] = None,
        starting_heading_deg: Optional[float] = None,
    ) -> "InstructionPlan":
        plan, errors = normalize_instruction_plan(
            raw,
            plan_id=plan_id,
            turns=turns or [],
            starting_heading_deg=starting_heading_deg,
        )
        if errors:
            raise ValueError("invalid InstructionPlan: " + "; ".join(errors))
        return plan

    def to_dict(self) -> Dict[str, Any]:
        heading = (
            {"angle": normalize_angle(self.starting_heading_deg)}
            if self.starting_heading_deg is not None
            else None
        )
        return {
            "trajectory_id": self.plan_id,
            "starting_heading": heading,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "events": [event.to_dict() for event in self.events],
        }

    def validate(self) -> List[str]:
        errors: List[str] = []
        goal_ids = {key for key, entity in self.entities.items() if entity.goal}
        if len(goal_ids) != 1:
            errors.append("entities must contain exactly one final goal")
        if not self.events:
            errors.append("events must contain at least one event")
        for event in self.events:
            if event.entity_ref and event.entity_ref not in self.entities:
                errors.append(
                    f"event {event.event_id} references missing entity {event.entity_ref}"
                )
            if event.event_type in TARGET_EVENT_TYPES and not event.entity_ref:
                errors.append(f"event {event.event_id} requires an entity")
            if event.event_type in {EventType.MOVE, EventType.TURN} and event.entity_ref:
                errors.append(f"event {event.event_id} must not reference an entity")
            if event.event_type == EventType.TURN and event.direction is None:
                errors.append(f"event {event.event_id} requires a direction")
            if event.event_type != EventType.TURN and event.direction is not None:
                errors.append(f"event {event.event_id} must not carry a direction")
        if goal_ids and not any(
            event.event_type == EventType.REACH and event.entity_ref in goal_ids
            for event in self.events
        ):
            errors.append("the final goal must be bound to a REACH event")
        return errors

    def source_for_turn(self, turn: int) -> str:
        for item in self.turns:
            if int(item.get("turn") or 0) == int(turn):
                return str(item.get("text") or "")
        return ""


def _starting_heading(raw: Dict[str, Any], supplied: Optional[float]) -> Optional[float]:
    heading: Any = supplied
    if heading is None:
        heading = raw.get("starting_heading")
        if isinstance(heading, dict):
            heading = heading.get("angle")
    value = finite_float(heading)
    return normalize_angle(value) if value is not None else None


def normalize_instruction_plan(
    raw: Any,
    *,
    plan_id: str,
    turns: List[Dict[str, Any]],
    starting_heading_deg: Optional[float] = None,
) -> Tuple[InstructionPlan, List[str]]:
    """Normalize one QWEN response and return all contract violations."""
    if not isinstance(raw, dict):
        return InstructionPlan(plan_id=plan_id, turns=list(turns)), [
            "model output must be a JSON object"
        ]

    errors: List[str] = []
    raw_entities = raw.get("entities")
    raw_events = raw.get("events")
    if not isinstance(raw_entities, list):
        errors.append("entities must be a list")
        raw_entities = []
    if not isinstance(raw_events, list):
        errors.append("events must be a list")
        raw_events = []

    entities: Dict[str, Entity] = {}
    descriptions: Dict[str, str] = {}
    for index, item in enumerate(raw_entities, start=1):
        if not isinstance(item, dict):
            errors.append(f"entity {index} must be an object")
            continue
        description = str(item.get("description") or "").strip()
        entity_id = _slug(item.get("id") or description, f"entity_{index}")
        if not description:
            errors.append(f"entity {index} requires a description")
            description = entity_id.replace("_", " ")
        if entity_id in entities:
            errors.append(f"entity {index} duplicates id {entity_id}")
            continue
        entities[entity_id] = Entity(
            entity_id=entity_id,
            description=description,
            kind=_entity_kind(item.get("kind")),
            goal=item.get("goal") is True,
        )
        descriptions[description.casefold()] = entity_id

    turn_text = {
        int(item.get("turn") or 0): str(item.get("text") or "")
        for item in turns if isinstance(item, dict)
    }
    turn_roles = {
        int(item.get("turn") or 0): str(item.get("role") or "").strip().upper()
        for item in turns if isinstance(item, dict)
    }
    events: List[InstructionEvent] = []
    for index, item in enumerate(raw_events, start=1):
        if not isinstance(item, dict):
            errors.append(f"event {index} must be an object")
            continue
        turn = finite_float(item.get("turn"))
        if turn is None or turn < 1 or int(turn) != turn:
            errors.append(f"event {index} has invalid turn")
            continue
        turn_number = int(turn)
        if turn_roles and turn_roles.get(turn_number) != "INS":
            errors.append(f"event {index} must originate from an INS turn")
            continue
        raw_event_type = str(item.get("type") or "").strip().upper()
        try:
            event_type = EventType(raw_event_type)
        except ValueError:
            if raw_event_type == "PASS":
                errors.append(
                    f"event {index} uses removed type PASS; use REACH with an "
                    "explicit pass/fly-over/cross/go-through/travel-along relation"
                )
            else:
                errors.append(f"event {index} has unknown type {item.get('type')!r}")
            continue

        direction = None
        if item.get("direction") not in (None, ""):
            try:
                direction = Direction.from_dict(item["direction"])
            except ValueError as exc:
                errors.append(f"event {index}: {exc}")

        raw_entity = str(item.get("entity") or "").strip()
        entity_ref = ""
        if raw_entity:
            slug = _slug(raw_entity, "entity")
            if slug in entities:
                entity_ref = slug
            elif raw_entity.casefold() in descriptions:
                entity_ref = descriptions[raw_entity.casefold()]
            else:
                errors.append(f"event {index} references undeclared entity {raw_entity}")

        constraints = item.get("spatial_constraints") or []
        if isinstance(constraints, str):
            constraints = [constraints]
        if not isinstance(constraints, list):
            errors.append(f"event {index} spatial_constraints must be a list")
            constraints = []
        count_value = finite_float(item.get("count"))
        count = max(1, int(count_value)) if count_value is not None else 1
        events.append(InstructionEvent(
            event_id=f"e{len(events) + 1}",
            event_type=event_type,
            turn=turn_number,
            entity_ref=entity_ref,
            direction=direction,
            relation=str(item.get("relation") or "").strip(),
            distance=str(item.get("distance") or "").strip(),
            spatial_constraints=[str(value).strip() for value in constraints if str(value).strip()],
            count=count,
            side=str(item.get("side") or "").strip().lower(),
            source_span=turn_text.get(turn_number, ""),
        ))

    output_plan_id = str(plan_id or raw.get("trajectory_id") or "").strip()
    plan = InstructionPlan(
        plan_id=output_plan_id,
        starting_heading_deg=_starting_heading(raw, starting_heading_deg),
        entities=entities,
        events=events,
        turns=list(turns),
    )
    errors.extend(plan.validate())
    return plan, list(dict.fromkeys(errors))


def event_summary_rows(plan: InstructionPlan) -> Iterable[Dict[str, Any]]:
    for index, event in enumerate(plan.events, start=1):
        direction = event.direction.to_dict() if event.direction is not None else {}
        yield {
            "plan_id": plan.plan_id,
            "event_index": index,
            "turn": event.turn,
            "type": event.event_type.value,
            "entity": event.entity_ref,
            "direction_frame": direction.get("frame", ""),
            "direction_angle": direction.get("angle", ""),
            "direction_clock": direction.get("clock", ""),
            "relation": event.relation,
            "distance": event.distance,
            "spatial_constraints": " | ".join(event.spatial_constraints),
            "count": event.count,
            "side": event.side,
            "source": event.source_span,
        }
