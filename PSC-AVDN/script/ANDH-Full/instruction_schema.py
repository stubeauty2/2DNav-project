"""Minimal instruction-plan contract shared by parsing and execution.

The language model emits only ``entities`` and an ordered ``events`` list.
Runtime-only identifiers and source text are derived here and are never part
of the model contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


FORMAT_VERSION = 1
class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DirectionFrame(StringEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class TravelMode(StringEnum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"


class EntityKind(StringEnum):
    LANDMARK = "LANDMARK"
    REGION = "REGION"
    CORRIDOR = "CORRIDOR"
    BOUNDARY = "BOUNDARY"


class EventType(StringEnum):
    MOVE = "MOVE"
    TURN = "TURN"
    REACH = "REACH"
    STOP_AT = "STOP_AT"
    PASS = "PASS"
    APPROACH = "APPROACH"
    CROSS = "CROSS"
    GO_THROUGH = "GO_THROUGH"
    ENTER = "ENTER"
    EXIT = "EXIT"
    FOLLOW = "FOLLOW"
    AVOID = "AVOID"
    UPDATE_ENTITY = "UPDATE_ENTITY"
    PROGRESS = "PROGRESS"


NAVIGATION_EVENT_TYPES = frozenset({
    EventType.MOVE,
    EventType.TURN,
    EventType.REACH,
    EventType.STOP_AT,
    EventType.PASS,
    EventType.APPROACH,
    EventType.CROSS,
    EventType.GO_THROUGH,
    EventType.ENTER,
    EventType.EXIT,
    EventType.FOLLOW,
    EventType.AVOID,
})

TARGET_EVENT_TYPES = NAVIGATION_EVENT_TYPES - {EventType.MOVE, EventType.TURN}
STATE_EVENT_TYPES = frozenset({
    EventType.PROGRESS,
})

_EVENT_ALIASES = {
    "STOP": "STOP_AT",
    "ARRIVE": "REACH",
    "ARRIVE_AT": "REACH",
    "GO_THROUGH": "GO_THROUGH",
    "GO THROUGH": "GO_THROUGH",
    "GOAL_REFINEMENT": "UPDATE_ENTITY",
    "ENTITY_UPDATE": "UPDATE_ENTITY",
    "PROGRESS_ASSERTION": "PROGRESS",
}

_IGNORED_EVENT_NAMES = frozenset({"QUERY", "ORIENTATION_QUERY"})


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


def _event_type(value: Any) -> Optional[EventType]:
    name = str(value or "").strip().upper().replace("-", "_")
    name = _EVENT_ALIASES.get(name, name)
    try:
        return EventType(name)
    except ValueError:
        return None


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

    @classmethod
    def from_dict(cls, raw: Any) -> "Direction":
        if not isinstance(raw, dict):
            raise ValueError("direction must be an object")
        frame_text = str(raw.get("frame") or "").strip().lower()
        try:
            frame = DirectionFrame(frame_text)
        except ValueError as exc:
            raise ValueError("direction.frame must be absolute or relative") from exc
        angle = finite_float(raw.get("angle"))
        if angle is None:
            raise ValueError("direction.angle must be a finite number")
        return cls(frame=frame, angle=normalize_angle(angle))

    def resolve(self, body_heading_deg: float) -> float:
        if self.frame == DirectionFrame.ABSOLUTE:
            return normalize_angle(self.angle)
        return normalize_angle(float(body_heading_deg) + self.angle)

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame.value, "angle": self.angle}


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
    travel_mode: TravelMode = TravelMode.FORWARD
    count: int = 1
    side: str = ""
    ref: str = ""
    completed: bool = False
    description: str = ""
    source_span: str = ""

    @property
    def turn_id(self) -> int:
        return self.turn

    @property
    def required(self) -> bool:
        return self.event_type in NAVIGATION_EVENT_TYPES

    @property
    def requires_visual_search(self) -> bool:
        """Whether executing this event requires entity-grounded vision.

        Visual verification follows the data contract, rather than a hard-coded
        list of verbs: every executable event carrying an entity is searched
        and confirmed.  This also covers a model-emitted ``MOVE`` with an
        entity, although the parser prompt normally represents that relation as
        ``APPROACH``.
        """
        return self.required and bool(self.entity_ref)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.event_type.value, "turn": self.turn}
        if self.entity_ref:
            result["entity"] = self.entity_ref
        if self.direction is not None:
            result["direction"] = self.direction.to_dict()
        if self.travel_mode != TravelMode.FORWARD:
            result["mode"] = self.travel_mode.value
        if self.count != 1:
            result["count"] = self.count
        if self.side:
            result["side"] = self.side
        if self.event_type == EventType.PROGRESS and self.ref:
            try:
                result["ref"] = int(self.ref[1:]) if self.ref.startswith("e") else self.ref
            except ValueError:
                result["ref"] = self.ref
        if self.event_type == EventType.PROGRESS:
            result["completed"] = bool(self.completed)
        if self.event_type == EventType.UPDATE_ENTITY and self.description:
            result["description"] = self.description
        return result


@dataclass
class InstructionPlan:
    plan_id: str
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
    ) -> "InstructionPlan":
        plan, errors = normalize_instruction_plan(
            raw,
            plan_id=plan_id,
            turns=turns or [],
        )
        if errors:
            raise ValueError("invalid InstructionPlan: " + "; ".join(errors))
        return plan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "events": [event.to_dict() for event in self.events],
        }

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.entities:
            errors.append("entities must contain at least one entity")
        executable = [event for event in self.events if event.required]
        if not executable:
            errors.append("events must contain at least one executable event")
        for event in self.events:
            if event.entity_ref and event.entity_ref not in self.entities:
                errors.append(
                    f"event {event.event_id} references missing entity {event.entity_ref}"
                )
            if event.event_type in TARGET_EVENT_TYPES and not event.entity_ref:
                errors.append(f"event {event.event_id} requires an entity")
            if event.event_type == EventType.UPDATE_ENTITY:
                errors.append(
                    f"event {event.event_id} was not converted to an entity-bound event"
                )
            if event.event_type == EventType.PROGRESS and event.ref:
                try:
                    ref_index = int(event.ref[1:])
                except (TypeError, ValueError):
                    ref_index = -1
                event_index = int(event.event_id[1:])
                if ref_index < 1 or ref_index >= event_index:
                    errors.append(f"event {event.event_id} has invalid progress ref {event.ref}")
        goal_ids = {key for key, entity in self.entities.items() if entity.goal}
        if not any(
            event.requires_visual_search and event.entity_ref in goal_ids
            for event in self.events
        ):
            errors.append(
                "events must contain an entity-bound navigation event for a goal entity"
            )
        return errors

    def source_for_turn(self, turn: int) -> str:
        for item in self.turns:
            if int(item.get("turn") or 0) == int(turn):
                return str(item.get("text") or "")
        return ""


def _resolve_ref(raw: Any) -> str:
    if isinstance(raw, bool) or raw in (None, ""):
        return ""
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return f"e{max(1, int(raw))}"
    text = str(raw).strip().lower()
    if re.fullmatch(r"e\d+", text):
        return text
    if text.isdigit():
        return f"e{int(text)}"
    return text


def _unique_entity_id(
    entities: Dict[str, Entity],
    description: str,
    fallback: str,
) -> str:
    base = _slug(description, fallback)
    candidate = base
    suffix = 2
    while candidate in entities:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _replace_update_entity_events(
    entities: Dict[str, Entity],
    events: List[InstructionEvent],
) -> List[InstructionEvent]:
    """Migrate legacy UPDATE_ENTITY records to independent visual entities.

    Each legacy description is intentionally treated as a new entity even when
    it may refer to the same physical object.  The new entity is bound to the
    most relevant navigation event in the same INS turn.  When the turn only
    has an ungrounded MOVE, its last MOVE becomes APPROACH; when it has no such
    event, UPDATE_ENTITY is replaced in place by a synthetic APPROACH.
    """
    update_indices = [
        index
        for index, event in enumerate(events)
        if event.event_type == EventType.UPDATE_ENTITY
    ]
    if not update_indices:
        return events

    removed_indices = set(update_indices)
    claimed_targets: set[int] = set()
    replacement_refs: Dict[str, str] = {}
    synthetic_by_index: Dict[int, InstructionEvent] = {}
    goal_lineages = {
        entity_id
        for entity_id, entity in entities.items()
        if entity.goal
    }
    last_goal_update: Dict[str, int] = {}
    for index in update_indices:
        entity_ref = events[index].entity_ref
        if entity_ref in goal_lineages:
            last_goal_update[entity_ref] = index
    for entity_ref in last_goal_update:
        entities[entity_ref].goal = False

    for update_index in update_indices:
        update = events[update_index]
        original = entities.get(update.entity_ref)
        description = str(update.description or "").strip()
        if not description:
            description = (
                original.description if original is not None else update.entity_ref
            )
        new_entity_id = _unique_entity_id(
            entities,
            description,
            f"{update.entity_ref or 'entity'}_t{update.turn}",
        )
        entities[new_entity_id] = Entity(
            entity_id=new_entity_id,
            description=description,
            kind=original.kind if original is not None else EntityKind.LANDMARK,
            goal=last_goal_update.get(update.entity_ref) == update_index,
        )

        same_turn = [
            index
            for index in range(update_index + 1, len(events))
            if events[index].turn == update.turn
            and index not in removed_indices
            and index not in claimed_targets
        ]
        target_index = next(
            (
                index
                for index in same_turn
                if events[index].event_type in TARGET_EVENT_TYPES
                and events[index].entity_ref == update.entity_ref
            ),
            None,
        )
        if target_index is None:
            move_indices = [
                index
                for index in same_turn
                if events[index].event_type == EventType.MOVE
                and not events[index].entity_ref
            ]
            if move_indices:
                # A destination description commonly precedes a sequence such
                # as MOVE -> checkpoint -> TURN -> MOVE.  The final MOVE is the
                # one that approaches the newly described destination.
                target_index = move_indices[-1]

        if target_index is not None:
            target = events[target_index]
            target.entity_ref = new_entity_id
            if target.event_type == EventType.MOVE:
                target.event_type = EventType.APPROACH
            claimed_targets.add(target_index)
            replacement_refs[update.event_id] = target.event_id
            continue

        synthetic = InstructionEvent(
            event_id=update.event_id,
            event_type=EventType.APPROACH,
            turn=update.turn,
            entity_ref=new_entity_id,
            direction=update.direction,
            travel_mode=update.travel_mode,
            count=update.count,
            side=update.side,
            source_span=update.source_span,
        )
        synthetic_by_index[update_index] = synthetic
        replacement_refs[update.event_id] = synthetic.event_id

    migrated: List[InstructionEvent] = []
    for index, event in enumerate(events):
        if index in removed_indices:
            synthetic = synthetic_by_index.get(index)
            if synthetic is not None:
                migrated.append(synthetic)
            continue
        migrated.append(event)

    old_to_new: Dict[str, str] = {}
    for new_index, event in enumerate(migrated, start=1):
        old_id = event.event_id
        event.event_id = f"e{new_index}"
        old_to_new[old_id] = event.event_id
    for old_update_id, old_target_id in replacement_refs.items():
        if old_target_id in old_to_new:
            old_to_new[old_update_id] = old_to_new[old_target_id]

    for event in migrated:
        if event.event_type == EventType.PROGRESS and event.ref:
            event.ref = old_to_new.get(event.ref, "")
    return migrated


def normalize_instruction_plan(
    raw: Any,
    *,
    plan_id: str,
    turns: List[Dict[str, Any]],
) -> Tuple[InstructionPlan, List[str]]:
    """Tolerantly normalize a model candidate and return structural issues."""
    errors: List[str] = []
    if not isinstance(raw, dict):
        return InstructionPlan(plan_id=plan_id, turns=list(turns)), [
            "model output must be a JSON object"
        ]
    unexpected = set(raw) - {"entities", "events"}
    if unexpected:
        # Unknown top-level fields are intentionally ignored.
        pass
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
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, dict):
            errors.append(f"entity {index} must be an object")
            continue
        description = str(
            item.get("description") or item.get("text") or item.get("name") or ""
        ).strip()
        entity_id = _slug(
            str(item.get("id") or item.get("entity_id") or description),
            f"entity_{index}",
        )
        if not description:
            description = entity_id.replace("_", " ")
        candidate = entity_id
        suffix = 2
        while candidate in entities:
            candidate = f"{entity_id}_{suffix}"
            suffix += 1
        entity_id = candidate
        entity = Entity(
            entity_id=entity_id,
            description=description,
            kind=_entity_kind(item.get("kind") or item.get("geometry_type")),
            goal=item.get("goal") is True or str(item.get("role") or "").upper() == "GOAL",
        )
        entities[entity_id] = entity
        descriptions[description.casefold()] = entity_id

    events: List[InstructionEvent] = []
    raw_index_to_event_id: Dict[int, str] = {}
    turn_text = {
        int(item.get("turn") or 0): str(item.get("text") or "")
        for item in turns
        if isinstance(item, dict)
    }
    turn_roles = {
        int(item.get("turn") or 0): str(item.get("role") or "").strip().upper()
        for item in turns
        if isinstance(item, dict)
    }
    for index, item in enumerate(raw_events, start=1):
        if not isinstance(item, dict):
            errors.append(f"event {index} must be an object")
            continue
        turn_number = finite_float(item.get("turn") or item.get("turn_id"))
        if turn_number is None or turn_number < 1 or int(turn_number) != turn_number:
            errors.append(f"event {index} has invalid turn")
            continue
        # QUE turns remain in the numbered dialogue as context for later INS
        # answers, but they never materialize entities, progress, or actions.
        if turn_roles.get(int(turn_number)) == "QUE":
            continue
        raw_event_name = str(
            item.get("type") or item.get("event_type") or ""
        ).strip().upper().replace("-", "_")
        if raw_event_name in _IGNORED_EVENT_NAMES:
            continue
        event_type = _event_type(raw_event_name)
        if event_type is None:
            errors.append(f"event {index} has unknown type {item.get('type')!r}")
            continue
        direction = None
        if item.get("direction") not in (None, ""):
            try:
                direction = Direction.from_dict(item.get("direction"))
            except ValueError as exc:
                errors.append(f"event {index}: {exc}")
        mode_text = str(item.get("mode") or "FORWARD").strip().upper()
        try:
            travel_mode = TravelMode(mode_text)
        except ValueError:
            travel_mode = TravelMode.FORWARD

        raw_entity = str(
            item.get("entity") or item.get("entity_ref") or item.get("target") or ""
        ).strip()
        entity_ref = ""
        if raw_entity:
            slug = _slug(raw_entity, f"entity_{len(entities) + 1}")
            if slug in entities:
                entity_ref = slug
            elif raw_entity.casefold() in descriptions:
                entity_ref = descriptions[raw_entity.casefold()]
            else:
                entity_ref = slug
                candidate = entity_ref
                suffix = 2
                while candidate in entities:
                    candidate = f"{entity_ref}_{suffix}"
                    suffix += 1
                entity_ref = candidate
                entities[entity_ref] = Entity(
                    entity_id=entity_ref,
                    description=raw_entity,
                )
                descriptions[raw_entity.casefold()] = entity_ref

        count_value = finite_float(item.get("count"))
        count = max(1, int(count_value)) if count_value is not None else 1
        side = str(item.get("side") or "").strip().lower()
        if side not in {"", "left", "right", "front", "behind"}:
            side = ""
        event = InstructionEvent(
            event_id=f"e{len(events) + 1}",
            event_type=event_type,
            turn=int(turn_number),
            entity_ref=entity_ref,
            direction=direction,
            travel_mode=travel_mode,
            count=count,
            side=side,
            ref=(
                _resolve_ref(item.get("ref") or item.get("event_ref"))
                if event_type == EventType.PROGRESS
                else ""
            ),
            completed=(
                item.get("completed") is True
                if event_type == EventType.PROGRESS
                else False
            ),
            description=(
                str(item.get("description") or "").strip()
                if event_type == EventType.UPDATE_ENTITY
                else ""
            ),
            source_span=turn_text.get(int(turn_number), ""),
        )
        events.append(event)
        raw_index_to_event_id[index] = event.event_id

    # PROGRESS refs are indexes in the model's raw event array. Remap them after
    # dropping QUE/QUERY entries so a surviving update cannot complete the wrong
    # navigation event. A ref to a discarded event becomes an inert empty ref.
    for event in events:
        if event.event_type != EventType.PROGRESS or not event.ref:
            continue
        try:
            raw_ref_index = int(event.ref[1:])
        except (TypeError, ValueError):
            event.ref = ""
            continue
        event.ref = raw_index_to_event_id.get(raw_ref_index, "")

    events = _replace_update_entity_events(entities, events)

    visual_targets = [
        event
        for event in events
        if event.requires_visual_search
    ]
    # Trust an explicit goal flag even when the final relation is APPROACH or
    # another visual event.  Only infer a goal when the model supplied none.
    if visual_targets and not any(entity.goal for entity in entities.values()):
        entities[visual_targets[-1].entity_ref].goal = True

    plan = InstructionPlan(
        plan_id=str(plan_id or "").strip(),
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
            "mode": event.travel_mode.value,
            "count": event.count,
            "side": event.side,
            "source": event.source_span,
        }
