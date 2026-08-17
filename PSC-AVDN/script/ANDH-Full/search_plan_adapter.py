"""Project a flat :class:`InstructionPlan` onto baseline search steps.

The adapter is deliberately deterministic.  It does not reinterpret language,
call a model, or reproduce the event runtime.  At most one executable search
step is emitted for each INS turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from instruction_schema import (
    DirectionFrame,
    EventType,
    InstructionEvent,
    InstructionPlan,
    NAVIGATION_EVENT_TYPES,
    TravelMode,
    normalize_angle,
)


TERMINAL_TARGET_TYPES = frozenset({
    EventType.REACH,
    EventType.STOP_AT,
    EventType.APPROACH,
    EventType.ENTER,
    EventType.EXIT,
    EventType.FOLLOW,
})
WAYPOINT_TARGET_TYPES = frozenset({
    EventType.PASS,
    EventType.CROSS,
    EventType.GO_THROUGH,
})


@dataclass(frozen=True)
class HeadingOperation:
    event_id: str
    event_type: str
    frame: str = ""
    angle: Optional[float] = None
    mode: str = TravelMode.FORWARD.value

    @classmethod
    def from_event(cls, event: InstructionEvent) -> "HeadingOperation":
        return cls(
            event_id=event.event_id,
            event_type=event.event_type.value,
            frame=(event.direction.frame.value if event.direction else ""),
            angle=(float(event.direction.angle) if event.direction else None),
            mode=event.travel_mode.value,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "mode": self.mode,
        }
        if self.frame:
            result["frame"] = self.frame
        if self.angle is not None:
            result["angle"] = self.angle
        return result


@dataclass(frozen=True)
class ViaTarget:
    event_id: str
    event_type: str
    entity: str
    description: str
    count: int = 1
    side: str = ""

    def phrase(self) -> str:
        parts = [self.event_type.lower()]
        if self.count > 1:
            parts.append(str(self.count))
        if self.side:
            parts.append(self.side)
        parts.append(self.description or self.entity)
        return " ".join(item for item in parts if item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "entity": self.entity,
            "description": self.description,
            "count": self.count,
            "side": self.side,
        }


@dataclass(frozen=True)
class SearchStep:
    step_index: int
    turn: int
    source_text: str
    source_event_ids: Sequence[str]
    source_event_types: Sequence[str]
    heading_operations: Sequence[HeadingOperation]
    destination_entity: str = ""
    destination_description: str = ""
    destination_event_type: str = ""
    grounding_query: str = ""
    via: Sequence[ViaTarget] = field(default_factory=tuple)
    motion_only: bool = False
    forward: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "turn": self.turn,
            "source_text": self.source_text,
            "source_event_ids": list(self.source_event_ids),
            "source_event_types": list(self.source_event_types),
            "heading_operations": [item.to_dict() for item in self.heading_operations],
            "destination_entity": self.destination_entity,
            "destination_description": self.destination_description,
            "destination_event_type": self.destination_event_type,
            "grounding_query": self.grounding_query,
            "via": [item.to_dict() for item in self.via],
            "motion_only": self.motion_only,
            "forward": self.forward,
        }


@dataclass(frozen=True)
class HeadingState:
    body_heading_deg: float
    travel_heading_deg: float

    def normalized(self) -> "HeadingState":
        return HeadingState(
            normalize_angle(self.body_heading_deg),
            normalize_angle(self.travel_heading_deg),
        )


def _target_snapshot(
    event: InstructionEvent,
    descriptions: Dict[str, str],
) -> ViaTarget:
    return ViaTarget(
        event_id=event.event_id,
        event_type=event.event_type.value,
        entity=event.entity_ref,
        description=(descriptions.get(event.entity_ref) or event.entity_ref.replace("_", " ")),
        count=event.count,
        side=event.side,
    )


def _grounding_query(destination: ViaTarget, via: Sequence[ViaTarget]) -> str:
    query = f"Final target to localize: {destination.description}."
    if via:
        context = "; ".join(item.phrase() for item in via)
        query += (
            " Route context only (do not return a bounding box for these "
            f"waypoints): {context}."
        )
    return query


def _ins_turns(plan: InstructionPlan) -> List[int]:
    turns = [
        int(item.get("turn") or 0)
        for item in plan.turns
        if isinstance(item, dict)
        and str(item.get("role") or "").strip().upper() == "INS"
        and int(item.get("turn") or 0) > 0
    ]
    if turns:
        return list(dict.fromkeys(turns))
    return sorted({event.turn for event in plan.events})


def project_instruction_plan(plan: InstructionPlan) -> List[SearchStep]:
    """Return at most one baseline-compatible step for every INS turn."""
    descriptions = {
        entity_id: entity.description
        for entity_id, entity in plan.entities.items()
    }
    text_by_turn = {
        int(item.get("turn") or 0): str(item.get("text") or "")
        for item in plan.turns
        if isinstance(item, dict)
    }
    events_by_turn: Dict[int, List[InstructionEvent]] = {}
    for event in plan.events:
        events_by_turn.setdefault(event.turn, []).append(event)

    steps: List[SearchStep] = []
    for turn in _ins_turns(plan):
        events = events_by_turn.get(turn, [])
        navigation: List[InstructionEvent] = []
        targets: List[ViaTarget] = []
        for event in events:
            if event.event_type == EventType.PROGRESS:
                continue
            if event.event_type not in NAVIGATION_EVENT_TYPES:
                continue
            navigation.append(event)
            if event.entity_ref:
                targets.append(_target_snapshot(event, descriptions))

        # PROGRESS-only answers do not become visual-search or motion steps.
        if not navigation:
            continue

        explicit = [
            item for item in targets
            if EventType(item.event_type) in TERMINAL_TARGET_TYPES
        ]
        waypoint = [
            item for item in targets
            if EventType(item.event_type) in WAYPOINT_TARGET_TYPES
        ]
        destination = explicit[-1] if explicit else (waypoint[-1] if waypoint else None)
        via = tuple(item for item in targets if item is not destination)
        motion_only = destination is None
        steps.append(SearchStep(
            step_index=len(steps) + 1,
            turn=turn,
            source_text=text_by_turn.get(turn, navigation[0].source_span if navigation else ""),
            source_event_ids=tuple(event.event_id for event in events),
            source_event_types=tuple(event.event_type.value for event in events),
            heading_operations=tuple(HeadingOperation.from_event(event) for event in navigation),
            destination_entity=(destination.entity if destination else ""),
            destination_description=(destination.description if destination else ""),
            destination_event_type=(destination.event_type if destination else ""),
            grounding_query=(_grounding_query(destination, via) if destination else ""),
            via=via,
            motion_only=motion_only,
            forward=any(event.event_type == EventType.MOVE for event in navigation),
        ))
    return steps


def resolve_step_heading(step: SearchStep, state: HeadingState) -> HeadingState:
    """Fold the plan's direction contract into one baseline travel heading."""
    body = normalize_angle(state.body_heading_deg)
    travel = normalize_angle(state.travel_heading_deg)
    for operation in step.heading_operations:
        if operation.angle is None:
            resolved = travel
        elif operation.frame == DirectionFrame.ABSOLUTE.value:
            resolved = normalize_angle(operation.angle)
        else:
            resolved = normalize_angle(body + operation.angle)

        if operation.event_type == EventType.TURN.value:
            body = resolved
            travel = resolved
        elif operation.mode == TravelMode.BACKWARD.value:
            travel = normalize_angle(resolved + 180.0)
        elif operation.angle is not None:
            body = resolved
            travel = resolved
    return HeadingState(body, travel).normalized()
