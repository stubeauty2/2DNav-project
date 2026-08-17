"""Same-heading execution windows and progress-confirmation contracts.

This module is intentionally independent of the visual-search implementation.
It turns one parsed INS turn into ordered, fixed-travel-heading windows and
provides the pure aggregation rules used by the windowed executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from instruction_schema import (
    DirectionFrame,
    EventType,
    InstructionEvent,
    InstructionPlan,
    NAVIGATION_EVENT_TYPES,
    TravelMode,
    normalize_angle,
)
from search_plan_adapter import HeadingState


HEADING_CHANGE_TOLERANCE_DEG = 5.0


def angular_distance(left: float, right: float) -> float:
    """Return the unsigned shortest difference between two headings."""
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def apply_event_heading(
    event: InstructionEvent,
    state: HeadingState,
) -> HeadingState:
    """Apply exactly one event's heading contract.

    Unlike the legacy adapter this function is never used to fold an entire
    INS turn.  The scheduler invokes it event-by-event and can therefore split
    before the event that changes travel direction.
    """
    body = normalize_angle(state.body_heading_deg)
    travel = normalize_angle(state.travel_heading_deg)
    if event.direction is None:
        resolved = travel
    elif event.direction.frame == DirectionFrame.ABSOLUTE:
        resolved = normalize_angle(event.direction.angle)
    else:
        resolved = normalize_angle(body + event.direction.angle)

    if event.event_type == EventType.TURN:
        body = resolved
        travel = resolved
    elif event.travel_mode == TravelMode.BACKWARD:
        travel = normalize_angle(resolved + 180.0)
    elif event.direction is not None:
        body = resolved
        travel = resolved
    return HeadingState(body, travel).normalized()


def _event_to_dict(event: InstructionEvent, description: str) -> Dict[str, Any]:
    result = {
        "event_id": event.event_id,
        "turn": event.turn,
        "type": event.event_type.value,
        "entity": event.entity_ref,
        "entity_description": description,
        "mode": event.travel_mode.value,
        "count": event.count,
        "side": event.side,
        "source_span": event.source_span,
    }
    if event.direction is not None:
        result["direction"] = event.direction.to_dict()
    return result


@dataclass(frozen=True)
class ExecutionWindow:
    window_id: str
    turn: int
    start_event_index: int
    end_event_index: int
    body_heading_before_deg: float
    body_heading_after_deg: float
    travel_heading_deg: float
    events: Tuple[InstructionEvent, ...]
    event_indices: Tuple[int, ...]
    event_descriptions: Dict[str, str]
    visual_event_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "turn": self.turn,
            "start_event_index": self.start_event_index,
            "end_event_index": self.end_event_index,
            "body_heading_before_deg": self.body_heading_before_deg,
            "body_heading_after_deg": self.body_heading_after_deg,
            "travel_heading_deg": self.travel_heading_deg,
            "event_ids": [event.event_id for event in self.events],
            "events": [
                _event_to_dict(
                    event,
                    self.event_descriptions.get(event.event_id, ""),
                )
                for event in self.events
            ],
            "visual_event_ids": list(self.visual_event_ids),
        }


@dataclass(frozen=True)
class INSSession:
    session_id: str
    turn: int
    instruction_text: str
    windows: Tuple[ExecutionWindow, ...]
    state_events: Tuple[InstructionEvent, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "instruction_text": self.instruction_text,
            "window_ids": [window.window_id for window in self.windows],
            "state_events": [
                {
                    "event_id": event.event_id,
                    "type": event.event_type.value,
                    "entity": event.entity_ref,
                    "description": event.description,
                    "ref": event.ref,
                    "completed": event.completed,
                }
                for event in self.state_events
            ],
        }


@dataclass
class TentativeTrace:
    trace_id: str
    event_ids: List[str]
    position: List[float]
    heading_deg: float
    view_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_ids": list(self.event_ids),
            "pose": {
                "lat": float(self.position[0]),
                "lng": float(self.position[1]),
                "heading_deg": float(self.heading_deg),
            },
            "view_ids": list(self.view_ids),
        }


@dataclass
class EventEvidence:
    event_id: str
    event_type: str
    candidate_id: str
    candidate_index: int
    found: bool
    destination_position: Optional[List[float]]
    bbox_2d: Optional[List[float]]
    confidence: float
    confirmation_present: bool
    confirmation_confidence: float
    evidence_view_ids: List[str]
    search_log_path: str
    attempts: List[Dict[str, Any]]
    progress_claim: str
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "found": self.found,
            "destination_position": self.destination_position,
            "bbox_2d": self.bbox_2d,
            "confidence": self.confidence,
            "confirmation_present": self.confirmation_present,
            "confirmation_confidence": self.confirmation_confidence,
            "evidence_view_ids": list(self.evidence_view_ids),
            "search_log_path": self.search_log_path,
            "attempts": list(self.attempts),
            "progress_claim": self.progress_claim,
            "explanation": self.explanation,
        }


@dataclass
class WindowProgressReport:
    window_id: str
    run_index: int
    run_status: str
    claimed_completed_event_ids: List[str]
    active_event_id: Optional[str]
    tentative_trace: List[TentativeTrace]
    event_evidence: List[EventEvidence]
    progress_summary: str
    stop_reason: str
    correction_applied: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "run_index": self.run_index,
            "run_status": self.run_status,
            "claimed_completed_event_ids": list(
                self.claimed_completed_event_ids
            ),
            "active_event_id": self.active_event_id,
            "tentative_trace": [item.to_dict() for item in self.tentative_trace],
            "event_evidence": [item.to_dict() for item in self.event_evidence],
            "progress_summary": self.progress_summary,
            "stop_reason": self.stop_reason,
            "correction_applied": self.correction_applied,
        }


@dataclass(frozen=True)
class EventCheck:
    event_id: str
    same_candidate: bool
    order_valid: bool
    motion_valid: bool
    visual_valid: bool
    result: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "same_candidate": self.same_candidate,
            "order_valid": self.order_valid,
            "motion_valid": self.motion_valid,
            "visual_valid": self.visual_valid,
            "result": self.result,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ConfirmationResult:
    window_id: str
    run_index: int
    verdict: str
    accepted_event_ids: Tuple[str, ...]
    confirmed_event_cursor: int
    commit_trace_id: Optional[str]
    event_checks: Tuple[EventCheck, ...]
    correction: str
    progress_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "run_index": self.run_index,
            "verdict": self.verdict,
            "accepted_event_ids": list(self.accepted_event_ids),
            "confirmed_event_cursor": self.confirmed_event_cursor,
            "commit_trace_id": self.commit_trace_id,
            "event_checks": [item.to_dict() for item in self.event_checks],
            "correction": self.correction,
            "progress_summary": self.progress_summary,
        }


def _ins_turns(plan: InstructionPlan) -> List[int]:
    turns = [
        int(item.get("turn") or 0)
        for item in plan.turns
        if isinstance(item, dict)
        and str(item.get("role") or "").strip().upper() == "INS"
        and int(item.get("turn") or 0) > 0
    ]
    return list(dict.fromkeys(turns)) if turns else sorted(
        {event.turn for event in plan.events}
    )


def build_ins_sessions(
    plan: InstructionPlan,
    starting_heading: HeadingState,
    *,
    heading_tolerance_deg: float = HEADING_CHANGE_TOLERANCE_DEG,
) -> Tuple[List[INSSession], HeadingState]:
    """Compile all INS turns while preserving chronological entity updates."""
    indexed_events = list(enumerate(plan.events))
    events_by_turn: Dict[int, List[Tuple[int, InstructionEvent]]] = {}
    for index, event in indexed_events:
        events_by_turn.setdefault(event.turn, []).append((index, event))
    descriptions = {
        entity_id: entity.description for entity_id, entity in plan.entities.items()
    }
    state = starting_heading.normalized()
    sessions: List[INSSession] = []

    for turn in _ins_turns(plan):
        windows: List[ExecutionWindow] = []
        state_events: List[InstructionEvent] = []
        current: List[Tuple[int, InstructionEvent, str]] = []
        window_body_before = state.body_heading_deg
        window_heading = state.travel_heading_deg
        window_number = 0

        def finish_window(body_after: float) -> None:
            nonlocal current, window_number
            if not current:
                return
            window_number += 1
            indices = tuple(item[0] for item in current)
            events = tuple(item[1] for item in current)
            event_descriptions = {
                item[1].event_id: item[2] for item in current
            }
            windows.append(ExecutionWindow(
                window_id=f"{plan.plan_id}_t{turn}_w{window_number}",
                turn=turn,
                start_event_index=indices[0],
                end_event_index=indices[-1],
                body_heading_before_deg=normalize_angle(window_body_before),
                body_heading_after_deg=normalize_angle(body_after),
                travel_heading_deg=normalize_angle(window_heading),
                events=events,
                event_indices=indices,
                event_descriptions=event_descriptions,
                visual_event_ids=tuple(
                    event.event_id
                    for event in events
                    if event.requires_visual_search
                ),
            ))
            current = []

        for event_index, event in events_by_turn.get(turn, []):
            if event.event_type == EventType.PROGRESS:
                state_events.append(event)
                continue
            if event.event_type not in NAVIGATION_EVENT_TYPES:
                continue

            state_before_event = state
            state_after_event = apply_event_heading(event, state)
            changes_heading = angular_distance(
                state_after_event.travel_heading_deg,
                window_heading,
            ) > float(heading_tolerance_deg)
            if current and changes_heading:
                finish_window(state_before_event.body_heading_deg)
            if not current:
                window_body_before = state_before_event.body_heading_deg
                window_heading = state_after_event.travel_heading_deg
            description = descriptions.get(
                event.entity_ref,
                event.entity_ref.replace("_", " ") if event.entity_ref else "",
            )
            current.append((event_index, event, description))
            state = state_after_event

        finish_window(state.body_heading_deg)
        sessions.append(INSSession(
            session_id=f"{plan.plan_id}_t{turn}",
            turn=turn,
            instruction_text=plan.source_for_turn(turn),
            windows=tuple(windows),
            state_events=tuple(state_events),
        ))
    return sessions, state.normalized()


def event_context_text(
    session: INSSession,
    window: ExecutionWindow,
    active_event: InstructionEvent,
    *,
    confirmed_event_ids: Sequence[str] = (),
    correction: str = "",
    excluded_candidates: Sequence[str] = (),
) -> str:
    """Build the event-aware text actually supplied to the visual search."""
    ordered = []
    for event in window.events:
        description = window.event_descriptions.get(event.event_id, "")
        phrase = f"{event.event_id}:{event.event_type.value}"
        if description:
            phrase += f"({description})"
        if event.side:
            phrase += f" side={event.side}"
        if event.count > 1:
            phrase += f" count={event.count}"
        if event.direction is not None:
            phrase += (
                f" direction={event.direction.frame.value}:"
                f"{event.direction.angle:.1f}deg"
            )
        if event.travel_mode != TravelMode.FORWARD:
            phrase += f" mode={event.travel_mode.value}"
        ordered.append(phrase)
    parts = [
        f"Instruction: {session.instruction_text}",
        f"Window travel heading: {window.travel_heading_deg:.1f} degrees.",
        "Ordered same-heading window events: " + " -> ".join(ordered),
        f"Active event: {active_event.event_id}:{active_event.event_type.value}",
        "Return a box only for the active event entity; other events are route context.",
    ]
    if confirmed_event_ids:
        parts.append("Confirmed prefix: " + ", ".join(confirmed_event_ids))
    if excluded_candidates:
        parts.append(
            "Previously used/rejected candidates (do not select again): "
            + "; ".join(excluded_candidates)
        )
    if correction:
        parts.append("Confirmation correction from the previous run: " + correction)
    return " ".join(parts)


def longest_contiguous_prefix(
    ordered_event_ids: Sequence[str],
    passed_event_ids: Iterable[str],
) -> List[str]:
    passed = set(passed_event_ids)
    prefix: List[str] = []
    for event_id in ordered_event_ids:
        if event_id not in passed:
            break
        prefix.append(event_id)
    return prefix


def aggregate_confirmation(
    window: ExecutionWindow,
    report: WindowProgressReport,
    checks: Sequence[EventCheck],
) -> ConfirmationResult:
    """Aggregate per-event checks into PASS/PARTIAL/REJECT deterministically."""
    ordered_ids = [event.event_id for event in window.events]
    claimed = list(report.claimed_completed_event_ids)
    claim_order_valid = claimed == ordered_ids[:len(claimed)]
    passed_ids = [
        check.event_id for check in checks
        if check.result == "PASS"
        and check.order_valid
        and check.motion_valid
        and check.visual_valid
        and check.same_candidate
    ]
    accepted = longest_contiguous_prefix(ordered_ids, passed_ids)
    if not claim_order_valid:
        accepted = []

    if accepted == ordered_ids and report.run_status == "COMPLETE":
        verdict = "PASS"
    elif accepted:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"

    commit_trace_id: Optional[str] = None
    accepted_set = set(accepted)
    for trace in report.tentative_trace:
        if trace.event_ids and set(trace.event_ids).issubset(accepted_set):
            commit_trace_id = trace.trace_id

    confirmed_cursor = window.start_event_index
    if accepted:
        index_by_id = {
            event.event_id: event_index
            for event, event_index in zip(window.events, window.event_indices)
        }
        confirmed_cursor = index_by_id[accepted[-1]] + 1

    failed = [check for check in checks if check.result != "PASS"]
    if verdict == "PASS":
        correction = ""
        summary = (
            f"Confirmed all {len(accepted)} events in {window.window_id}; "
            "event order, motion contract, and visual evidence all passed."
        )
    elif verdict == "PARTIAL":
        first_failed = failed[0].event_id if failed else ordered_ids[len(accepted)]
        correction = f"Resume from {first_failed}; do not repeat the confirmed prefix."
        summary = (
            f"Confirmed {len(accepted)} leading events; "
            f"the remaining window requires another run. First issue: "
            f"{failed[0].explanation if failed else 'unfinished event'}."
        )
    else:
        reason = failed[0].explanation if failed else "reported progress is not contiguous"
        correction = f"Restart this window from its confirmed pose: {reason}"
        summary = (
            "No event progress was accepted for this window run. "
            f"First issue: {reason}."
        )

    return ConfirmationResult(
        window_id=window.window_id,
        run_index=report.run_index,
        verdict=verdict,
        accepted_event_ids=tuple(accepted),
        confirmed_event_cursor=confirmed_cursor,
        commit_trace_id=commit_trace_id,
        event_checks=tuple(checks),
        correction=correction,
        progress_summary=summary,
    )
