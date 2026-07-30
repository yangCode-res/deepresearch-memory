"""Adapters from OpenResearcher trajectory records to CL-GISM primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from .schema import (
    LoopMemory,
    MemoryStatus,
    RawMemory,
    SourceType,
    TaskAnchor,
    new_id,
    utc_now,
)


def _message_text(message: dict[str, Any]) -> str:
    """Extract visible text while retaining structured tool-call information."""

    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif item:
                parts.append(str(item))
    elif content is not None:
        parts.append(str(content))

    if message.get("reasoning_content"):
        parts.append(f"[reasoning]\n{message['reasoning_content']}")
    if message.get("tool_calls"):
        parts.append(f"[tool_calls]\n{json.dumps(message['tool_calls'], ensure_ascii=False)}")
    return "\n".join(part for part in parts if part).strip()


def _source_type(role: str | None) -> SourceType:
    return {
        "user": SourceType.USER,
        "assistant": SourceType.AGENT,
        "tool": SourceType.TOOL,
        "system": SourceType.SYSTEM,
        "developer": SourceType.SYSTEM,
    }.get(role or "", SourceType.SYSTEM)


@dataclass
class TrajectoryEvent:
    """A normalized event used by the Loop builder and retrieval layer."""

    sequence: int
    role: str
    text: str
    raw_memory: RawMemory
    message: dict[str, Any]
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class ParsedTrajectory:
    task_id: str
    anchor: TaskAnchor
    events: list[TrajectoryEvent]
    loops: list[LoopMemory] = field(default_factory=list)
    top_level_answer: str | None = None
    status: str | None = None


def parse_openresearcher_row(row: dict[str, Any]) -> ParsedTrajectory:
    """Convert one OpenResearcher row into Task Anchor and Raw Memory events.

    The adapter accepts the JSON shape used by the downloaded examples and by
    the dataset exporter.  It deliberately keeps each original message in
    ``RawMemory.metadata['raw_message']`` so the parser is lossless.
    """

    qid = row.get("qid", new_id("task"))
    task_id = f"task_openresearcher_{qid}"
    messages = row.get("messages") or []
    user_index = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    question = str(row.get("question") or "")
    if not question and user_index is not None:
        question = _message_text(messages[user_index])
    if not question:
        raise ValueError("OpenResearcher row must contain question or a user message")

    events: list[TrajectoryEvent] = []
    first_user_raw_id: str | None = None
    for sequence, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        text = _message_text(message)
        raw = RawMemory(
            task_id=task_id,
            source_type=_source_type(role),
            content={
                "sequence": sequence,
                "role": role,
                "text": text,
                "raw_message": message,
            },
            content_type="application/json",
            metadata={"qid": qid, "role": role, "sequence": sequence},
            parent_loop_id=None,
        )
        if role == "user" and first_user_raw_id is None:
            first_user_raw_id = raw.raw_id
        events.append(
            TrajectoryEvent(
                sequence=sequence,
                role=role,
                text=text,
                raw_memory=raw,
                message=message,
                name=message.get("name"),
                tool_call_id=message.get("tool_call_id"),
            )
        )

    anchor = TaskAnchor(
        task_id=task_id,
        original_goal=question,
        success_criteria=["produce an evidence-backed answer"],
        immutable_constraints=["preserve the original trajectory and evidence"],
        domain="deep_research",
        evidence_ids=[first_user_raw_id] if first_user_raw_id else [],
    )
    return ParsedTrajectory(
        task_id=task_id,
        anchor=anchor,
        events=events,
        top_level_answer=row.get("answer"),
        status=row.get("status"),
    )


class RuleBasedLoopBuilder:
    """Build candidate Loops from assistant/tool interaction boundaries.

    A Loop closes after a tool observation is followed by an assistant
    interpretation, or at the end of the trajectory.  This is intentionally a
    transparent MVP heuristic; a learned boundary detector can replace it.
    """

    def build(self, trajectory: ParsedTrajectory) -> list[LoopMemory]:
        research_events = [e for e in trajectory.events if e.role not in {"system", "developer", "user"}]
        if not research_events:
            return []

        loops: list[LoopMemory] = []
        current: list[TrajectoryEvent] = []
        saw_tool = False

        def close_loop(events: list[TrajectoryEvent]) -> None:
            if not events:
                return
            assistants = [e for e in events if e.role == "assistant" and e.text]
            tools = [e for e in events if e.role == "tool"]
            first_assistant = assistants[0].text if assistants else "research step"
            subgoal = first_assistant.splitlines()[0].strip()[:240] or "research step"
            actions = [
                {"sequence": e.sequence, "text": e.text, "role": e.role}
                for e in events
                if e.role == "assistant"
            ]
            observations = [
                {"sequence": e.sequence, "text": e.text, "role": e.role}
                for e in tools
            ]
            conclusion = assistants[-1].text[:2000] if assistants else None
            loop = LoopMemory(
                task_id=trajectory.task_id,
                subgoal=subgoal,
                actions=actions,
                observations=observations,
                conclusion=conclusion,
                evidence_ids=[e.raw_memory.raw_id for e in events],
                started_at=utc_now(),
                ended_at=utc_now(),
                status=MemoryStatus.RESOLVED,
            )
            loops.append(loop)

        for event in research_events:
            current.append(event)
            if event.role == "tool":
                saw_tool = True
            elif event.role == "assistant" and saw_tool:
                close_loop(current)
                current = []
                saw_tool = False
        close_loop(current)
        trajectory.loops = loops
        return loops


__all__ = [
    "ParsedTrajectory",
    "RuleBasedLoopBuilder",
    "TrajectoryEvent",
    "parse_openresearcher_row",
]
