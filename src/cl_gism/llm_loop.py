"""Optional LLM-backed loop boundary detection for CL-GISM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .llm_planner import OpenAIChatJSONClient
from .schema import LoopMemory, MemoryStatus, TaskAnchor
from .trajectory import ParsedTrajectory, RuleBasedLoopBuilder, TrajectoryEvent


def _truncate(text: str, limit: int = 400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _event_payload(event: TrajectoryEvent) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "role": event.role,
        "name": event.name,
        "tool_call_id": event.tool_call_id,
        "text": _truncate(event.text),
    }


def _loop_payload(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
    return [_event_payload(event) for event in events]


def _extract_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "1", "split"}
    return bool(raw)


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


@dataclass
class LoopBoundaryDecision:
    split: bool
    reason: str
    confidence: float = 0.5
    current_loop_subgoal: str = ""
    next_loop_subgoal: str = ""


class LLMLoopBoundaryJudge:
    """Ask a model whether the next event should start a new loop."""

    def __init__(self, client: OpenAIChatJSONClient | None) -> None:
        self.client = client

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "LLMLoopBoundaryJudge" | None:
        client = OpenAIChatJSONClient.from_env(model=model, base_url=base_url)
        if not client:
            return None
        return cls(client)

    def judge(
        self,
        anchor: TaskAnchor,
        current_loop_events: list[TrajectoryEvent],
        candidate_next_event: TrajectoryEvent,
    ) -> LoopBoundaryDecision:
        if not self.client:
            return LoopBoundaryDecision(split=False, reason="no client available", confidence=0.0)

        system_prompt = (
            "You segment deep-research trajectories into coherent loops.\n"
            "You will see the current loop messages and one candidate next event.\n"
            "Decide whether the candidate should continue the current loop or start a new one.\n"
            "Return only valid JSON.\n"
            "Be conservative: split only when the candidate clearly begins a new subtask, new search thread, "
            "new evidence phase, or the prior loop has reached a conclusion and the candidate starts fresh.\n"
            "Do not split merely because the role changes from assistant to tool or tool to assistant."
        )
        user_payload = {
            "task_id": anchor.task_id,
            "goal": anchor.original_goal,
            "current_loop_messages": _loop_payload(current_loop_events),
            "candidate_next_event": _event_payload(candidate_next_event),
            "allowed_output": {
                "split": True,
                "reason": "short explanation",
                "confidence": 0.75,
                "current_loop_subgoal": "short label for the current loop",
                "next_loop_subgoal": "short label for the new loop, if split is true",
            },
            "rules": [
                "If the candidate is a direct follow-up in the same search or reasoning thread, keep it in the current loop.",
                "If the candidate begins a fresh question, fresh evidence chase, or fresh answer synthesis, split before it.",
                "If you are unsure, prefer not to split.",
                "Return split=false when the candidate should be appended to the current loop.",
            ],
        }
        raw = self.client.complete_json(system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2))
        return LoopBoundaryDecision(
            split=_extract_bool(raw.get("split")),
            reason=str(raw.get("reason") or "").strip(),
            confidence=_clamp_confidence(raw.get("confidence", 0.5)),
            current_loop_subgoal=str(raw.get("current_loop_subgoal") or "").strip(),
            next_loop_subgoal=str(raw.get("next_loop_subgoal") or "").strip(),
        )


class LLMLoopBuilder:
    """Build loops by asking a model whether each incoming event starts a new one."""

    def __init__(
        self,
        judge: LLMLoopBoundaryJudge | None,
        *,
        fallback: RuleBasedLoopBuilder | None = None,
    ) -> None:
        self.judge = judge
        self.fallback = fallback or RuleBasedLoopBuilder()

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "LLMLoopBuilder" | None:
        judge = LLMLoopBoundaryJudge.from_env(model=model, base_url=base_url)
        if not judge:
            return None
        return cls(judge)

    def _materialize_loop(self, task_id: str, events: list[TrajectoryEvent], decision: LoopBoundaryDecision | None) -> LoopMemory:
        assistants = [event for event in events if event.role == "assistant" and event.text]
        tools = [event for event in events if event.role == "tool"]
        first_assistant = assistants[0].text if assistants else "research step"
        subgoal = (
            (decision.current_loop_subgoal if decision and decision.current_loop_subgoal else "").strip()
            or first_assistant.splitlines()[0].strip()[:240]
            or "research step"
        )
        actions = [{"sequence": event.sequence, "text": event.text, "role": event.role} for event in assistants]
        observations = [{"sequence": event.sequence, "text": event.text, "role": event.role} for event in tools]
        conclusion = assistants[-1].text[:2000] if assistants else None
        context = decision.reason if decision else None
        return LoopMemory(
            task_id=task_id,
            subgoal=subgoal,
            context=context,
            actions=actions,
            observations=observations,
            conclusion=conclusion,
            evidence_ids=[event.raw_memory.raw_id for event in events],
            started_at=events[0].raw_memory.occurred_at,
            ended_at=events[-1].raw_memory.occurred_at,
            status=MemoryStatus.RESOLVED,
        )

    def build(self, trajectory: ParsedTrajectory) -> list[LoopMemory]:
        research_events = [e for e in trajectory.events if e.role not in {"system", "developer", "user"}]
        if not research_events:
            return []
        if not self.judge:
            loops = self.fallback.build(trajectory)
            trajectory.loops = loops
            return loops

        loops: list[LoopMemory] = []
        current: list[TrajectoryEvent] = [research_events[0]]

        for event in research_events[1:]:
            try:
                decision = self.judge.judge(trajectory.anchor, current, event)
            except Exception:  # pragma: no cover - network and parse fallback path
                loops = self.fallback.build(trajectory)
                trajectory.loops = loops
                return loops

            if decision.split:
                loops.append(self._materialize_loop(trajectory.task_id, current, decision))
                current = [event]
            else:
                current.append(event)

        loops.append(self._materialize_loop(trajectory.task_id, current, None))
        trajectory.loops = loops
        return loops


__all__ = [
    "LLMLoopBoundaryJudge",
    "LLMLoopBuilder",
    "LoopBoundaryDecision",
]
