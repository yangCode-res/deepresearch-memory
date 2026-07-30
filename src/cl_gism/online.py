"""Online cross-loop memory for an active deep-research trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .llm_loop import LLMLoopBoundaryJudge, LoopBoundaryDecision
from .llm_state_update import LLMStateUpdater
from .retrieval import LexicalMemoryIndex, MemoryHit, build_retrieval_query
from .schema import LoopMemory, MemoryStatus, RawMemory, SourceType, TaskAnchor, utc_now
from .trajectory import TrajectoryEvent
from .unified_controller import UnifiedMemoryController


def _message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    calls = message.get("tool_calls")
    if calls:
        parts.append(f"[tool_calls]\n{json.dumps(calls, ensure_ascii=False)}")
    return "\n".join(parts).strip()


def _source_type(role: str) -> SourceType:
    return {
        "user": SourceType.USER,
        "assistant": SourceType.AGENT,
        "tool": SourceType.TOOL,
        "system": SourceType.SYSTEM,
    }.get(role, SourceType.SYSTEM)


def _compact_values(items: list[Any], limit: int = 6) -> list[str]:
    values: list[str] = []
    for item in items[-limit:]:
        value = str(getattr(item, "value", item)).strip()
        if value:
            values.append(value[:800])
    return values


@dataclass
class OnlineMemoryTrace:
    round_number: int
    canonical_message_count: int
    prompt_message_count: int
    current_loop_message_count: int
    state_version: int
    retrieval_query: str
    retrieved_memory_ids: list[str] = field(default_factory=list)
    loop_switched: bool = False
    loop_reason: str = ""
    controller_error: str | None = None


class OnlineMemorySession:
    """Keep full audit history while constructing compact per-round prompts."""

    def __init__(
        self,
        *,
        qid: Any,
        question: str,
        system_prompt: str,
        boundary_judge: LLMLoopBoundaryJudge,
        state_updater: LLMStateUpdater,
        unified_controller: UnifiedMemoryController | None = None,
        top_k: int = 4,
        memory_text_limit: int = 1800,
    ) -> None:
        self.qid = qid
        self.task_id = f"task_openresearcher_{qid}"
        self.question = question
        self.system_prompt = system_prompt
        self.boundary_judge = boundary_judge
        self.state_updater = state_updater
        self.unified_controller = unified_controller
        self.top_k = top_k
        self.memory_text_limit = memory_text_limit
        self.anchor = TaskAnchor(
            task_id=self.task_id,
            original_goal=question,
            success_criteria=["produce an evidence-backed exact answer with confidence"],
            immutable_constraints=["preserve citations and never invent tool evidence"],
            domain="deep_research",
        )
        self.state = state_updater.initialize(self.anchor)
        self.index = LexicalMemoryIndex()
        self.current_events: list[TrajectoryEvent] = []
        self.completed_loops: list[LoopMemory] = []
        self.deltas: list[dict[str, Any]] = []
        self.traces: list[OnlineMemoryTrace] = []
        self._sequence = 0
        self._known_message_count = 2
        self._pending_switch = LoopBoundaryDecision(False, "initial loop", 1.0)
        self._controller_query = ""
        self._controller_selected_ids: list[str] = []
        self._controller_selected_hits: list[MemoryHit] = []
        self._controller_succeeded = False

    def _event(self, message: dict[str, Any]) -> TrajectoryEvent:
        self._sequence += 1
        role = str(message.get("role") or "unknown")
        text = _message_text(message)
        raw = RawMemory(
            task_id=self.task_id,
            source_type=_source_type(role),
            content={"sequence": self._sequence, "role": role, "text": text},
            content_type="application/json",
            metadata={"qid": self.qid, "sequence": self._sequence, "role": role},
        )
        return TrajectoryEvent(
            sequence=self._sequence,
            role=role,
            text=text,
            raw_memory=raw,
            message=message,
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
        )

    def _materialize_current_loop(self, decision: LoopBoundaryDecision | None = None) -> LoopMemory | None:
        if not self.current_events:
            return None
        assistants = [event for event in self.current_events if event.role == "assistant" and event.text]
        tools = [event for event in self.current_events if event.role == "tool" and event.text]
        first = assistants[0].text if assistants else "research step"
        subgoal = (
            (decision.current_loop_subgoal if decision else "").strip()
            or first.splitlines()[0][:240]
            or "research step"
        )
        conclusion = assistants[-1].text[:2400] if assistants else tools[-1].text[:2400] if tools else None
        return LoopMemory(
            task_id=self.task_id,
            subgoal=subgoal,
            context=decision.reason if decision else "finalized at task completion",
            actions=[{"sequence": event.sequence, "text": event.text} for event in assistants],
            observations=[{"sequence": event.sequence, "text": event.text} for event in tools],
            conclusion=conclusion,
            evidence_ids=[event.raw_memory.raw_id for event in self.current_events],
            started_at=self.current_events[0].raw_memory.occurred_at,
            ended_at=utc_now(),
            status=MemoryStatus.RESOLVED,
        )

    def _archive_current_loop(
        self,
        decision: LoopBoundaryDecision | None = None,
        planned_state_delta: dict[str, Any] | None = None,
    ) -> None:
        loop = self._materialize_current_loop(decision)
        if loop is None:
            return
        try:
            if planned_state_delta and planned_state_delta.get("operations"):
                update = self.state_updater.apply_result(
                    self.anchor, self.state, loop, planned_state_delta
                )
            else:
                update = self.state_updater.update(self.anchor, self.state, loop)
            self.state = update.state
            loop.state_delta_ids.append(update.delta.delta_id)
            self.deltas.append(update.delta.to_dict())
        except Exception as exc:
            self._pending_switch.controller_error = f"state_update:{exc.__class__.__name__}"
        for event in self.current_events:
            self.index.add_raw(event.raw_memory)
        self.index.add_loop(loop)
        self.completed_loops.append(loop)
        self.current_events = []

    def ingest_new_messages(self, canonical_messages: list[dict[str, Any]]) -> None:
        new_messages = canonical_messages[self._known_message_count :]
        self._known_message_count = len(canonical_messages)
        self._pending_switch = LoopBoundaryDecision(False, "same loop", 0.5)
        self._controller_succeeded = False
        self._controller_query = ""
        self._controller_selected_ids = []
        self._controller_selected_hits = []
        if self.unified_controller and new_messages:
            latest_events = [self._event(message) for message in new_messages]
            latest_text = "\n".join(event.text for event in latest_events)
            seed_query = build_retrieval_query(self.anchor, self.state, latest_text)
            candidates = self.index.search(seed_query, top_k=12, task_id=self.task_id)
            try:
                control = self.unified_controller.decide(
                    anchor=self.anchor,
                    state=self.state,
                    current_loop=self.current_events,
                    latest_events=latest_events,
                    candidates=candidates,
                )
                decision = LoopBoundaryDecision(
                    split=control.switch_loop,
                    reason=control.reason,
                    confidence=control.confidence,
                    current_loop_subgoal=control.current_loop_subgoal,
                    next_loop_subgoal=control.next_loop_subgoal,
                )
                self._pending_switch = decision
                self._controller_query = control.retrieval_query
                self._controller_selected_ids = control.selected_memory_ids
                by_id = {hit.memory_id: hit for hit in candidates}
                self._controller_selected_hits = [
                    by_id[memory_id]
                    for memory_id in control.selected_memory_ids
                    if memory_id in by_id
                ]
                self._controller_succeeded = True
                if decision.split and self.current_events:
                    self._archive_current_loop(decision, control.state_delta)
            except Exception as exc:
                self._pending_switch = LoopBoundaryDecision(False, "controller fallback", 0.0)
                self._pending_switch.controller_error = f"unified_controller:{exc.__class__.__name__}"
            self.current_events.extend(latest_events)
            return
        for message in new_messages:
            event = self._event(message)
            if event.role == "assistant" and self.current_events:
                try:
                    decision = self.boundary_judge.judge(self.anchor, self.current_events, event)
                    self._pending_switch = decision
                except Exception as exc:
                    decision = LoopBoundaryDecision(False, "controller fallback", 0.0)
                    self._pending_switch = decision
                    self._pending_switch.controller_error = f"loop_judge:{exc.__class__.__name__}"
                if decision.split:
                    self._archive_current_loop(decision)
            self.current_events.append(event)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "version": self.state.state_version,
            "current_goal": self.state.current_goal.value if self.state.current_goal else self.question,
            "active_subgoals": _compact_values(self.state.active_subgoals),
            "working_hypotheses": _compact_values(self.state.working_hypotheses),
            "resolved_findings": _compact_values(self.state.resolved_findings),
            "open_questions": _compact_values(self.state.open_questions),
            "uncertainties": _compact_values(self.state.uncertainties),
            "next_actions": _compact_values(self.state.next_actions),
        }

    def _memory_block(self, query: str, hits: list[MemoryHit]) -> str:
        memories = [
            {
                "id": hit.memory_id,
                "type": hit.memory_type,
                "score": round(hit.score, 4),
                "text": hit.text[: self.memory_text_limit],
            }
            for hit in hits
        ]
        payload = {
            "task_anchor": {
                "goal": self.anchor.original_goal,
                "success_criteria": self.anchor.success_criteria,
                "immutable_constraints": self.anchor.immutable_constraints,
            },
            "global_intent_state": self._state_payload(),
            "retrieval_query": query,
            "retrieved_cross_loop_memories": memories,
        }
        return (
            "\n\n<cross_loop_memory>\n"
            "This is compact state and retrieved prior-loop evidence. Treat it as memory, not as new tool output.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n</cross_loop_memory>"
        )

    def build_prompt(self, canonical_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.ingest_new_messages(canonical_messages)
        recent_event = self.current_events[-1].text if self.current_events else "start research"
        query = self._controller_query or build_retrieval_query(self.anchor, self.state, recent_event)
        candidates = self.index.search(query, top_k=max(self.top_k, 12), task_id=self.task_id)
        if self._controller_succeeded:
            hits = self._controller_selected_hits
        else:
            hits = candidates[: self.top_k]
        system = dict(canonical_messages[0])
        system["content"] = str(system.get("content") or "") + self._memory_block(query, hits)
        prompt_messages = [system, dict(canonical_messages[1])]
        prompt_messages.extend(dict(event.message) for event in self.current_events)
        self.traces.append(
            OnlineMemoryTrace(
                round_number=len(self.traces) + 1,
                canonical_message_count=len(canonical_messages),
                prompt_message_count=len(prompt_messages),
                current_loop_message_count=len(self.current_events),
                state_version=self.state.state_version,
                retrieval_query=query,
                retrieved_memory_ids=[hit.memory_id for hit in hits],
                loop_switched=self._pending_switch.split,
                loop_reason=self._pending_switch.reason,
                controller_error=getattr(self._pending_switch, "controller_error", None),
            )
        )
        return prompt_messages

    def finalize(self, canonical_messages: list[dict[str, Any]]) -> None:
        self.ingest_new_messages(canonical_messages)
        self._archive_current_loop(None)

    def trace_payload(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "state": self.state.to_dict(),
            "completed_loops": [loop.to_dict() for loop in self.completed_loops],
            "state_deltas": self.deltas,
            "rounds": [trace.__dict__ for trace in self.traces],
        }


__all__ = ["OnlineMemorySession", "OnlineMemoryTrace"]
