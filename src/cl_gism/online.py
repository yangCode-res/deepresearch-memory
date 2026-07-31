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
from .unified_controller import UnifiedMemoryController, empty_loop_progress


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


def _compact_state_items(items: list[Any], limit: int = 6) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in items[-limit:]:
        value = str(getattr(item, "value", item)).strip()
        if not value:
            continue
        status = getattr(getattr(item, "status", None), "value", None)
        compacted.append(
            {
                "value": value[:800],
                "status": status,
                "confidence": getattr(item, "confidence", None),
            }
        )
    return compacted


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
    task_status: str = "CONTINUE"
    research_phase: str = "DISCOVERY"
    current_loop_subgoal: str = ""
    next_loop_subgoal: str = ""
    loop_outcome: str = "IN_PROGRESS"
    boundary_basis: str = "NONE"
    loop_progress: dict[str, Any] = field(default_factory=empty_loop_progress)
    loop_rounds: int = 0
    stagnant_rounds: int = 0
    controller_error: str | None = None
    controller_validation_retries: int = 0
    controller_response: dict[str, Any] | None = None


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
        self._last_control = None
        self._task_status = "CONTINUE"
        self._research_phase = "DISCOVERY"
        self._current_loop_subgoal = ""
        self._loop_progress = empty_loop_progress()
        self._loop_rounds = 0
        self._stagnant_rounds = 0
        self._just_archived_memory_ids: list[str] = []

    def _evidence_memory_candidates(self, query: str, limit: int = 12) -> list[MemoryHit]:
        """Recall evidence-bearing memories, excluding assistant narration.

        Lexical overlap in an agent's old search plan is especially prone to
        reinforcing a wrong direction. Completed-loop summaries and raw tool
        observations retain the evidence while avoiding that feedback loop.
        """

        broad_pool = self.index.search(
            query,
            top_k=max(limit * 3, limit),
            task_id=self.task_id,
        )
        return [
            hit for hit in broad_pool
            if hit.memory_type == "loop" or hit.metadata.get("source_type") == "tool"
        ][:limit]

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

    def _materialize_current_loop(
        self,
        decision: LoopBoundaryDecision | None = None,
        planned_state_delta: dict[str, Any] | None = None,
    ) -> LoopMemory | None:
        if not self.current_events:
            return None
        assistants = [event for event in self.current_events if event.role == "assistant" and event.text]
        tools = [event for event in self.current_events if event.role == "tool" and event.text]
        first = assistants[0].text if assistants else "research step"
        subgoal = (
            (decision.current_loop_subgoal if decision else "").strip()
            or self._current_loop_subgoal.strip()
            or first.splitlines()[0][:240]
            or "research step"
        )
        durable_parts: list[str] = []
        if planned_state_delta:
            summary = str(planned_state_delta.get("summary") or "").strip()
            if summary:
                durable_parts.append(summary)
            for operation in planned_state_delta.get("operations") or []:
                if not isinstance(operation, dict):
                    continue
                target = str(operation.get("target") or "finding")
                value = str(operation.get("value") or "").strip()
                if value:
                    durable_parts.append(f"{target}: {value}")
            progress_summary = str(self._loop_progress.get("progress_summary") or "").strip()
            if progress_summary:
                durable_parts.append(f"loop_progress: {progress_summary}")
            for evidence in self._loop_progress.get("key_evidence") or []:
                durable_parts.append(f"key_evidence: {evidence}")
        conclusion = "\n".join(durable_parts)[:2400] or (
            assistants[-1].text[:2400] if assistants else tools[-1].text[:2400] if tools else None
        )
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
    ) -> bool:
        loop = self._materialize_current_loop(decision, planned_state_delta)
        if loop is None:
            return False
        try:
            update = None
            if planned_state_delta is not None:
                if planned_state_delta.get("operations"):
                    update = self.state_updater.apply_result(
                        self.anchor, self.state, loop, planned_state_delta
                    )
            elif self.unified_controller:
                # The controller already made all online state decisions. The
                # final answer remains durable in Loop Memory; copying a whole
                # answer transcript into Global State would pollute it.
                update = None
            else:
                update = self.state_updater.update(self.anchor, self.state, loop)
            if update is not None:
                self.state = update.state
                loop.state_delta_ids.append(update.delta.delta_id)
                self.deltas.append(update.delta.to_dict())
        except Exception as exc:
            # Never copy a raw assistant transcript into Global State after a
            # controller plan fails. Archive the loop with a safe NOOP instead.
            detail = str(exc).replace("\n", " ")[:500]
            self._pending_switch.controller_error = (
                f"state_update:{exc.__class__.__name__}: {detail}"
            )
        for event in self.current_events:
            self.index.add_raw(event.raw_memory)
        self.index.add_loop(loop)
        self.completed_loops.append(loop)
        self._just_archived_memory_ids = [loop.loop_id]
        self.current_events = []
        return True

    def ingest_new_messages(self, canonical_messages: list[dict[str, Any]]) -> None:
        new_messages = canonical_messages[self._known_message_count :]
        self._known_message_count = len(canonical_messages)
        self._pending_switch = LoopBoundaryDecision(False, "same loop", 0.5)
        self._controller_succeeded = False
        self._last_control = None
        self._task_status = "CONTINUE"
        self._just_archived_memory_ids = []
        self._controller_query = ""
        self._controller_selected_ids = []
        self._controller_selected_hits = []
        if self.unified_controller and new_messages:
            latest_events = [self._event(message) for message in new_messages]
            latest_text = "\n".join(event.text for event in latest_events)
            seed_query = build_retrieval_query(self.anchor, self.state, latest_text)
            candidates = self._evidence_memory_candidates(seed_query)
            try:
                control = self.unified_controller.decide(
                    anchor=self.anchor,
                    state=self.state,
                    current_loop=self.current_events,
                    latest_events=latest_events,
                    candidates=candidates,
                    current_phase=self._research_phase,
                    current_loop_subgoal=self._current_loop_subgoal,
                    loop_progress=self._loop_progress,
                    loop_rounds=self._loop_rounds + 1,
                    stagnant_rounds=self._stagnant_rounds,
                )
                self._last_control = control
                self._task_status = control.task_status
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
                previous_signature = self._progress_signature(self._loop_progress)
                next_progress = control.loop_progress
                next_signature = self._progress_signature(next_progress)
                if previous_signature and previous_signature == next_signature:
                    self._stagnant_rounds += 1
                else:
                    self._stagnant_rounds = 0
                self._loop_progress = next_progress
                self._loop_rounds += 1
                # The latest events are the completed model/tool turn that the
                # controller just evaluated. They belong to the current work
                # unit; a switch affects the *next* model call.
                self.current_events.extend(latest_events)
                if decision.split and self.current_events:
                    archived = self._archive_current_loop(decision, control.state_delta)
                    if not archived:
                        decision.split = False
                    else:
                        self._current_loop_subgoal = control.next_loop_subgoal
                        # Start a clean evidence ledger while preserving the
                        # outgoing controller's action-policy handoff. Without
                        # this, a loop switch discards the very next action it
                        # selected for the research model.
                        handoff = empty_loop_progress()
                        for name in (
                            "tried_strategies",
                            "rejected_hypotheses",
                            "blocked_subgoals",
                            "promising_leads",
                            "research_direction",
                            "avoid",
                        ):
                            handoff[name] = control.loop_progress.get(name, handoff[name])
                        if control.loop_outcome == "BLOCKED":
                            blocked = [
                                *handoff.get("blocked_subgoals", []),
                                control.current_loop_subgoal,
                            ]
                            handoff["blocked_subgoals"] = list(
                                dict.fromkeys(item for item in blocked if str(item).strip())
                            )[:12]
                        self._loop_progress = handoff
                        self._loop_rounds = 0
                        self._stagnant_rounds = 0
                elif control.current_loop_subgoal:
                    self._current_loop_subgoal = control.current_loop_subgoal
                self._research_phase = control.research_phase
            except Exception as exc:
                self._pending_switch = LoopBoundaryDecision(False, "controller fallback", 0.0)
                detail = str(exc).replace("\n", " ")[:500]
                self._pending_switch.controller_error = (
                    f"unified_controller:{exc.__class__.__name__}: {detail}"
                )
                self.current_events.extend(latest_events)
                self._loop_rounds += 1
                self._stagnant_rounds += 1
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
            "current_goal": (
                {
                    "value": self.state.current_goal.value,
                    "status": self.state.current_goal.status.value,
                    "confidence": self.state.current_goal.confidence,
                }
                if self.state.current_goal
                else {"value": self.question, "status": "confirmed", "confidence": 1.0}
            ),
            "active_subgoals": _compact_state_items(self.state.active_subgoals),
            "working_hypotheses": _compact_state_items(self.state.working_hypotheses),
            "resolved_findings": _compact_state_items(self.state.resolved_findings),
            "open_questions": _compact_state_items(self.state.open_questions),
            "uncertainties": _compact_state_items(self.state.uncertainties),
            "next_actions": _compact_state_items(self.state.next_actions),
        }

    @staticmethod
    def _progress_signature(progress: dict[str, Any]) -> str:
        durable = {
            "resolved_aspects": progress.get("resolved_aspects") or [],
            "key_evidence": progress.get("key_evidence") or [],
            "candidate_answer": progress.get("candidate_answer") or "",
            "answer_stable": bool(progress.get("answer_stable")),
            "evidence_sufficient": bool(progress.get("evidence_sufficient")),
            # Discovering a lead or disproving a hypothesis is material
            # progress even before it becomes citation-bearing evidence.
            "promising_leads": sorted(
                (
                    str(item.get("kind") or ""),
                    str(item.get("entity") or "").casefold(),
                    str(item.get("status") or "ACTIVE"),
                )
                for item in progress.get("promising_leads") or []
                if isinstance(item, dict)
            ),
            "rejected_hypotheses": progress.get("rejected_hypotheses") or [],
        }
        return json.dumps(durable, ensure_ascii=False, sort_keys=True)

    def _research_instruction(self) -> str:
        """Turn controller Working State into a concrete research-model contract."""

        if self._task_status == "READY_TO_ANSWER":
            return (
                "All success criteria have sufficient citable evidence. Do not call any more tools. "
                "Produce the final answer now as Explanation, Exact Answer, and Confidence, using "
                "citations already present in the current loop or evidence memories."
            )
        progress = self._loop_progress
        direction = progress.get("research_direction") or {}
        lines = [
            "Use the Working State as a directional contract for the next research step.",
        ]
        if direction.get("objective"):
            lines.append(f"Research objective: {direction['objective']}")
        if direction.get("rationale"):
            lines.append(f"Why this direction matters: {direction['rationale']}")
        if direction.get("stop_condition"):
            lines.append(f"Stop this research direction when: {direction['stop_condition']}")
        leads = [
            item for item in progress.get("promising_leads") or []
            if isinstance(item, dict) and item.get("status") in {"ACTIVE", "VERIFIED"}
        ]
        if leads:
            lead_names = [
                str(item.get("entity") or item.get("source") or "").strip()
                for item in leads[:3]
                if isinstance(item, dict)
            ]
            lead_names = [item for item in lead_names if item]
            if lead_names:
                lines.append(
                    "Evidence-backed candidate hypotheses (optional, reassess against new evidence): "
                    + "; ".join(lead_names)
                )
        avoid = [str(item).strip() for item in progress.get("avoid") or [] if str(item).strip()]
        rejected = [
            str(item).strip()
            for item in progress.get("rejected_hypotheses") or []
            if str(item).strip()
        ]
        exclusions = [*avoid, *[f"rejected hypothesis: {item}" for item in rejected]]
        exclusions.extend(
            f"blocked subgoal: {item}"
            for item in progress.get("blocked_subgoals") or []
            if str(item).strip()
        )
        if exclusions:
            lines.append("Do not repeat: " + "; ".join(exclusions[:8]))
        if self._stagnant_rounds >= 2:
            lines.append(
                f"Anti-stagnation rule: there have been {self._stagnant_rounds} rounds without "
                "material ledger change. Do not issue a semantically equivalent reformulation of a "
                "failed query; exploit a promising result or change strategy."
            )
        lines.append(
            "No candidate is mandatory; retire one when evidence does not improve. You own the concrete "
            "research action: choose the browser tool, source, URL, and exact "
            "query yourself. The memory controller supplies direction and constraints only."
        )
        return "\n".join(lines)

    def _memory_block(self, query: str, hits: list[MemoryHit]) -> str:
        memories = [
            {
                "id": hit.memory_id,
                "type": hit.memory_type,
                "source_type": hit.metadata.get("source_type"),
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
            "research_control": {
                "task_status": self._task_status,
                "research_phase": self._research_phase,
                "current_loop_subgoal": self._current_loop_subgoal,
                "loop_working_state": {
                    "completion_test": self._loop_progress.get("completion_test", ""),
                    "progress_summary": self._loop_progress.get("progress_summary", ""),
                    "resolved_aspects": self._loop_progress.get("resolved_aspects", []),
                    "open_aspects": self._loop_progress.get("open_aspects", []),
                    "key_evidence": self._loop_progress.get("key_evidence", []),
                    "candidate_answer": self._loop_progress.get("candidate_answer", ""),
                    "promising_leads": self._loop_progress.get("promising_leads", []),
                    "rejected_hypotheses": self._loop_progress.get("rejected_hypotheses", []),
                    "avoid": self._loop_progress.get("avoid", []),
                },
                "loop_runtime": {
                    "rounds_in_current_loop": self._loop_rounds,
                    "rounds_without_material_progress": self._stagnant_rounds,
                },
                "instruction": self._research_instruction(),
            },
            "retrieval_query": query,
            "retrieved_cross_loop_memories": memories,
        }
        return (
            "\n\n<cross_loop_memory>\n"
            + (
                "IMPORTANT: The memory controller has marked this task READY_TO_ANSWER. "
                "Do not call tools; write the final cited answer now with Explanation, Exact Answer, "
                "and Confidence.\n"
                if self._task_status == "READY_TO_ANSWER"
                else ""
            )
            + "This is compact state and retrieved prior-loop evidence. Treat it as memory, not as new tool output.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n</cross_loop_memory>"
        )

    def build_prompt(self, canonical_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.ingest_new_messages(canonical_messages)
        recent_event = self.current_events[-1].text if self.current_events else "start research"
        query = self._controller_query or build_retrieval_query(self.anchor, self.state, recent_event)
        candidates = self._evidence_memory_candidates(query, limit=max(self.top_k, 12))
        if self._controller_succeeded:
            handoff_hits = self.index.lookup(self._just_archived_memory_ids)
            combined = [*handoff_hits, *self._controller_selected_hits]
            seen_ids: set[str] = set()
            hits = []
            for hit in combined:
                if hit.memory_id in seen_ids:
                    continue
                seen_ids.add(hit.memory_id)
                hits.append(hit)
                if len(hits) >= self.top_k:
                    break
        elif self.unified_controller is None:
            hits = candidates[: self.top_k]
        else:
            # A failed controller must fail closed. Direct BM25 fallback here
            # repeatedly injected old lexical matches during API/JSON errors
            # and amplified stale wrong directions.
            hits = []
        system = dict(canonical_messages[0])
        has_useful_memory = bool(self.completed_loops or hits or self.state.state_version > 1)
        if has_useful_memory or self._task_status == "READY_TO_ANSWER":
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
                task_status=self._task_status,
                research_phase=self._research_phase,
                current_loop_subgoal=self._current_loop_subgoal,
                next_loop_subgoal=getattr(getattr(self, "_last_control", None), "next_loop_subgoal", ""),
                loop_outcome=getattr(getattr(self, "_last_control", None), "loop_outcome", "IN_PROGRESS"),
                boundary_basis=getattr(getattr(self, "_last_control", None), "boundary_basis", "NONE"),
                loop_progress=getattr(
                    getattr(self, "_last_control", None), "loop_progress", self._loop_progress
                ),
                loop_rounds=self._loop_rounds,
                stagnant_rounds=self._stagnant_rounds,
                controller_error=getattr(self._pending_switch, "controller_error", None),
                controller_validation_retries=getattr(
                    getattr(self, "_last_control", None), "validation_retries", 0
                ),
                controller_response=getattr(
                    getattr(self, "_last_control", None), "raw_response", None
                ),
            )
        )
        return prompt_messages

    def finalize(self, canonical_messages: list[dict[str, Any]]) -> None:
        # Task termination is already known here. Preserve any final assistant
        # message without paying for another control-plane decision.
        new_messages = canonical_messages[self._known_message_count :]
        self._known_message_count = len(canonical_messages)
        self.current_events.extend(self._event(message) for message in new_messages)
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
