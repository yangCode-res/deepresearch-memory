"""Unified API controller for loop, state, and memory-selection decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .llm_planner import OpenAIChatJSONClient
from .retrieval import MemoryHit
from .schema import GlobalIntentState, TaskAnchor
from .trajectory import TrajectoryEvent


TASK_STATUSES = {"CONTINUE", "SWITCH_LOOP", "READY_TO_ANSWER"}
RESEARCH_PHASES = {
    "DISCOVERY",
    "CANDIDATE_VERIFICATION",
    "EVIDENCE_COMPLETION",
    "ANSWER_SYNTHESIS",
}
VALID_ITEM_STATUSES = {"active", "tentative", "confirmed", "rejected", "superseded", "resolved"}
VALID_SOURCE_TYPES = {"user", "agent", "tool", "paper", "web", "experiment", "system"}


@dataclass
class UnifiedControlDecision:
    task_status: str = "CONTINUE"
    research_phase: str = "DISCOVERY"
    switch_loop: bool = False
    reason: str = ""
    confidence: float = 0.0
    current_loop_subgoal: str = ""
    next_loop_subgoal: str = ""
    state_delta: dict[str, Any] = field(default_factory=lambda: {"summary": "", "operations": []})
    retrieval_query: str = ""
    selected_memory_ids: list[str] = field(default_factory=list)
    validation_retries: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


def _event(event: TrajectoryEvent, limit: int = 1000) -> dict[str, Any]:
    return {"sequence": event.sequence, "role": event.role, "text": event.text[:limit]}


def _state(state: GlobalIntentState) -> dict[str, Any]:
    def values(name: str) -> list[Any]:
        return [
            {
                "id": item.item_id,
                "value": item.value,
                "status": item.status.value,
                "confidence": item.confidence,
            }
            for item in getattr(state, name)[-6:]
        ]
    return {
        "version": state.state_version,
        "current_goal": (
            {
                "id": state.current_goal.item_id,
                "value": state.current_goal.value,
                "status": state.current_goal.status.value,
                "confidence": state.current_goal.confidence,
            }
            if state.current_goal
            else None
        ),
        "active_subgoals": values("active_subgoals"),
        "working_hypotheses": values("working_hypotheses"),
        "resolved_findings": values("resolved_findings"),
        "open_questions": values("open_questions"),
        "uncertainties": values("uncertainties"),
        "next_actions": values("next_actions"),
    }


class UnifiedMemoryController:
    """Make all control-plane decisions in one structured API call per round."""

    def __init__(self, client: OpenAIChatJSONClient, *, max_selected_memories: int = 4) -> None:
        self.client = client
        self.max_selected_memories = max_selected_memories

    def decide(
        self,
        *,
        anchor: TaskAnchor,
        state: GlobalIntentState,
        current_loop: list[TrajectoryEvent],
        latest_events: list[TrajectoryEvent],
        candidates: list[MemoryHit],
        current_phase: str = "DISCOVERY",
    ) -> UnifiedControlDecision:
        current_phase = str(current_phase or "DISCOVERY").upper()
        if current_phase not in RESEARCH_PHASES:
            raise ValueError(f"invalid current_phase {current_phase}")
        allowed_ids = {hit.memory_id for hit in candidates}
        system_prompt = (
            "You are the control plane for a deep-research memory system. Return only valid JSON. "
            "In one decision: (1) determine whether research should continue, switch to a genuinely new "
            "semantic subgoal, or stop researching and answer; (2) if switching, describe the minimal "
            "StateDelta learned from the completed current_loop; (3) select prior memories useful for the "
            "next research-model call. A loop is a stable semantic subgoal, not one search query. Query "
            "rephrasing, opening a result, collecting citations, and verifying multiple criteria for the "
            "same candidate normally remain in the same loop. Do not switch merely because a search stalled. "
            "Classify the next call into exactly one research phase: DISCOVERY means no concrete candidate "
            "has been identified; CANDIDATE_VERIFICATION means a concrete candidate is being tested against "
            "the requirements; EVIDENCE_COMPLETION means the answer candidate is stable and only missing "
            "source or citation coverage is being filled; ANSWER_SYNTHESIS means research is complete and the "
            "next call must answer. A phase change is a semantic loop boundary. "
            "Never solve the user's research question yourself and never invent memory IDs or evidence IDs. "
            "Operation contract: ADD may target active_subgoals, confirmed_constraints, soft_preferences, "
            "candidate_options, rejected_options, working_hypotheses, resolved_findings, open_questions, "
            "uncertainties, or next_actions. UPDATE may target current_goal only. RESOLVE may target "
            "open_questions only. Use mode=NOOP with operations=[] when a closed loop contains no durable "
            "state change. Never put citations or external document IDs in evidence_ids; leave it empty."
        )
        payload = {
            "task_anchor": {
                "goal": anchor.original_goal,
                "success_criteria": anchor.success_criteria,
                "immutable_constraints": anchor.immutable_constraints,
            },
            "global_state": _state(state),
            "current_research_phase": current_phase,
            "current_loop": [_event(event) for event in current_loop[-30:]],
            "latest_events": [_event(event) for event in latest_events],
            "memory_candidates": [
                {
                    "id": hit.memory_id,
                    "type": hit.memory_type,
                    "source_type": hit.metadata.get("source_type"),
                    "text": hit.text[:1200],
                }
                for hit in candidates
            ],
            "required_output": {
                "task_status": "CONTINUE|SWITCH_LOOP|READY_TO_ANSWER",
                "research_phase": "DISCOVERY|CANDIDATE_VERIFICATION|EVIDENCE_COMPLETION|ANSWER_SYNTHESIS",
                "loop": {
                    "switch": False,
                    "reason": "short reason",
                    "confidence": 0.8,
                    "current_loop_subgoal": "short label",
                    "next_loop_subgoal": "short label",
                },
                "state_delta": {
                    "mode": "APPLY|NOOP",
                    "summary": "short summary; empty when switch=false",
                    "operations": [
                        {
                            "operation": "ADD|UPDATE|RESOLVE",
                            "target": "working_hypotheses|resolved_findings|open_questions|uncertainties|next_actions|active_subgoals",
                            "value": "concise value",
                            "reason": "why",
                            "evidence_ids": [],
                            "target_item_ids": [],
                            "item": {
                                "status": "tentative",
                                "confidence": 0.7,
                                "source_type": "agent",
                                "valid_time": None,
                                "contradicts": [],
                                "supersedes": [],
                                "user_confirmed": False,
                            },
                        }
                    ],
                },
                "retrieval": {
                    "query": "short semantic memory query",
                    "selected_memory_ids": [],
                    "reason": "short reason",
                },
            },
            "rules": [
                "Return READY_TO_ANSWER only when the exact answer is identified and every required claim has sufficient citable evidence in current_loop, latest_events, state, or selected memories.",
                "Return SWITCH_LOOP only for a genuinely different subgoal, candidate, or research phase; ordinary assistant/tool continuations remain CONTINUE.",
                "READY_TO_ANSWER and CONTINUE require loop.switch=false; SWITCH_LOOP requires loop.switch=true.",
                "Return CONTINUE only when research_phase equals current_research_phase.",
                "Return SWITCH_LOOP when research_phase changes between DISCOVERY, CANDIDATE_VERIFICATION, and EVIDENCE_COMPLETION.",
                "When a candidate changes from unknown to a concrete named candidate, switch from DISCOVERY to CANDIDATE_VERIFICATION.",
                "When the candidate is stable and the remaining work is only missing sources or citations, switch from CANDIDATE_VERIFICATION to EVIDENCE_COMPLETION.",
                "If a candidate is rejected and broad search resumes, switch back to DISCOVERY.",
                "READY_TO_ANSWER requires research_phase=ANSWER_SYNTHESIS and sufficient citable evidence.",
                "Only emit StateDelta operations when loop.switch=true.",
                "When switch=false, state_delta.mode must be NOOP and operations must be empty.",
                "When switch=true, use APPLY with at least one concise operation that records the durable phase result.",
                "StateDelta values must be concise durable facts, never reasoning transcripts, search narration, or tool-call JSON.",
                "Preserve citation markers or source coordinates inside finding values when they are available.",
                "UPDATE+current_goal and RESOLVE+open_questions are the only valid non-ADD combinations.",
                "Use at most four StateDelta operations.",
                f"Select at most {self.max_selected_memories} candidate memory IDs.",
                "Prefer tool-source memories containing evidence and citations over assistant search narration.",
                "Return a single JSON object with exactly task_status, research_phase, loop, state_delta, and retrieval.",
            ],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)
        raw: dict[str, Any] = {}
        last_error = ""
        for attempt in range(2):
            if attempt:
                repair = {
                    "validation_error": last_error,
                    "invalid_response": raw,
                    "instruction": "Return a corrected complete JSON object obeying the operation contract.",
                    "original_input": payload,
                }
                current_prompt = json.dumps(repair, ensure_ascii=False)
            else:
                current_prompt = user_prompt
            raw = self.client.complete_json(system_prompt, current_prompt)
            try:
                self._validate_contract(raw, state, allowed_ids, current_phase)
                break
            except ValueError as exc:
                last_error = str(exc)
        else:
            raise ValueError(f"controller contract invalid after retry: {last_error}")
        loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
        delta = raw.get("state_delta") if isinstance(raw.get("state_delta"), dict) else {}
        retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        requested = retrieval.get("selected_memory_ids")
        selected = []
        if isinstance(requested, list):
            selected = [str(item) for item in requested if str(item) in allowed_ids][
                : self.max_selected_memories
            ]
        try:
            confidence = max(0.0, min(1.0, float(loop.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return UnifiedControlDecision(
            task_status=str(raw.get("task_status") or "CONTINUE").upper(),
            research_phase=str(raw.get("research_phase") or current_phase).upper(),
            switch_loop=bool(loop.get("switch", False)),
            reason=str(loop.get("reason") or ""),
            confidence=confidence,
            current_loop_subgoal=str(loop.get("current_loop_subgoal") or ""),
            next_loop_subgoal=str(loop.get("next_loop_subgoal") or ""),
            state_delta=delta or {"summary": "", "operations": []},
            retrieval_query=str(retrieval.get("query") or ""),
            selected_memory_ids=selected,
            validation_retries=1 if last_error else 0,
            raw_response=raw,
        )

    @staticmethod
    def _validate_contract(
        raw: dict[str, Any],
        state: GlobalIntentState,
        allowed_ids: set[str],
        current_phase: str,
    ) -> None:
        if not isinstance(raw, dict):
            raise ValueError("response must be an object")
        if set(raw) != {"task_status", "research_phase", "loop", "state_delta", "retrieval"}:
            raise ValueError(
                "response must contain exactly task_status, research_phase, loop, state_delta, and retrieval"
            )
        task_status = str(raw.get("task_status") or "").upper()
        if task_status not in TASK_STATUSES:
            raise ValueError("task_status must be CONTINUE, SWITCH_LOOP, or READY_TO_ANSWER")
        research_phase = str(raw.get("research_phase") or "").upper()
        if research_phase not in RESEARCH_PHASES:
            raise ValueError("invalid research_phase")
        loop = raw.get("loop")
        delta = raw.get("state_delta")
        retrieval = raw.get("retrieval")
        if not all(isinstance(value, dict) for value in (loop, delta, retrieval)):
            raise ValueError("loop, state_delta, and retrieval must be objects")
        if not isinstance(loop.get("switch"), bool):
            raise ValueError("loop.switch must be boolean")
        if (task_status == "SWITCH_LOOP") != loop["switch"]:
            raise ValueError("SWITCH_LOOP requires loop.switch=true and other statuses require false")
        if task_status == "CONTINUE" and research_phase != current_phase:
            raise ValueError("CONTINUE must keep the current research phase")
        if task_status == "SWITCH_LOOP" and research_phase == current_phase:
            raise ValueError("SWITCH_LOOP must change the research phase")
        if task_status == "SWITCH_LOOP" and research_phase == "ANSWER_SYNTHESIS":
            raise ValueError("use READY_TO_ANSWER for ANSWER_SYNTHESIS")
        if task_status == "READY_TO_ANSWER" and research_phase != "ANSWER_SYNTHESIS":
            raise ValueError("READY_TO_ANSWER requires ANSWER_SYNTHESIS")
        if research_phase == "ANSWER_SYNTHESIS" and task_status != "READY_TO_ANSWER":
            raise ValueError("ANSWER_SYNTHESIS requires READY_TO_ANSWER")
        operations = delta.get("operations")
        if not isinstance(operations, list):
            raise ValueError("state_delta.operations must be a list")
        mode = str(delta.get("mode") or "").upper()
        if not loop["switch"] and (mode != "NOOP" or operations):
            raise ValueError("switch=false requires mode=NOOP and no operations")
        if mode not in {"APPLY", "NOOP"}:
            raise ValueError("state_delta.mode must be APPLY or NOOP")
        if mode == "NOOP" and operations:
            raise ValueError("NOOP cannot contain operations")
        if task_status == "SWITCH_LOOP" and (mode != "APPLY" or not operations):
            raise ValueError("SWITCH_LOOP requires APPLY with at least one durable operation")
        add_targets = {
            "active_subgoals", "confirmed_constraints", "soft_preferences", "candidate_options",
            "rejected_options", "working_hypotheses", "resolved_findings", "open_questions",
            "uncertainties", "next_actions",
        }
        valid_open_ids = {item.item_id for item in state.open_questions}
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("each operation must be an object")
            operation = str(op.get("operation") or "").upper()
            target = str(op.get("target") or "")
            if operation == "ADD" and target not in add_targets:
                raise ValueError(f"ADD cannot target {target}")
            if operation == "UPDATE" and target != "current_goal":
                raise ValueError("UPDATE may target current_goal only")
            if operation == "RESOLVE" and target != "open_questions":
                raise ValueError("RESOLVE may target open_questions only")
            if operation not in {"ADD", "UPDATE", "RESOLVE"}:
                raise ValueError(f"unsupported operation {operation}")
            if not str(op.get("reason") or "").strip():
                raise ValueError("every operation requires a reason")
            if operation in {"ADD", "UPDATE"} and op.get("value") in (None, ""):
                raise ValueError("ADD/UPDATE requires a value")
            if operation in {"ADD", "UPDATE"} and len(str(op.get("value"))) > 800:
                raise ValueError("StateDelta values must be at most 800 characters")
            evidence_ids = op.get("evidence_ids", [])
            if evidence_ids:
                raise ValueError("controller evidence_ids must be empty")
            target_ids = {str(item) for item in op.get("target_item_ids", [])}
            if operation == "RESOLVE" and target_ids and not target_ids <= valid_open_ids:
                raise ValueError("RESOLVE contains unknown open_question IDs")
            item = op.get("item")
            if operation in {"ADD", "UPDATE"}:
                if item is not None and not isinstance(item, dict):
                    raise ValueError("item must be an object")
                if isinstance(item, dict):
                    status = str(item.get("status") or "tentative")
                    source_type = str(item.get("source_type") or "agent")
                    if status not in VALID_ITEM_STATUSES:
                        raise ValueError(f"invalid item.status {status}")
                    if source_type not in VALID_SOURCE_TYPES:
                        raise ValueError(f"invalid item.source_type {source_type}")
                    try:
                        confidence = float(item.get("confidence", 0.5))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("item.confidence must be numeric") from exc
                    if not 0.0 <= confidence <= 1.0:
                        raise ValueError("item.confidence must be between 0 and 1")
                    for field_name in ("contradicts", "supersedes"):
                        if not isinstance(item.get(field_name, []), list):
                            raise ValueError(f"item.{field_name} must be a list")
        selected = retrieval.get("selected_memory_ids", [])
        if not isinstance(selected, list) or any(str(item) not in allowed_ids for item in selected):
            raise ValueError("retrieval contains unknown memory IDs")


__all__ = ["UnifiedControlDecision", "UnifiedMemoryController"]
