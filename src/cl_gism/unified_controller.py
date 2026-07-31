"""Unified API controller for loop, state, and memory-selection decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .llm_planner import OpenAIChatJSONClient
from .retrieval import MemoryHit
from .schema import GlobalIntentState, TaskAnchor
from .trajectory import TrajectoryEvent


@dataclass
class UnifiedControlDecision:
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
        return [item.value for item in getattr(state, name)[-6:]]
    return {
        "version": state.state_version,
        "current_goal": state.current_goal.value if state.current_goal else None,
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
    ) -> UnifiedControlDecision:
        allowed_ids = {hit.memory_id for hit in candidates}
        system_prompt = (
            "You are the control plane for a deep-research memory system. Return only valid JSON. "
            "In one decision: (1) determine whether latest_events start a new semantic research loop; "
            "(2) if switching, describe the minimal StateDelta learned from the completed current_loop; "
            "(3) select prior memories useful for the next research-model call. "
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
            "current_loop": [_event(event) for event in current_loop[-30:]],
            "latest_events": [_event(event) for event in latest_events],
            "memory_candidates": [
                {"id": hit.memory_id, "type": hit.memory_type, "text": hit.text[:1200]}
                for hit in candidates
            ],
            "required_output": {
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
                "Decide a boundary on every call; do not split ordinary assistant/tool continuations.",
                "Only emit StateDelta operations when loop.switch=true.",
                "When switch=false, state_delta.mode must be NOOP and operations must be empty.",
                "When switch=true, use APPLY with valid operations, or NOOP if nothing durable was learned.",
                "UPDATE+current_goal and RESOLVE+open_questions are the only valid non-ADD combinations.",
                "Use at most four StateDelta operations.",
                f"Select at most {self.max_selected_memories} candidate memory IDs.",
                "Return a single JSON object with exactly loop, state_delta, and retrieval.",
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
                self._validate_contract(raw, state, allowed_ids)
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
    def _validate_contract(raw: dict[str, Any], state: GlobalIntentState, allowed_ids: set[str]) -> None:
        if not isinstance(raw, dict):
            raise ValueError("response must be an object")
        loop = raw.get("loop")
        delta = raw.get("state_delta")
        retrieval = raw.get("retrieval")
        if not all(isinstance(value, dict) for value in (loop, delta, retrieval)):
            raise ValueError("loop, state_delta, and retrieval must be objects")
        if not isinstance(loop.get("switch"), bool):
            raise ValueError("loop.switch must be boolean")
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
            evidence_ids = op.get("evidence_ids", [])
            if evidence_ids:
                raise ValueError("controller evidence_ids must be empty")
            target_ids = {str(item) for item in op.get("target_item_ids", [])}
            if operation == "RESOLVE" and target_ids and not target_ids <= valid_open_ids:
                raise ValueError("RESOLVE contains unknown open_question IDs")
        selected = retrieval.get("selected_memory_ids", [])
        if not isinstance(selected, list) or any(str(item) not in allowed_ids for item in selected):
            raise ValueError("retrieval contains unknown memory IDs")


__all__ = ["UnifiedControlDecision", "UnifiedMemoryController"]
