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
            "Never solve the user's research question yourself and never invent memory IDs or evidence IDs."
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
                "Use at most four StateDelta operations.",
                f"Select at most {self.max_selected_memories} candidate memory IDs.",
                "Return a single JSON object with exactly loop, state_delta, and retrieval.",
            ],
        }
        raw = self.client.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False))
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
        )


__all__ = ["UnifiedControlDecision", "UnifiedMemoryController"]
