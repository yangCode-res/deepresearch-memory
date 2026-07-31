"""Optional LLM-backed StateDelta planner for CL-GISM."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from .llm_planner import OpenAIChatJSONClient
from .schema import (
    DeltaOperation,
    GlobalIntentState,
    LoopMemory,
    MemoryStatus,
    SourceType,
    StateDelta,
    StateDeltaOperation,
    StateItem,
    StateItemKind,
    TaskAnchor,
)
from .state_update import HeuristicStateUpdater, StateUpdateResult


TARGET_KIND_MAP: dict[str, StateItemKind] = {
    "current_goal": StateItemKind.CURRENT_GOAL,
    "active_subgoals": StateItemKind.ACTIVE_SUBGOAL,
    "confirmed_constraints": StateItemKind.CONFIRMED_CONSTRAINT,
    "soft_preferences": StateItemKind.SOFT_PREFERENCE,
    "candidate_options": StateItemKind.CANDIDATE_OPTION,
    "rejected_options": StateItemKind.REJECTED_OPTION,
    "working_hypotheses": StateItemKind.WORKING_HYPOTHESIS,
    "resolved_findings": StateItemKind.RESOLVED_FINDING,
    "open_questions": StateItemKind.OPEN_QUESTION,
    "uncertainties": StateItemKind.UNCERTAINTY,
    "next_actions": StateItemKind.NEXT_ACTION,
}

SUPPORTED_TARGETS = set(TARGET_KIND_MAP)
SUPPORTED_OPERATIONS = {DeltaOperation.ADD, DeltaOperation.UPDATE, DeltaOperation.RESOLVE}
DEFAULT_STATUS_BY_TARGET: dict[str, MemoryStatus] = {
    "current_goal": MemoryStatus.CONFIRMED,
    "active_subgoals": MemoryStatus.ACTIVE,
    "confirmed_constraints": MemoryStatus.CONFIRMED,
    "soft_preferences": MemoryStatus.TENTATIVE,
    "candidate_options": MemoryStatus.TENTATIVE,
    "rejected_options": MemoryStatus.REJECTED,
    "working_hypotheses": MemoryStatus.TENTATIVE,
    "resolved_findings": MemoryStatus.CONFIRMED,
    "open_questions": MemoryStatus.ACTIVE,
    "uncertainties": MemoryStatus.ACTIVE,
    "next_actions": MemoryStatus.ACTIVE,
}


def _compact_state_item(item: StateItem) -> dict[str, Any]:
    return {
        "id": item.item_id,
        "kind": item.kind.value,
        "value": item.value,
        "status": item.status.value,
        "confidence": item.confidence,
    }


def _compact_state(state: GlobalIntentState) -> dict[str, Any]:
    return {
        "state_version": state.state_version,
        "current_goal": _compact_state_item(state.current_goal) if state.current_goal else None,
        "active_subgoals": [_compact_state_item(item) for item in state.active_subgoals],
        "confirmed_constraints": [_compact_state_item(item) for item in state.confirmed_constraints],
        "soft_preferences": [_compact_state_item(item) for item in state.soft_preferences],
        "candidate_options": [_compact_state_item(item) for item in state.candidate_options],
        "rejected_options": [_compact_state_item(item) for item in state.rejected_options],
        "working_hypotheses": [_compact_state_item(item) for item in state.working_hypotheses],
        "resolved_findings": [_compact_state_item(item) for item in state.resolved_findings],
        "open_questions": [_compact_state_item(item) for item in state.open_questions],
        "uncertainties": [_compact_state_item(item) for item in state.uncertainties],
        "next_actions": [_compact_state_item(item) for item in state.next_actions],
    }


def _compact_loop(loop: LoopMemory) -> dict[str, Any]:
    return {
        "loop_id": loop.loop_id,
        "subgoal": loop.subgoal,
        "actions": loop.actions,
        "observations": loop.observations,
        "conclusion": loop.conclusion,
        "evidence_ids": loop.evidence_ids,
    }


def _strip_empty(items: list[str]) -> list[str]:
    return [item for item in items if item]


def _build_state_item(
    target: str,
    value: Any,
    draft: dict[str, Any] | None,
    loop: LoopMemory,
    evidence_ids: list[str],
) -> StateItem:
    item_data = draft or {}
    kind = TARGET_KIND_MAP[target]
    status = MemoryStatus(item_data.get("status") or DEFAULT_STATUS_BY_TARGET[target].value)
    source_type = SourceType(item_data.get("source_type") or SourceType.AGENT.value)
    confidence = float(item_data.get("confidence", 0.5))
    valid_time = item_data.get("valid_time")
    contradicts = [str(item_id) for item_id in item_data.get("contradicts", [])]
    supersedes = [str(item_id) for item_id in item_data.get("supersedes", [])]
    user_confirmed = bool(item_data.get("user_confirmed", False))
    if source_type is not SourceType.USER:
        user_confirmed = False
    return StateItem(
        kind=kind,
        value=value,
        status=status,
        confidence=confidence,
        source_type=source_type,
        evidence_ids=evidence_ids or [loop.loop_id, *loop.evidence_ids],
        created_in_loop=loop.loop_id,
        updated_in_loop=loop.loop_id,
        valid_time=valid_time,
        contradicts=contradicts,
        supersedes=supersedes,
        user_confirmed=user_confirmed,
    )


@dataclass
class PlannedStateOperation:
    operation: DeltaOperation
    target: str
    reason: str
    value: Any = None
    evidence_ids: list[str] = field(default_factory=list)
    target_item_ids: list[str] = field(default_factory=list)
    item: dict[str, Any] | None = None
    old_value: Any = None


@dataclass
class StateDeltaPlan:
    summary: str
    operations: list[PlannedStateOperation]
    source: str = "llm"
    model: str | None = None


def _parse_plan(raw: dict[str, Any]) -> StateDeltaPlan:
    summary = str(raw.get("summary") or "").strip()
    operations: list[PlannedStateOperation] = []
    raw_operations = raw.get("operations")
    if not isinstance(raw_operations, list):
        raise ValueError("LLM state plan operations must be a list")
    for entry in raw_operations:
        if not isinstance(entry, dict):
            raise ValueError("LLM state plan operations must be objects")
        operation = DeltaOperation(str(entry.get("operation") or "").upper())
        target = str(entry.get("target") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported state operation: {operation}")
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported state target: {target}")
        evidence_ids = [str(item) for item in entry.get("evidence_ids", []) if str(item)]
        target_item_ids = [str(item) for item in entry.get("target_item_ids", []) if str(item)]
        operations.append(
            PlannedStateOperation(
                operation=operation,
                target=target,
                reason=reason,
                value=entry.get("value"),
                evidence_ids=evidence_ids,
                target_item_ids=target_item_ids,
                item=entry.get("item") if isinstance(entry.get("item"), dict) else None,
                old_value=entry.get("old_value"),
            )
        )
    return StateDeltaPlan(summary=summary, operations=operations)


def _find_items(state: GlobalIntentState, target: str, target_item_ids: list[str]) -> list[StateItem]:
    if not target_item_ids:
        if target == "current_goal":
            return [state.current_goal] if state.current_goal else []
        return list(getattr(state, target))
    wanted = set(target_item_ids)
    if target == "current_goal":
        return [state.current_goal] if state.current_goal and state.current_goal.item_id in wanted else []
    return [item for item in getattr(state, target) if item.item_id in wanted]


def _apply_planned_operations(
    anchor: TaskAnchor,
    previous: GlobalIntentState,
    loop: LoopMemory,
    plan: StateDeltaPlan,
) -> StateUpdateResult:
    state = deepcopy(previous)
    state.state_version = previous.state_version + 1
    state.source_loop_id = loop.loop_id
    state.updated_at = loop.ended_at or loop.started_at

    delta_operations: list[StateDeltaOperation] = []
    base_evidence = [loop.loop_id, *loop.evidence_ids]

    for planned in plan.operations:
        evidence_ids = _strip_empty(planned.evidence_ids) or base_evidence
        delta_operation = StateDeltaOperation(
            operation=planned.operation,
            target=planned.target,
            reason=planned.reason,
            value=planned.value,
            old_value=planned.old_value,
            evidence_ids=evidence_ids,
            loop_id=loop.loop_id,
        )

        if planned.operation is DeltaOperation.ADD:
            item = _build_state_item(planned.target, planned.value, planned.item, loop, evidence_ids)
            if planned.target == "current_goal":
                old_item = state.current_goal
                delta_operation.old_value = old_item.value if old_item else None
                state.current_goal = item
            else:
                getattr(state, planned.target).append(item)

        elif planned.operation is DeltaOperation.UPDATE:
            if planned.target != "current_goal":
                raise ValueError("UPDATE is only supported for current_goal in this MVP")
            item = _build_state_item(planned.target, planned.value, planned.item, loop, evidence_ids)
            old_item = state.current_goal
            delta_operation.old_value = old_item.value if old_item else None
            state.current_goal = item

        elif planned.operation is DeltaOperation.RESOLVE:
            if planned.target != "open_questions":
                raise ValueError("RESOLVE is only supported for open_questions in this MVP")
            if planned.target_item_ids:
                targets = _find_items(state, planned.target, planned.target_item_ids)
                if not targets:
                    raise ValueError("RESOLVE referenced unknown open_question ids")
            else:
                targets = list(state.open_questions)
            for item in targets:
                item.status = MemoryStatus.RESOLVED
                item.updated_in_loop = loop.loop_id
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported state operation: {planned.operation}")

        delta_operations.append(delta_operation)

    delta = StateDelta(
        task_id=anchor.task_id,
        from_state_version=previous.state_version,
        to_state_version=state.state_version,
        operations=delta_operations,
        generated_from_loop_id=loop.loop_id,
    )
    delta.validate()
    state.validate()
    return StateUpdateResult(state=state, delta=delta)


class LLMStateUpdater:
    """Use an OpenAI-compatible model to decide the next StateDelta."""

    def __init__(
        self,
        client: OpenAIChatJSONClient | None,
        *,
        fallback: HeuristicStateUpdater | None = None,
        max_operations: int = 4,
    ) -> None:
        self.client = client
        self.fallback = fallback or HeuristicStateUpdater()
        self.max_operations = max_operations

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_operations: int = 4,
    ) -> "LLMStateUpdater" | None:
        client = OpenAIChatJSONClient.from_env(model=model, base_url=base_url)
        if not client:
            return None
        return cls(client, max_operations=max_operations)

    def initialize(self, anchor: TaskAnchor) -> GlobalIntentState:
        return self.fallback.initialize(anchor)

    def _build_prompts(
        self,
        anchor: TaskAnchor,
        previous: GlobalIntentState,
        loop: LoopMemory,
    ) -> tuple[str, str]:
        system_prompt = (
            "You plan StateDelta mutations for a deep-research memory system.\n"
            "Return only valid JSON.\n"
            "Use the minimum number of operations needed to update state.\n"
            "Allowed operations: ADD, UPDATE, RESOLVE.\n"
            "Allowed targets: current_goal, active_subgoals, confirmed_constraints, soft_preferences, "
            "candidate_options, rejected_options, working_hypotheses, resolved_findings, open_questions, "
            "uncertainties, next_actions.\n"
            "For ADD and UPDATE, include an item object with status, confidence, source_type, "
            "valid_time, contradicts, supersedes, and user_confirmed.\n"
            "For RESOLVE, target open_questions and include target_item_ids when possible.\n"
            "Keep evidence_ids grounded in the loop or raw memory ids already present."
        )
        user_payload = {
            "task_id": anchor.task_id,
            "goal": anchor.original_goal,
            "success_criteria": anchor.success_criteria,
            "immutable_constraints": anchor.immutable_constraints,
            "previous_state": _compact_state(previous),
            "loop": _compact_loop(loop),
            "allowed_output": {
                "summary": "short explanation",
                "operations": [
                    {
                        "operation": "ADD",
                        "target": "working_hypotheses",
                        "value": "string or structured value",
                        "reason": "why this mutation is needed",
                        "evidence_ids": ["raw_...", "loop_..."],
                        "target_item_ids": [],
                        "item": {
                            "status": "tentative",
                            "confidence": 0.6,
                            "source_type": "agent",
                            "valid_time": None,
                            "contradicts": [],
                            "supersedes": [],
                            "user_confirmed": False
                        }
                    }
                ]
            },
            "rules": [
                "Use at most four operations.",
                "Prefer ADD for new hypotheses, findings, next actions, or subgoals.",
                "Use RESOLVE only when the loop clearly closes an open question.",
                "Use UPDATE only for current_goal reframing.",
                "Do not invent evidence ids or state item ids.",
            ],
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
        return system_prompt, user_prompt

    def update(self, anchor: TaskAnchor, previous: GlobalIntentState, loop: LoopMemory) -> StateUpdateResult:
        if not self.client:
            return self.fallback.update(anchor, previous, loop)

        system_prompt, user_prompt = self._build_prompts(anchor, previous, loop)
        try:
            raw_result = self.client.complete_json(system_prompt, user_prompt)
            return self.apply_result(anchor, previous, loop, raw_result)
        except Exception:  # pragma: no cover - network and parse fallback path
            return self.fallback.update(anchor, previous, loop)

    def apply_result(
        self,
        anchor: TaskAnchor,
        previous: GlobalIntentState,
        loop: LoopMemory,
        raw_result: dict[str, Any],
    ) -> StateUpdateResult:
        """Apply an already generated StateDelta plan without another LLM call."""
        plan = _parse_plan(raw_result)
        if len(plan.operations) > self.max_operations:
            plan.operations = plan.operations[: self.max_operations]
        return _apply_planned_operations(anchor, previous, loop, plan)


__all__ = [
    "LLMStateUpdater",
    "PlannedStateOperation",
    "StateDeltaPlan",
]
