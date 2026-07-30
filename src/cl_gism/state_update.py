"""A transparent, non-neural Global State updater for the MVP."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

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
    utc_now,
)


@dataclass
class StateUpdateResult:
    state: GlobalIntentState
    delta: StateDelta


class HeuristicStateUpdater:
    """Create auditable state snapshots without requiring an LLM.

    This is a baseline for validating the data flow.  It is intentionally
    conservative: it records Loop conclusions as tentative findings and only
    treats messages containing an explicit answer marker as confirmed.
    """

    def initialize(self, anchor: TaskAnchor) -> GlobalIntentState:
        goal = StateItem(
            kind=StateItemKind.CURRENT_GOAL,
            value=anchor.original_goal,
            status=MemoryStatus.CONFIRMED,
            confidence=1.0,
            source_type=SourceType.USER,
            evidence_ids=list(anchor.evidence_ids),
            user_confirmed=True,
        )
        question = StateItem(
            kind=StateItemKind.OPEN_QUESTION,
            value=anchor.original_goal,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
            source_type=SourceType.USER,
            evidence_ids=list(anchor.evidence_ids),
            user_confirmed=True,
        )
        state = GlobalIntentState(
            task_id=anchor.task_id,
            current_goal=goal,
            open_questions=[question],
        )
        state.validate()
        return state

    def update(self, anchor: TaskAnchor, previous: GlobalIntentState, loop: LoopMemory) -> StateUpdateResult:
        state = deepcopy(previous)
        next_version = previous.state_version + 1
        state.state_version = next_version
        state.updated_at = utc_now()
        state.source_loop_id = loop.loop_id

        conclusion = (loop.conclusion or "").strip()
        operations: list[StateDeltaOperation] = []
        if conclusion:
            confirmed = bool(re.search(r"exact answer:|final answer:|<answer>", conclusion, re.I))
            item = StateItem(
                kind=StateItemKind.RESOLVED_FINDING if confirmed else StateItemKind.WORKING_HYPOTHESIS,
                value=conclusion,
                status=MemoryStatus.CONFIRMED if confirmed else MemoryStatus.TENTATIVE,
                confidence=0.9 if confirmed else 0.55,
                source_type=SourceType.AGENT,
                evidence_ids=[loop.loop_id, *loop.evidence_ids],
                created_in_loop=loop.loop_id,
                updated_in_loop=loop.loop_id,
            )
            if confirmed:
                state.resolved_findings.append(item)
                for question in state.open_questions:
                    question.status = MemoryStatus.RESOLVED
                operations.append(
                    StateDeltaOperation(
                        operation=DeltaOperation.RESOLVE,
                        target="open_questions",
                        value=anchor.original_goal,
                        reason="Loop contains an explicit final answer marker",
                        evidence_ids=[loop.loop_id, *loop.evidence_ids],
                        loop_id=loop.loop_id,
                    )
                )
            else:
                state.working_hypotheses.append(item)
            operations.append(
                StateDeltaOperation(
                    operation=DeltaOperation.ADD,
                    target="resolved_findings" if confirmed else "working_hypotheses",
                    value=conclusion,
                    reason="Record the latest Loop conclusion",
                    evidence_ids=[loop.loop_id, *loop.evidence_ids],
                    loop_id=loop.loop_id,
                )
            )

        delta = StateDelta(
            task_id=anchor.task_id,
            from_state_version=previous.state_version,
            to_state_version=next_version,
            operations=operations,
            generated_from_loop_id=loop.loop_id,
        )
        delta.validate()
        state.validate()
        return StateUpdateResult(state=state, delta=delta)


__all__ = ["HeuristicStateUpdater", "StateUpdateResult"]
