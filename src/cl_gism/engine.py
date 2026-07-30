"""End-to-end CL-GISM MVP orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from typing import Any

from .retrieval import LexicalMemoryIndex, MemoryHit, build_retrieval_query, pack_context
from .llm_planner import LLMMemoryReranker, RetrievalPlan
from .llm_loop import LLMLoopBuilder
from .llm_state_update import LLMStateUpdater
from .state_update import HeuristicStateUpdater, StateUpdateResult
from .trajectory import ParsedTrajectory, RuleBasedLoopBuilder, parse_openresearcher_row
from .schema import GlobalIntentState, LoopMemory, RawMemory, TaskAnchor


@dataclass
class TaskMemory:
    task_id: str
    anchor: TaskAnchor
    raw_memories: list[RawMemory]
    loops: list[LoopMemory]
    state: GlobalIntentState
    state_history: list[GlobalIntentState] = field(default_factory=list)
    deltas: list[dict[str, Any]] = field(default_factory=list)


class CLGISMEngine:
    """Run the first state-conditioned Memory loop without a model dependency."""

    def __init__(
        self,
        loop_builder: Any | None = None,
        memory_reranker: LLMMemoryReranker | None = None,
        state_updater: Any | None = None,
    ) -> None:
        self.loop_builder = loop_builder or LLMLoopBuilder.from_env() or RuleBasedLoopBuilder()
        self.state_updater = state_updater or LLMStateUpdater.from_env() or HeuristicStateUpdater()
        self.index = LexicalMemoryIndex()
        self.memory_reranker = memory_reranker
        self.tasks: dict[str, TaskMemory] = {}

    def ingest_openresearcher_row(self, row: dict[str, Any]) -> TaskMemory:
        parsed: ParsedTrajectory = parse_openresearcher_row(row)
        loops = self.loop_builder.build(parsed)
        state = self.state_updater.initialize(parsed.anchor)
        state_history = [state]
        deltas: list[dict[str, Any]] = []
        for loop in loops:
            update: StateUpdateResult = self.state_updater.update(parsed.anchor, state, loop)
            state = update.state
            loop.state_delta_ids.append(update.delta.delta_id)
            state_history.append(state)
            deltas.append(update.delta.to_dict())

        task = TaskMemory(
            task_id=parsed.task_id,
            anchor=parsed.anchor,
            raw_memories=[event.raw_memory for event in parsed.events],
            loops=loops,
            state=state,
            state_history=state_history,
            deltas=deltas,
        )
        self.tasks[task.task_id] = task
        for memory in task.raw_memories:
            self.index.add_raw(memory)
        for loop in task.loops:
            self.index.add_loop(loop)
        return task

    def retrieve_for_task(
        self,
        task_id: str,
        current_event: str = "",
        top_k: int = 5,
        recent_messages: list[str] | None = None,
    ):
        query, hits, _ = self.retrieve_for_task_with_plan(
            task_id,
            current_event=current_event,
            top_k=top_k,
            recent_messages=recent_messages,
        )
        return query, hits

    def retrieve_for_task_with_plan(
        self,
        task_id: str,
        current_event: str = "",
        top_k: int = 5,
        recent_messages: list[str] | None = None,
    ) -> tuple[str, list[MemoryHit], RetrievalPlan]:
        task = self.tasks[task_id]
        seed_query = build_retrieval_query(task.anchor, task.state, current_event)
        candidate_top_k = top_k
        if self.memory_reranker:
            candidate_top_k = max(top_k, self.memory_reranker.candidate_pool_size)
        candidate_hits = self.index.search(seed_query, top_k=candidate_top_k, task_id=task_id)
        plan = RetrievalPlan(
            retrieval_query=seed_query,
            selected_memory_ids=[hit.memory_id for hit in candidate_hits[:top_k]],
            reason="lexical retrieval",
            source="lexical",
            model=(
                getattr(self.memory_reranker.client, "model", None)
                if self.memory_reranker and self.memory_reranker.client
                else None
            ),
        )
        selected_hits = candidate_hits[:top_k]
        if self.memory_reranker:
            plan, selected_hits = self.memory_reranker.rerank(
                task.anchor,
                task.state,
                current_event,
                candidate_hits,
                recent_messages=recent_messages,
            )
        return plan.retrieval_query, selected_hits, plan

    def build_llm_context(self, task_id: str, current_event: str = "", top_k: int = 5, recent_messages: list[str] | None = None):
        task = self.tasks[task_id]
        query, hits, plan = self.retrieve_for_task_with_plan(
            task_id,
            current_event=current_event,
            top_k=top_k,
            recent_messages=recent_messages,
        )
        context = pack_context(task.anchor, task.state, current_event, hits, recent_messages, retrieval_plan=asdict(plan))
        context["retrieval_query"] = query
        return context


__all__ = ["CLGISMEngine", "TaskMemory"]
