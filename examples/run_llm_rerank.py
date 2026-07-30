"""Run the CL-GISM retrieval path with an optional LLM reranker."""

from __future__ import annotations

import json
from pathlib import Path

from cl_gism import CLGISMEngine, HeuristicStateUpdater, LLMMemoryReranker, RuleBasedLoopBuilder


def main() -> None:
    path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
    row = json.loads(path.read_text())
    reranker = LLMMemoryReranker.from_env()
    engine = CLGISMEngine(
        loop_builder=RuleBasedLoopBuilder(),
        state_updater=HeuristicStateUpdater(),
        memory_reranker=reranker,
    )
    task = engine.ingest_openresearcher_row(row)
    context = engine.build_llm_context(
        task.task_id,
        current_event="Need to decide whether the evidence is sufficient for the final answer.",
        top_k=5,
        recent_messages=["tool returned a page excerpt with release dates"],
    )

    print("task_id:", task.task_id)
    print("reranker:", "enabled" if reranker else "disabled")
    print("retrieval_mode:", context["retrieval_plan"]["source"])
    print("retrieval_query:\n", context["retrieval_query"])
    print("retrieval_plan:")
    print(json.dumps(context["retrieval_plan"], ensure_ascii=False, indent=2))
    print("retrieved_memories:")
    for item in context["retrieved_memories"]:
        print(f"- {item['memory_id']} ({item['memory_type']}, {item['score']}): {item['text'][:160]}")


if __name__ == "__main__":
    main()
