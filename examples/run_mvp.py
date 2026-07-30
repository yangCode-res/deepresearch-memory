"""Run the first CL-GISM loop on the bundled short OpenResearcher example."""

from __future__ import annotations

import json
from pathlib import Path

from cl_gism import CLGISMEngine, HeuristicStateUpdater, RuleBasedLoopBuilder


def main() -> None:
    path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
    row = json.loads(path.read_text())
    engine = CLGISMEngine(loop_builder=RuleBasedLoopBuilder(), state_updater=HeuristicStateUpdater())
    task = engine.ingest_openresearcher_row(row)
    context = engine.build_llm_context(
        task.task_id,
        current_event="Need to decide whether the evidence is sufficient for the final answer.",
        top_k=5,
        recent_messages=["tool returned a page excerpt with release dates"],
    )

    print("task_id:", task.task_id)
    print("raw_memories:", len(task.raw_memories))
    print("loops:", len(task.loops))
    print("state_version:", task.state.state_version)
    print("retrieval_query:\n", context["retrieval_query"])
    print("retrieved_memories:")
    for item in context["retrieved_memories"]:
        print(f"- {item['memory_id']} ({item['memory_type']}, {item['score']}): {item['text'][:160]}")


if __name__ == "__main__":
    main()
