"""Run CL-GISM with an optional LLM loop boundary judge."""

from __future__ import annotations

import json
from pathlib import Path

from cl_gism import CLGISMEngine, HeuristicStateUpdater, LLMLoopBuilder, RuleBasedLoopBuilder


def main() -> None:
    path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
    row = json.loads(path.read_text())
    loop_builder = LLMLoopBuilder.from_env()
    engine = CLGISMEngine(
        loop_builder=loop_builder or RuleBasedLoopBuilder(),
        state_updater=HeuristicStateUpdater(),
    )
    task = engine.ingest_openresearcher_row(row)

    print("task_id:", task.task_id)
    print("loop_builder:", "enabled" if loop_builder else "disabled")
    print("loops:", len(task.loops))
    for index, loop in enumerate(task.loops, start=1):
        print(f"loop {index}:")
        print(f"  subgoal: {loop.subgoal}")
        print(f"  context: {loop.context}")
        print(f"  evidence_ids: {len(loop.evidence_ids)}")
        print(f"  conclusion: {loop.conclusion[:160] if loop.conclusion else ''}")


if __name__ == "__main__":
    main()
