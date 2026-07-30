"""Run CL-GISM with an optional LLM StateDelta updater."""

from __future__ import annotations

import json
from pathlib import Path

from cl_gism import CLGISMEngine, HeuristicStateUpdater, LLMStateUpdater, RuleBasedLoopBuilder


def main() -> None:
    path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
    row = json.loads(path.read_text())
    updater = LLMStateUpdater.from_env()
    engine = CLGISMEngine(
        loop_builder=RuleBasedLoopBuilder(),
        state_updater=updater or HeuristicStateUpdater(),
    )
    task = engine.ingest_openresearcher_row(row)

    print("task_id:", task.task_id)
    print("state_updater:", "enabled" if updater else "disabled")
    print("state_version:", task.state.state_version)
    print("resolved_findings:", len(task.state.resolved_findings))
    print("working_hypotheses:", len(task.state.working_hypotheses))
    print("open_questions:", len(task.state.open_questions))
    print("deltas:")
    for delta in task.deltas:
        print(json.dumps(delta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
