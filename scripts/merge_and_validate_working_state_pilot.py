#!/usr/bin/env python3
"""Merge pilot shards and run deployment-contract QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_working_state_dataset import (  # noqa: E402
    CONCRETE_ACTION_PATTERN,
    validate_label,
    write_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-source", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_source)
    rows: list[dict[str, Any]] = []
    for path in args.input:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            source_key = f"{row['source']['qid']}:{row['source']['step_index']}"
            if source_key not in excluded:
                rows.append(row)

    violations: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    unique_sources: set[tuple[str, int]] = set()
    non_bootstrap_continue = 0
    subgoal_drift = 0
    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"ws_v2_pilot_{index:04d}"
        target = row["target"]
        action = target["loop_decision"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
        source_key = (str(row["source"]["qid"]), int(row["source"]["step_index"]))
        if source_key in unique_sources:
            violations.append({"sample": row["sample_id"], "error": "duplicate source decision point"})
        unique_sources.add(source_key)
        before = row["input"]["working_state_before"]
        before_loop = int(str(before["loop_id"]).rsplit("_", 1)[-1])
        seen_ids = set(row["input"].get("current_loop_evidence_ids") or [])
        allowed_ids = {
            str(memory["memory_id"])
            for memory in row["input"].get("available_cross_loop_memories") or []
        }
        if target.get("cross_loop_memory"):
            allowed_ids.add(str(target["cross_loop_memory"]["memory_id"]))
        try:
            validate_label(
                target,
                seen_message_ids=seen_ids,
                allowed_memory_ids=allowed_ids,
                loop_number=before_loop,
            )
        except ValueError as exc:
            violations.append({"sample": row["sample_id"], "error": str(exc)})
        direction = str(target["working_state_after"].get("next_direction") or "")
        if CONCRETE_ACTION_PATTERN.search(direction):
            violations.append({"sample": row["sample_id"], "error": "concrete action in direction"})
        if action == "CONTINUE_CURRENT_LOOP" and row["source"]["step_index"] > 0:
            non_bootstrap_continue += 1
            if before["current_subgoal"] != target["working_state_after"]["current_subgoal"]:
                subgoal_drift += 1
                violations.append({"sample": row["sample_id"], "error": "subgoal drift under CONTINUE"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_preview(args.output.with_suffix(".preview.md"), rows, count=len(rows))
    report = {
        "samples": len(rows),
        "action_counts": action_counts,
        "unique_questions": len({str(row["source"]["qid"]) for row in rows}),
        "unique_source_points": len(unique_sources),
        "non_bootstrap_continue": non_bootstrap_continue,
        "subgoal_drift_under_continue": subgoal_drift,
        "violations": violations,
        "valid": not violations,
        "manually_excluded_sources": sorted(excluded),
    }
    args.output.with_suffix(".qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
