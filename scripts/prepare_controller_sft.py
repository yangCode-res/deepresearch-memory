#!/usr/bin/env python3
"""Convert curated controller replay rows into trajectory-disjoint chat SFT splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cl_gism.controller_sft import training_messages, trajectory_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["target"]["loop_decision"]["action"] for row in rows))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_ratio < 0.5:
        raise SystemExit("--validation-ratio must be between 0 and 0.5")
    rows = read_jsonl(args.input)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tid = trajectory_id(row)
        if not tid:
            raise SystemExit("every row must contain source.trajectory_id or source.qid")
        grouped[tid].append(row)

    trajectory_ids = sorted(grouped)
    random.Random(args.seed).shuffle(trajectory_ids)
    validation_count = max(1, round(len(trajectory_ids) * args.validation_ratio))
    validation_ids = set(trajectory_ids[:validation_count])

    prepared: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for tid in trajectory_ids:
        split = "validation" if tid in validation_ids else "train"
        for row in sorted(grouped[tid], key=lambda item: int(item["source"]["step_index"])):
            prepared[split].append(
                {
                    "sample_id": row["sample_id"],
                    "trajectory_id": tid,
                    "qid": row["source"].get("qid"),
                    "step_index": row["source"]["step_index"],
                    "action": row["target"]["loop_decision"]["action"],
                    "messages": training_messages(row),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", prepared["train"])
    write_jsonl(args.output_dir / "validation.jsonl", prepared["validation"])
    report = {
        "source": str(args.input),
        "seed": args.seed,
        "trajectory_disjoint": True,
        "trajectories": len(trajectory_ids),
        "train_trajectories": len(trajectory_ids) - validation_count,
        "validation_trajectories": validation_count,
        "train_samples": len(prepared["train"]),
        "validation_samples": len(prepared["validation"]),
        "train_actions": action_counts(
            [row for tid in trajectory_ids if tid not in validation_ids for row in grouped[tid]]
        ),
        "validation_actions": action_counts(
            [row for tid in trajectory_ids if tid in validation_ids for row in grouped[tid]]
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
