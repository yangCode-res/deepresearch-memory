#!/usr/bin/env python3
"""Curate five complete multi-Loop replays into a balanced 20-sample set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(ROOT / "src"), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_working_state_dataset import apply_delta, validate_label, write_preview  # noqa: E402
from build_working_state_retrospective import (  # noqa: E402
    GLOBAL_CONCRETE_PATTERN,
    audit_record_causality,
    boundary_for_decision,
    completion_fallback,
    sanitize_contract_text,
    scrub_future_literals,
    subgoal_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--min-positive-retrieval", type=int, default=3)
    parser.add_argument("--include-qids", nargs="*", default=[])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def qid_of(row: dict[str, Any]) -> str:
    return str(row["source"]["qid"])


def selected_memories(row: dict[str, Any]) -> list[str]:
    return [str(value) for value in row["target"]["retrieval"].get("relevant_memory_ids") or []]


def normalize_directional_contracts(
    grouped: dict[str, list[dict[str, Any]]],
    segment_map: dict[str, dict[str, Any]],
) -> None:
    """Apply the latest mechanical policy to an already-generated causal replay."""

    for qid, group in grouped.items():
        question = str(group[0]["input"].get("question") or "") if group else ""
        segment_record = segment_map.get(qid)
        if segment_record:
            for loop in segment_record["segmentation"]["loops"]:
                loop_number = int(loop.get("loop_number") or 1)
                goal_fallback = subgoal_fallback(question, loop_number)
                test_fallback = completion_fallback(question, loop_number)
                raw_goal = str(loop.get("subgoal") or "")
                loop["subgoal"] = sanitize_contract_text(raw_goal, fallback=goal_fallback)
                if raw_goal.startswith("Establish information dependency"):
                    loop["subgoal"] = goal_fallback
                loop["completion_test"] = sanitize_contract_text(
                    loop["completion_test"], fallback=test_fallback
                )
        group.sort(key=lambda item: int(item["source"]["step_index"]))
        previous_after: dict[str, Any] | None = None
        activation_visible: dict[int, str] = {}
        for row in group:
            if previous_after is not None:
                row["input"]["working_state_before"] = deepcopy(previous_after)
            visible_text = "\n".join(
                [
                    question,
                    *[
                        str(message.get("text") or "")
                        for message in row["input"].get("observed_messages") or []
                    ],
                ]
            )
            target = row["target"]
            decision = target["loop_decision"]
            current_loop_number = int(str(row["input"]["working_state_before"]["loop_id"]).rsplit("_", 1)[-1])
            activation_visible.setdefault(current_loop_number, visible_text)
            current_fallback = subgoal_fallback(question, current_loop_number)
            current_raw = str(decision.get("current_subgoal") or "")
            decision["current_subgoal"] = sanitize_contract_text(current_raw, fallback=current_fallback)
            if current_raw.startswith("Establish information dependency"):
                decision["current_subgoal"] = current_fallback
            decision["current_subgoal"] = scrub_future_literals(
                decision["current_subgoal"],
                visible_text=activation_visible[current_loop_number],
            )
            if decision.get("next_subgoal"):
                decision["next_subgoal"] = sanitize_contract_text(
                    decision["next_subgoal"],
                    fallback="Establish the next distinct information dependency required by the user question",
                )
            after = target["working_state_after"]
            after_loop_number = int(str(after["loop_id"]).rsplit("_", 1)[-1])
            activation_visible.setdefault(after_loop_number, visible_text)
            after_fallback = subgoal_fallback(question, after_loop_number)
            after_raw = str(after.get("current_subgoal") or "")
            after["current_subgoal"] = sanitize_contract_text(after_raw, fallback=after_fallback)
            if after_raw.startswith("Establish information dependency"):
                after["current_subgoal"] = after_fallback
            after["current_subgoal"] = scrub_future_literals(
                after["current_subgoal"], visible_text=activation_visible[after_loop_number]
            )
            after["completion_test"] = sanitize_contract_text(
                after.get("completion_test"),
                fallback=completion_fallback(question, after_loop_number),
            )
            after["completion_test"] = scrub_future_literals(
                after["completion_test"], visible_text=activation_visible[after_loop_number]
            )
            if decision.get("next_subgoal"):
                decision["next_subgoal"] = scrub_future_literals(
                    decision["next_subgoal"], visible_text=activation_visible[after_loop_number]
                )
            after["open_aspects"] = [
                sanitize_contract_text(item, fallback=after["current_subgoal"])
                for item in after.get("open_aspects") or []
            ]
            after["evidence_gaps"] = [
                sanitize_contract_text(item, fallback=after["completion_test"])
                for item in after.get("evidence_gaps") or []
            ]
            after["next_direction"] = (
                f"Objective: {after['current_subgoal']}. Stop when: {after['completion_test']}."
            )
            previous_after = after


def validate_pool(
    rows: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[qid_of(row)].append(row)
    segment_map = {qid_of(item): item for item in segments}
    violations: list[dict[str, Any]] = []
    normalize_directional_contracts(grouped, segment_map)

    for qid, group in grouped.items():
        group.sort(key=lambda item: int(item["source"]["step_index"]))
        segment_record = segment_map.get(qid)
        if not segment_record:
            violations.append({"qid": qid, "error": "missing retrospective segmentation"})
            continue
        decision_count = int(segment_record["decision_count"])
        steps = [int(item["source"]["step_index"]) for item in group]
        if steps != list(range(decision_count)):
            violations.append(
                {"qid": qid, "error": f"incomplete replay: steps={steps}, expected=0..{decision_count - 1}"}
            )
            continue
        segmentation = segment_record["segmentation"]
        for row in group:
            step = int(row["source"]["step_index"])
            expected_action = boundary_for_decision(segmentation, step)["action"]
            actual_action = row["target"]["loop_decision"]["action"]
            if expected_action != actual_action:
                violations.append(
                    {"qid": qid, "step": step, "error": "sample action disagrees with segmentation"}
                )
            causal_errors = audit_record_causality(row)
            for error in causal_errors:
                violations.append({"qid": qid, "step": step, "error": error})
            before = row["input"]["working_state_before"]
            loop_number = int(str(before["loop_id"]).rsplit("_", 1)[-1])
            seen_ids = set(row["input"].get("current_loop_evidence_ids") or [])
            allowed_ids = {
                str(memory["memory_id"])
                for memory in row["input"].get("available_cross_loop_memories") or []
            }
            if row["target"].get("cross_loop_memory"):
                allowed_ids.add(str(row["target"]["cross_loop_memory"]["memory_id"]))
            try:
                validate_label(
                    row["target"],
                    seen_message_ids=seen_ids,
                    allowed_memory_ids=allowed_ids,
                    loop_number=loop_number,
                )
            except ValueError as exc:
                violations.append({"qid": qid, "step": step, "error": str(exc)})

            target = row["target"]
            directional_text = "\n".join(
                [
                    str(target["loop_decision"].get("current_subgoal") or ""),
                    str(target["loop_decision"].get("next_subgoal") or ""),
                    str(target["working_state_after"].get("completion_test") or ""),
                    str(target["working_state_after"].get("next_direction") or ""),
                    *[str(value) for value in target["working_state_after"].get("open_aspects") or []],
                    *[str(value) for value in target["working_state_after"].get("evidence_gaps") or []],
                ]
            )
            if GLOBAL_CONCRETE_PATTERN.search(directional_text):
                violations.append(
                    {"qid": qid, "step": step, "error": "tool/source/domain text in directional contract"}
                )

        for left, right in zip(group, group[1:]):
            if left["target"]["working_state_after"] != right["input"]["working_state_before"]:
                violations.append({"qid": qid, "error": "Working State replay chain is discontinuous"})
            expected_global = apply_delta(
                left["input"]["global_intent_state_before"],
                left["target"]["state_delta"]["operations"],
            )
            if expected_global != right["input"]["global_intent_state_before"]:
                violations.append({"qid": qid, "error": "Global State replay chain is discontinuous"})
            memory = left["target"].get("cross_loop_memory")
            if memory:
                next_ids = {
                    str(item["memory_id"])
                    for item in right["input"].get("available_cross_loop_memories") or []
                }
                if str(memory["memory_id"]) not in next_ids:
                    violations.append({"qid": qid, "error": "closed-Loop memory missing from next input"})

    return grouped, segment_map, violations


def choose_four(group: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    ordered = sorted(group, key=lambda item: int(item["source"]["step_index"]))
    switches = [item for item in ordered if item["target"]["loop_decision"]["action"] == "SWITCH_LOOP"]
    ready = [item for item in ordered if item["target"]["loop_decision"]["action"] == "READY_TO_ANSWER"]
    continues = [
        item for item in ordered if item["target"]["loop_decision"]["action"] == "CONTINUE_CURRENT_LOOP"
    ]
    if not switches or len(ready) != 1 or len(continues) < 2:
        raise ValueError("trajectory does not provide SWITCH + READY + two CONTINUE samples")
    switch = switches[0]
    switch_step = int(switch["source"]["step_index"])
    before = [item for item in continues if int(item["source"]["step_index"]) < switch_step]
    after = [item for item in continues if int(item["source"]["step_index"]) > switch_step]
    positive_after = [item for item in after if selected_memories(item)]

    chosen_continues: list[tuple[dict[str, Any], str]] = []
    if before:
        chosen_continues.append((before[-1], "continue_before_switch"))
    post = positive_after[0] if positive_after else (after[0] if after else None)
    if post is not None:
        chosen_continues.append((post, "continue_after_switch"))
    for item in continues:
        if len(chosen_continues) >= 2:
            break
        if all(item is not selected for selected, _ in chosen_continues):
            role = "continue_after_switch" if int(item["source"]["step_index"]) > switch_step else "continue_before_switch"
            chosen_continues.append((item, role))
    if len(chosen_continues) != 2:
        raise ValueError("could not choose two distinct CONTINUE samples")
    return [
        *chosen_continues,
        (switch, "switch_boundary"),
        (ready[0], "ready_boundary"),
    ]


def positive_retrieval_after_switch(group: list[dict[str, Any]]) -> bool:
    switch_steps = [
        int(item["source"]["step_index"])
        for item in group
        if item["target"]["loop_decision"]["action"] == "SWITCH_LOOP"
    ]
    if not switch_steps:
        return False
    first_switch = switch_steps[0]
    return any(
        int(item["source"]["step_index"]) > first_switch and selected_memories(item)
        for item in group
    )


def main() -> None:
    args = parse_args()
    pool = read_jsonl(args.pool)
    segments = read_jsonl(args.segments)
    grouped, segment_map, violations = validate_pool(pool, segments)
    invalid_qids = {str(item["qid"]) for item in violations if item.get("qid") is not None}
    global_violations = [item for item in violations if item.get("qid") is None]
    if global_violations:
        raise SystemExit(
            json.dumps({"pool_violations": global_violations[:30]}, ensure_ascii=False, indent=2)
        )

    eligible: list[str] = []
    for qid, group in grouped.items():
        if qid in invalid_qids:
            continue
        actions = Counter(item["target"]["loop_decision"]["action"] for item in group)
        if actions["SWITCH_LOOP"] >= 1 and actions["READY_TO_ANSWER"] == 1 and actions["CONTINUE_CURRENT_LOOP"] >= 2:
            eligible.append(qid)
    eligible.sort(
        key=lambda qid: (
            not positive_retrieval_after_switch(grouped[qid]),
            len(grouped[qid]),
            qid,
        )
    )
    if args.include_qids:
        chosen_qids = [str(qid) for qid in args.include_qids]
        if len(chosen_qids) != args.questions or len(set(chosen_qids)) != args.questions:
            raise SystemExit("--include-qids must name exactly --questions distinct qids")
        unavailable = [qid for qid in chosen_qids if qid not in eligible]
        if unavailable:
            raise SystemExit(f"requested qids are not eligible: {unavailable}")
    else:
        chosen_qids = eligible[: args.questions]
    if len(chosen_qids) != args.questions:
        raise SystemExit(f"only {len(chosen_qids)} eligible complete multi-Loop trajectories")

    selected: list[dict[str, Any]] = []
    for qid in chosen_qids:
        chosen = choose_four(grouped[qid])
        for row, role in sorted(chosen, key=lambda item: int(item[0]["source"]["step_index"])):
            copied = deepcopy(row)
            copied["source"]["selection_role"] = role
            copied["teacher"]["curation"] = "balanced_complete_multiloop_v1"
            selected.append(copied)
    selected.sort(key=lambda item: (chosen_qids.index(qid_of(item)), int(item["source"]["step_index"])))
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"ws_train_{index:04d}"

    action_counts = Counter(row["target"]["loop_decision"]["action"] for row in selected)
    per_qid = Counter(qid_of(row) for row in selected)
    positive_retrieval = sum(
        bool(selected_memories(row))
        for row in selected
        if row["source"]["selection_role"] == "continue_after_switch"
    )
    selection_violations: list[str] = []
    if len(selected) != 20:
        selection_violations.append("selection does not contain exactly 20 samples")
    if dict(action_counts) != {
        "CONTINUE_CURRENT_LOOP": 10,
        "SWITCH_LOOP": 5,
        "READY_TO_ANSWER": 5,
    }:
        selection_violations.append(f"unexpected action distribution: {dict(action_counts)}")
    if set(per_qid.values()) != {4} or len(per_qid) != 5:
        selection_violations.append(f"unexpected per-question distribution: {dict(per_qid)}")
    if positive_retrieval < args.min_positive_retrieval:
        selection_violations.append(
            f"only {positive_retrieval} selected post-switch CONTINUE samples retrieve prior memory"
        )
    if selection_violations:
        raise SystemExit(json.dumps({"selection_violations": selection_violations}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_preview(args.output.with_suffix(".preview.md"), selected, count=len(selected))
    chosen_segments = [segment_map[qid] for qid in chosen_qids]
    with args.output.with_suffix(".segments.jsonl").open("w", encoding="utf-8") as handle:
        for item in chosen_segments:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "valid": True,
        "samples": len(selected),
        "chosen_qids": chosen_qids,
        "action_counts": dict(action_counts),
        "samples_per_qid": dict(per_qid),
        "positive_post_switch_retrieval_samples": positive_retrieval,
        "pool_samples": len(pool),
        "pool_trajectories": len(grouped),
        "eligible_trajectories": len(eligible),
        "rejected_pool_qids": sorted(invalid_qids),
        "pool_violations": violations,
        "selection_violations": [],
        "output": str(args.output),
    }
    args.output.with_suffix(".qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
