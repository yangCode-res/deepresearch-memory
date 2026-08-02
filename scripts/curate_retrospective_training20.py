#!/usr/bin/env python3
"""Curate five complete multi-Loop replays into a balanced 20-sample set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(ROOT / "src"), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_working_state_dataset import apply_delta, validate_label, write_preview  # noqa: E402
from build_working_state_retrospective import (  # noqa: E402
    DURABLE_OPERATION_PATTERN,
    GLOBAL_CONCRETE_PATTERN,
    audit_record_causality,
    boundary_for_decision,
    completion_fallback,
    sanitize_contract_text,
    scrub_future_literals,
    subgoal_fallback,
    summarize_delta,
)


CURATION_OPERATION_PATTERN = re.compile(
    r"https?://|\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)\b|\bbrowser\.|"
    r"\b(?:search|query|open|view|visit|click|browse|"
    r"url|webpage|wikipedia|google|bing|source|search\s+result|snippet|doc\s*\d+)\b",
    re.IGNORECASE,
)

GENERIC_COMPLETION_PATTERN = re.compile(
    r"\bevidence resolves\b|\binformation dependency\s*\d*\b",
    re.IGNORECASE,
)

TEACHER_ARTIFACT_PATTERN = re.compile(
    r"requested values?|\bthe fatherthe\b|\bthe the\b|\bIranthe\b|"
    r"\bestablish evidence that\s+\(?s\)?(?=\s|\(|$)",
    re.IGNORECASE,
)


def normalize_curated_contract(value: Any, *, fallback: str) -> str:
    """Sanitize operations while retaining answer types such as 'website'."""

    raw = " ".join(str(value or "").strip().split()).rstrip(" .;；。")
    cleaned = sanitize_contract_text(raw, fallback=fallback)
    if raw and cleaned == fallback and not CURATION_OPERATION_PATTERN.search(raw):
        return raw
    return cleaned


def completion_from_subgoal(subgoal: str) -> str:
    return f"Evidence is sufficient to {subgoal[:1].lower() + subgoal[1:]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="append", required=True, type=Path)
    parser.add_argument("--segments", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--min-positive-retrieval", type=int, default=3)
    parser.add_argument("--include-qids", nargs="*", default=[])
    parser.add_argument("--contract-overrides", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def qid_of(row: dict[str, Any]) -> str:
    return str(row["source"]["qid"])


def trajectory_of(row: dict[str, Any]) -> str:
    return str(row["source"].get("trajectory_id") or qid_of(row))


def apply_contract_overrides(
    segment_by_qid: dict[str, dict[str, Any]], overrides: dict[str, Any]
) -> list[str]:
    """Apply explicit, reviewable text repairs without changing Loop boundaries."""

    applied: list[str] = []
    for qid, loop_overrides in overrides.items():
        qid = str(qid)
        segment = segment_by_qid.get(qid)
        if segment is None:
            raise ValueError(f"contract override references missing qid {qid}")
        if not isinstance(loop_overrides, dict) or not loop_overrides:
            raise ValueError(f"contract override for qid {qid} must name at least one Loop")
        loops = {
            str(loop["loop_number"]): loop
            for loop in segment["segmentation"]["loops"]
        }
        for loop_number, replacement in loop_overrides.items():
            loop = loops.get(str(loop_number))
            if loop is None:
                raise ValueError(f"contract override references missing qid {qid} Loop {loop_number}")
            if not isinstance(replacement, dict) or set(replacement) != {
                "subgoal",
                "completion_test",
            }:
                raise ValueError(
                    f"contract override for qid {qid} Loop {loop_number} must contain "
                    "subgoal and completion_test"
                )
            subgoal = " ".join(str(replacement["subgoal"]).split()).strip()
            completion = " ".join(str(replacement["completion_test"]).split()).strip()
            if not subgoal or not completion or TEACHER_ARTIFACT_PATTERN.search(
                f"{subgoal}\n{completion}"
            ):
                raise ValueError(f"contract override for qid {qid} Loop {loop_number} is invalid")
            loop["subgoal"] = subgoal
            loop["completion_test"] = completion
        applied.append(qid)
    return applied


def normalize_state_item(value: Any, *, fallback: str) -> str:
    raw = " ".join(str(value or "").strip().split())
    if (
        not raw
        or TEACHER_ARTIFACT_PATTERN.search(raw)
        or GENERIC_COMPLETION_PATTERN.search(raw)
    ):
        return fallback
    return sanitize_contract_text(raw, fallback=fallback)


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
                loop["subgoal"] = normalize_curated_contract(raw_goal, fallback=goal_fallback)
                if raw_goal.startswith("Establish information dependency"):
                    loop["subgoal"] = goal_fallback
                loop["completion_test"] = normalize_curated_contract(
                    loop["completion_test"], fallback=test_fallback
                )
                if GENERIC_COMPLETION_PATTERN.search(loop["completion_test"]):
                    loop["completion_test"] = completion_from_subgoal(loop["subgoal"])
        group.sort(key=lambda item: int(item["source"]["step_index"]))
        previous_after: dict[str, Any] | None = None
        current_global = (
            deepcopy(group[0]["input"].get("global_intent_state_before"))
            if group
            else None
        )
        activation_visible: dict[int, str] = {}
        loop_contracts = {
            int(loop.get("loop_number") or index): loop
            for index, loop in enumerate(
                (segment_record or {}).get("segmentation", {}).get("loops", []), start=1
            )
        }
        for row in group:
            if previous_after is not None:
                row["input"]["working_state_before"] = deepcopy(previous_after)
            if current_global is not None:
                row["input"]["global_intent_state_before"] = deepcopy(current_global)
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
            current_contract = loop_contracts.get(current_loop_number, {})
            current_raw = str(current_contract.get("subgoal") or decision.get("current_subgoal") or "")
            decision["current_subgoal"] = normalize_curated_contract(
                current_raw, fallback=current_fallback
            )
            if current_raw.startswith("Establish information dependency"):
                decision["current_subgoal"] = current_fallback
            decision["current_subgoal"] = scrub_future_literals(
                decision["current_subgoal"],
                visible_text=activation_visible[current_loop_number],
            )
            if decision.get("next_subgoal"):
                next_contract = loop_contracts.get(current_loop_number + 1, {})
                decision["next_subgoal"] = normalize_curated_contract(
                    next_contract.get("subgoal") or decision["next_subgoal"],
                    fallback="Establish the next distinct information dependency required by the user question",
                )
            after = target["working_state_after"]
            after_loop_number = int(str(after["loop_id"]).rsplit("_", 1)[-1])
            activation_visible.setdefault(after_loop_number, visible_text)
            after_fallback = subgoal_fallback(question, after_loop_number)
            after_contract = loop_contracts.get(after_loop_number, {})
            after_raw = str(after_contract.get("subgoal") or after.get("current_subgoal") or "")
            after["current_subgoal"] = normalize_curated_contract(
                after_raw, fallback=after_fallback
            )
            if after_raw.startswith("Establish information dependency"):
                after["current_subgoal"] = after_fallback
            after["current_subgoal"] = scrub_future_literals(
                after["current_subgoal"], visible_text=activation_visible[after_loop_number]
            )
            after["completion_test"] = normalize_curated_contract(
                after_contract.get("completion_test") or after.get("completion_test"),
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
                normalize_state_item(item, fallback=after["current_subgoal"])
                for item in after.get("open_aspects") or []
            ]
            after["evidence_gaps"] = [
                normalize_state_item(item, fallback=after["completion_test"])
                for item in after.get("evidence_gaps") or []
            ]
            after["next_direction"] = (
                f"Objective: {after['current_subgoal']}. Stop when: {after['completion_test']}."
            )
            action = str(decision.get("action") or "")
            delta = target.get("state_delta") if isinstance(target.get("state_delta"), dict) else {}
            operations = delta.get("operations") if isinstance(delta.get("operations"), dict) else None
            if operations is not None and action != "CONTINUE_CURRENT_LOOP":
                operations["completed_subgoal"] = decision["current_subgoal"]
                operations["add_confirmed_facts"] = [
                    item
                    for item in operations.get("add_confirmed_facts") or []
                    if not DURABLE_OPERATION_PATTERN.search(str(item))
                ]
                delta["summary"] = summarize_delta(operations)
            if current_global is not None and operations is not None:
                current_global = apply_delta(current_global, operations)
            previous_after = after


def validate_pool(
    rows: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[trajectory_of(row)].append(row)
    segment_map = {trajectory_of(item): item for item in segments}
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
            if CURATION_OPERATION_PATTERN.search(directional_text):
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
    pool_by_source: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.pool:
        for row in read_jsonl(path):
            pool_by_source[(trajectory_of(row), int(row["source"]["step_index"]))] = row
    segment_by_qid: dict[str, dict[str, Any]] = {}
    for path in args.segments:
        for row in read_jsonl(path):
            segment_by_qid[trajectory_of(row)] = row
    override_qids: list[str] = []
    if args.contract_overrides:
        raw_overrides = json.loads(args.contract_overrides.read_text(encoding="utf-8"))
        if not isinstance(raw_overrides, dict):
            raise SystemExit("--contract-overrides must contain a JSON object")
        try:
            override_qids = apply_contract_overrides(segment_by_qid, raw_overrides)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    pool = list(pool_by_source.values())
    segments = list(segment_by_qid.values())
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
    selected.sort(
        key=lambda item: (
            chosen_qids.index(trajectory_of(item)),
            int(item["source"]["step_index"]),
        )
    )
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"ws_train_{index:04d}"

    action_counts = Counter(row["target"]["loop_decision"]["action"] for row in selected)
    per_trajectory = Counter(trajectory_of(row) for row in selected)
    positive_retrieval = sum(
        bool(selected_memories(row))
        for row in selected
        if row["source"]["selection_role"] == "continue_after_switch"
    )
    selection_violations: list[str] = []
    expected_samples = args.questions * 4
    expected_actions = {
        "CONTINUE_CURRENT_LOOP": args.questions * 2,
        "SWITCH_LOOP": args.questions,
        "READY_TO_ANSWER": args.questions,
    }
    if len(selected) != expected_samples:
        selection_violations.append(
            f"selection does not contain exactly {expected_samples} samples"
        )
    if dict(action_counts) != expected_actions:
        selection_violations.append(f"unexpected action distribution: {dict(action_counts)}")
    if set(per_trajectory.values()) != {4} or len(per_trajectory) != args.questions:
        selection_violations.append(
            f"unexpected per-trajectory distribution: {dict(per_trajectory)}"
        )
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
        "chosen_trajectory_ids": chosen_qids,
        "chosen_qids": [qid_of(grouped[item][0]) for item in chosen_qids],
        "action_counts": dict(action_counts),
        "samples_per_trajectory": dict(per_trajectory),
        "positive_post_switch_retrieval_samples": positive_retrieval,
        "pool_samples": len(pool),
        "pool_trajectories": len(grouped),
        "eligible_trajectories": len(eligible),
        "rejected_pool_qids": sorted(invalid_qids),
        "contract_override_qids": override_qids,
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
