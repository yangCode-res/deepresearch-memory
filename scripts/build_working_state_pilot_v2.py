#!/usr/bin/env python3
"""Two-stage, action-balanced Working State pilot dataset builder."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(ROOT / "src"), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cl_gism.llm_planner import OpenAIChatJSONClient  # noqa: E402
from build_working_state_dataset import (  # noqa: E402
    ACTIONS,
    CONCRETE_ACTION_PATTERN,
    DELTA_FIELDS,
    GAIN_LEVELS,
    apply_delta,
    decision_steps,
    initial_global_state,
    initial_working_state,
    iter_rows,
    validate_label,
    write_preview,
)


BOUNDARY_SYSTEM_PROMPT = """You are only the Loop boundary judge for a deep-research memory system.
Return valid JSON only and do not write Working State, StateDelta, memory, tools, queries, or chain-of-thought.

A Loop is one coherent, locally decidable subgoal with one observable completion test. Tool choice, query
wording, source changes, and one failed search are not boundaries. CONTINUE only while the committed subgoal
and completion test remain unchanged. SWITCH when that subgoal is resolved, refuted, saturated, superseded,
or the next call must pursue an independently decidable dependency. READY only when an exact answer is stable,
core claims have citable support, and another search has low expected information gain.

The committed subgoal is system-owned: copy it exactly when supplied. Never silently rename it under CONTINUE.
Subgoals and completion tests describe information to establish, never an exact search/open/view action,
named website, URL, or tool. On the first step, establish a narrow immediate subgoal, not the whole user
question; the first action must be CONTINUE unless the available evidence already supports READY. A SWITCH
requires a genuinely different next subgoal and completion test."""


STATE_SYSTEM_PROMPT = """You write semantic state content after a separate judge has fixed the Loop action.
Return valid JSON only. Do not reconsider or change fixed_action. Do not reveal chain-of-thought.

Use only supplied messages and prior memories. Search snippets are provisional. Keep hypotheses distinct from
confirmed evidence. Cite only supplied msg_NNNN IDs. Working State describes information objectives and
observable stop conditions; never prescribe an exact query, named website, URL, browser tool, open/view/search
action, or mandatory action. For CONTINUE, durable_update must be empty and loop_memory must be null. For
SWITCH or READY, summarize only durable findings from the closing loop. Select only allowed memory IDs."""


BOUNDARY_KEYS = {
    "action",
    "reason",
    "current_subgoal",
    "current_completion_test",
    "next_subgoal",
    "next_completion_test",
    "outcome",
    "boundary_basis",
    "confidence",
}
UPDATE_KEYS = {
    "progress_summary",
    "resolved_aspects",
    "open_aspects",
    "key_evidence",
    "candidate_answer",
    "active_hypotheses",
    "failed_strategies",
    "evidence_gaps",
    "answer_stable",
    "evidence_sufficient",
    "expected_information_gain",
}


def boundary_contract() -> dict[str, Any]:
    return {
        "action": "CONTINUE_CURRENT_LOOP|SWITCH_LOOP|READY_TO_ANSWER",
        "reason": "short evidence-grounded reason",
        "current_subgoal": "copy committed subgoal, or establish a narrow one on first step",
        "current_completion_test": "copy committed test, or establish one on first step",
        "next_subgoal": "non-empty only for SWITCH_LOOP",
        "next_completion_test": "non-empty only for SWITCH_LOOP",
        "outcome": "IN_PROGRESS|RESOLVED|REFUTED|BLOCKED|SUPERSEDED",
        "boundary_basis": "NONE|SUBGOAL_COMPLETED|SUBGOAL_CHANGED|CANDIDATE_CHANGED|BLOCKED_OR_SATURATED|PHASE_TRANSITION|TASK_COMPLETE",
        "confidence": 0.8,
    }


def validate_boundary(
    raw: dict[str, Any], *, committed_subgoal: str, committed_completion_test: str, first_step: bool
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != BOUNDARY_KEYS:
        raise ValueError("boundary response has missing or extra fields")
    action = str(raw["action"] or "").upper()
    if action not in ACTIONS:
        raise ValueError("invalid boundary action")
    current = str(raw["current_subgoal"] or "").strip()
    completion = str(raw["current_completion_test"] or "").strip()
    next_subgoal = str(raw["next_subgoal"] or "").strip()
    next_test = str(raw["next_completion_test"] or "").strip()
    # These are system-owned fields. Do not spend repair calls asking the
    # teacher to reproduce them byte-for-byte.
    if committed_subgoal:
        current = committed_subgoal
    if committed_completion_test:
        completion = committed_completion_test
    # Before a contract exists, models sometimes describe initialization as a
    # switch from the generic bootstrap state. Normalize that mechanically.
    if first_step and action == "SWITCH_LOOP":
        current = next_subgoal or current
        completion = next_test or completion
        action = "CONTINUE_CURRENT_LOOP"
        next_subgoal = ""
        next_test = ""
        raw["outcome"] = "IN_PROGRESS"
        raw["boundary_basis"] = "NONE"
    if not current or not completion:
        raise ValueError("current subgoal and completion test are required")
    outcome = str(raw["outcome"] or "").upper()
    basis = str(raw["boundary_basis"] or "").upper()
    # action is the learned judgment; dependent enum fields are mechanical.
    if action == "CONTINUE_CURRENT_LOOP":
        outcome, basis, next_subgoal, next_test = "IN_PROGRESS", "NONE", "", ""
    elif action == "READY_TO_ANSWER":
        outcome, basis, next_subgoal, next_test = "RESOLVED", "TASK_COMPLETE", "", ""
    elif action == "SWITCH_LOOP":
        if outcome == "IN_PROGRESS":
            outcome = "RESOLVED"
        if basis == "NONE":
            basis = "SUBGOAL_COMPLETED" if outcome == "RESOLVED" else "SUBGOAL_CHANGED"
    policy_text = "\n".join([current, completion, next_subgoal, next_test])
    if CONCRETE_ACTION_PATTERN.search(policy_text):
        raise ValueError(
            "subgoals and completion tests must describe information outcomes, not concrete actions, "
            "named websites, URLs, or tools"
        )
    if action == "CONTINUE_CURRENT_LOOP" and (
        outcome != "IN_PROGRESS" or basis != "NONE" or next_subgoal or next_test
    ):
        raise ValueError("CONTINUE requires IN_PROGRESS, NONE, and no next subgoal")
    if action == "SWITCH_LOOP" and (
        outcome == "IN_PROGRESS"
        or basis == "NONE"
        or not next_subgoal
        or not next_test
        or " ".join(current.casefold().split()) == " ".join(next_subgoal.casefold().split())
    ):
        raise ValueError("SWITCH requires a terminal outcome and distinct next work-unit contract")
    if action == "READY_TO_ANSWER" and (
        outcome != "RESOLVED" or basis != "TASK_COMPLETE" or next_subgoal or next_test
    ):
        raise ValueError("READY requires RESOLVED, TASK_COMPLETE, and no next subgoal")
    raw.update(
        {
            "action": action,
            "current_subgoal": current,
            "current_completion_test": completion,
            "next_subgoal": next_subgoal,
            "next_completion_test": next_test,
            "outcome": outcome,
            "boundary_basis": basis,
        }
    )
    return raw


def call_and_repair(
    client: OpenAIChatJSONClient,
    *,
    system_prompt: str,
    payload: dict[str, Any],
    validator: Any,
    max_attempts: int = 2,
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    error = ""
    for attempt in range(max_attempts):
        prompt_payload = payload if not attempt else {
            "validation_error": error,
            "invalid_response": raw,
            "instruction": "Repair the complete object without changing the fixed decision.",
            "original_input": payload,
        }
        raw = client.complete_json(system_prompt, json.dumps(prompt_payload, ensure_ascii=False))
        try:
            return validator(raw)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(f"invalid after repair attempts: {error}")


def judge_boundary(
    client: OpenAIChatJSONClient,
    *,
    question: str,
    global_state: dict[str, Any],
    working_state: dict[str, Any],
    prior_memories: list[dict[str, Any]],
    observed_messages: list[dict[str, Any]],
    first_step: bool,
) -> dict[str, Any]:
    committed_subgoal = "" if first_step else str(working_state["current_subgoal"])
    committed_test = "" if first_step else str(working_state["completion_test"])
    boundary_state = {
        name: working_state.get(name)
        for name in (
            "current_subgoal",
            "completion_test",
            "progress_summary",
            "resolved_aspects",
            "open_aspects",
            "candidate_answer",
            "answer_stable",
            "evidence_sufficient",
            "expected_information_gain",
        )
    }
    payload = {
        "question": question,
        "committed_loop_contract": {
            "current_subgoal": committed_subgoal,
            "completion_test": committed_test,
        },
        "working_state_before": boundary_state,
        "observed_messages": observed_messages,
        "available_cross_loop_memories": compact_memories(prior_memories),
        "required_output": boundary_contract(),
    }
    return call_and_repair(
        client,
        system_prompt=BOUNDARY_SYSTEM_PROMPT,
        payload=payload,
        validator=lambda raw: validate_boundary(
            raw,
            committed_subgoal=committed_subgoal,
            committed_completion_test=committed_test,
            first_step=first_step,
        ),
    )


def semantic_contract(action: str, loop_number: int) -> dict[str, Any]:
    empty_ops = {name: ([] if name != "completed_subgoal" else "") for name in DELTA_FIELDS}
    return {
        "working_state_update": {
            "progress_summary": "at most two sentences",
            "resolved_aspects": [],
            "open_aspects": [],
            "key_evidence": ["fact with msg_NNNN source coordinate"],
            "candidate_answer": "",
            "active_hypotheses": [],
            "failed_strategies": [],
            "evidence_gaps": [],
            "answer_stable": action == "READY_TO_ANSWER",
            "evidence_sufficient": action == "READY_TO_ANSWER",
            "expected_information_gain": "LOW" if action == "READY_TO_ANSWER" else "HIGH|MEDIUM|LOW",
        },
        "durable_update": empty_ops,
        "loop_memory": (
            None
            if action == "CONTINUE_CURRENT_LOOP"
            else {
                "memory_id": f"memory_loop_{loop_number:03d}",
                "summary": "durable handoff summary",
                "durable_findings": [],
                "rejected_leads": [],
                "unresolved_questions": [],
                "evidence_ids": ["msg_NNNN"],
            }
        ),
        "retrieval": {
            "query": "semantic memory query",
            "relevant_memory_ids": [],
            "reason": "short reason",
        },
        "next_loop_setup": (
            {
                "evidence_gaps": [],
            }
            if action == "SWITCH_LOOP"
            else None
        ),
    }


def build_target(
    raw: dict[str, Any],
    *,
    boundary: dict[str, Any],
    working_before: dict[str, Any],
    loop_number: int,
    seen_message_ids: set[str],
    allowed_memory_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "working_state_update", "durable_update", "loop_memory", "retrieval", "next_loop_setup"
    }:
        raise ValueError("semantic response has missing or extra fields")
    update = raw["working_state_update"]
    if not isinstance(update, dict) or set(update) != UPDATE_KEYS:
        raise ValueError("working_state_update has missing or extra fields")
    action = boundary["action"]
    if action == "READY_TO_ANSWER":
        update["answer_stable"] = True
        update["evidence_sufficient"] = True
        update["expected_information_gain"] = "LOW"
    # The controller's directional policy is deterministic from the validated
    # work-unit contract. This avoids a second model restating it as a tool
    # action and then consuming repair tokens.
    update["next_direction"] = (
        f"Objective: {boundary['current_subgoal'].rstrip('.')}. "
        f"Stop when: {boundary['current_completion_test'].rstrip('.')}."
    )
    for name in ("open_aspects", "evidence_gaps"):
        values = update.get(name) if isinstance(update.get(name), list) else []
        update[name] = [
            str(item) for item in values
            if str(item).strip() and not CONCRETE_ACTION_PATTERN.search(str(item))
        ][:8]
    if not update["evidence_sufficient"] and not (update["open_aspects"] or update["evidence_gaps"]):
        update["evidence_gaps"] = [
            f"Evidence has not yet satisfied: {boundary['current_completion_test']}"
        ]
    current = deepcopy(working_before)
    current.update(update)
    current.update(
        {
            "loop_id": f"loop_{loop_number:03d}",
            "status": "COMPLETED" if action == "READY_TO_ANSWER" else "IN_PROGRESS",
            "current_subgoal": boundary["current_subgoal"],
            "completion_test": boundary["current_completion_test"],
        }
    )
    operations = raw["durable_update"]
    if not isinstance(operations, dict) or set(operations) != DELTA_FIELDS:
        raise ValueError("durable_update has missing or extra fields")
    if action == "CONTINUE_CURRENT_LOOP":
        operations = {name: ([] if name != "completed_subgoal" else "") for name in DELTA_FIELDS}
        memory = None
        working_after = current
    else:
        operations["completed_subgoal"] = boundary["current_subgoal"]
        memory = raw["loop_memory"]
        if not isinstance(memory, dict):
            raise ValueError("terminal loop requires loop_memory")
        memory["memory_id"] = f"memory_loop_{loop_number:03d}"
        if action == "SWITCH_LOOP":
            setup = raw["next_loop_setup"]
            if not isinstance(setup, dict):
                raise ValueError("SWITCH requires next_loop_setup")
            working_after = initial_working_state()
            working_after.update(
                {
                    "loop_id": f"loop_{loop_number + 1:03d}",
                    "current_subgoal": boundary["next_subgoal"],
                    "completion_test": boundary["next_completion_test"],
                    "progress_summary": "New loop initialized from the previous loop handoff.",
                    "open_aspects": list(setup.get("evidence_gaps") or [boundary["next_subgoal"]])[:8],
                    "next_direction": (
                        f"Objective: {boundary['next_subgoal'].rstrip('.')}. "
                        f"Stop when: {boundary['next_completion_test'].rstrip('.')}."
                    ),
                    "evidence_gaps": list(setup.get("evidence_gaps") or [])[:8],
                }
            )
        else:
            working_after = current
    retrieval = raw["retrieval"] if isinstance(raw["retrieval"], dict) else {}
    target = {
        "working_state_after": working_after,
        "loop_decision": {
            "action": action,
            "reason": boundary["reason"],
            "current_subgoal": boundary["current_subgoal"],
            "next_subgoal": boundary["next_subgoal"],
            "outcome": boundary["outcome"],
            "boundary_basis": boundary["boundary_basis"],
        },
        "state_delta": {
            "mode": "NOOP" if action == "CONTINUE_CURRENT_LOOP" else "APPLY",
            "summary": "" if action == "CONTINUE_CURRENT_LOOP" else str(memory.get("summary") or ""),
            "operations": operations,
        },
        "cross_loop_memory": memory,
        "retrieval": {
            "query": str(retrieval.get("query") or ""),
            "relevant_memory_ids": list(retrieval.get("relevant_memory_ids") or []),
            "reason": str(retrieval.get("reason") or ""),
        },
    }
    current_memory_id = f"memory_loop_{loop_number:03d}" if memory else None
    allowed = set(allowed_memory_ids)
    if current_memory_id:
        allowed.add(current_memory_id)
    return validate_label(
        target,
        seen_message_ids=seen_message_ids,
        allowed_memory_ids=allowed,
        loop_number=loop_number,
    )


def write_semantic_state(
    client: OpenAIChatJSONClient,
    *,
    question: str,
    global_state: dict[str, Any],
    working_state: dict[str, Any],
    prior_memories: list[dict[str, Any]],
    observed_messages: list[dict[str, Any]],
    boundary: dict[str, Any],
    loop_number: int,
    seen_message_ids: set[str],
) -> dict[str, Any]:
    action = boundary["action"]
    allowed_memory_ids = {str(memory["memory_id"]) for memory in prior_memories}
    if action != "CONTINUE_CURRENT_LOOP":
        allowed_memory_ids.add(f"memory_loop_{loop_number:03d}")
    payload = {
        "question": question,
        "working_state_before": working_state,
        "observed_messages": observed_messages,
        "current_loop_evidence_ids": sorted(seen_message_ids),
        "available_cross_loop_memories": compact_memories(prior_memories),
        "fixed_boundary_decision": boundary,
        "allowed_memory_ids": sorted(allowed_memory_ids),
        "required_output": semantic_contract(action, loop_number),
    }
    return call_and_repair(
        client,
        system_prompt=STATE_SYSTEM_PROMPT,
        payload=payload,
        validator=lambda raw: build_target(
            raw,
            boundary=boundary,
            working_before=working_state,
            loop_number=loop_number,
            seen_message_ids=seen_message_ids,
            allowed_memory_ids=allowed_memory_ids,
        ),
    )


def compact_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": memory.get("memory_id"),
            "summary": str(memory.get("summary") or "")[:600],
            "durable_findings": list(memory.get("durable_findings") or [])[:4],
            "unresolved_questions": list(memory.get("unresolved_questions") or [])[:3],
        }
        for memory in memories[-6:]
    ]


def compact_observed_messages(
    messages: list[dict[str, Any]], *, tool_limit: int, assistant_limit: int
) -> list[dict[str, Any]]:
    """Bound duplicated context sent to the two teacher stages."""

    compacted: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        limit = tool_limit if item.get("role") == "tool" else assistant_limit
        text = str(item.get("text") or "")
        item["text"] = text[:limit]
        item["truncated"] = bool(item.get("truncated")) or len(text) > limit
        compacted.append(item)
    return compacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--continue-target", type=int, default=10)
    parser.add_argument("--switch-target", type=int, default=6)
    parser.add_argument("--ready-target", type=int, default=4)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=40)
    parser.add_argument("--max-trajectory-decision-points", type=int, default=12)
    parser.add_argument("--min-trajectory-decision-points", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model", default=os.getenv("CL_GISM_DATA_MODEL", "mimo-v2.5"))
    parser.add_argument("--base-url", default=os.getenv("CL_GISM_CONTROLLER_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("CL_GISM_CONTROLLER_API_KEY"))
    parser.add_argument("--preview-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    import glob

    args = parse_args()
    if not args.api_key or not args.base_url:
        raise SystemExit("controller API configuration is required")
    paths = glob.glob(args.input_glob)
    if not paths:
        raise SystemExit("no input parquet files matched")
    quotas = {
        "CONTINUE_CURRENT_LOOP": args.continue_target,
        "SWITCH_LOOP": args.switch_target,
        "READY_TO_ANSWER": args.ready_target,
    }
    total_target = sum(quotas.values())
    counts = {action: 0 for action in quotas}
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    processed = 0
    skipped_long_trajectories = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boundary_client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=300,
        max_tokens=1024,
    )
    state_client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=300,
        max_tokens=3072,
    )
    print(f"[pilot-v2] model={args.model} quotas={quotas}", flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in iter_rows(paths, seed=args.seed):
            if len(records) >= total_target:
                break
            if str(row.get("status") or "").lower() not in {"success", "completed", "ok"}:
                continue
            steps = decision_steps(row.get("messages") or [])[: args.max_steps_per_trajectory]
            if not steps:
                continue
            if len(steps) < args.min_trajectory_decision_points:
                continue
            if len(steps) > args.max_trajectory_decision_points:
                skipped_long_trajectories += 1
                continue
            question = str(row.get("question") or "").strip()
            global_state = initial_global_state(question)
            working_state = initial_working_state()
            memories: list[dict[str, Any]] = []
            loop_number = 1
            seen_ids: set[str] = set()
            first_step = True
            for step in steps:
                processed += 1
                boundary_messages = compact_observed_messages(
                    step["observed_messages"], tool_limit=2500, assistant_limit=600
                )
                state_messages = compact_observed_messages(
                    step["observed_messages"], tool_limit=4000, assistant_limit=1000
                )
                seen_ids.update(message["message_id"] for message in state_messages)
                try:
                    boundary = judge_boundary(
                        boundary_client,
                        question=question,
                        global_state=global_state,
                        working_state=working_state,
                        prior_memories=memories,
                        observed_messages=boundary_messages,
                        first_step=first_step,
                    )
                    if (
                        boundary["action"] == "READY_TO_ANSWER"
                        and counts["READY_TO_ANSWER"] >= quotas["READY_TO_ANSWER"]
                    ):
                        break
                    target = write_semantic_state(
                        state_client,
                        question=question,
                        global_state=global_state,
                        working_state=working_state,
                        prior_memories=memories,
                        observed_messages=state_messages,
                        boundary=boundary,
                        loop_number=loop_number,
                        seen_message_ids=seen_ids,
                    )
                except Exception as exc:
                    errors.append({"qid": row.get("qid"), "step": step["step_index"], "error": str(exc)})
                    print(f"[pilot-v2] abandon qid={row.get('qid')} step={step['step_index']}: {exc}", flush=True)
                    break
                action = boundary["action"]
                if counts[action] < quotas[action]:
                    record = {
                        "sample_id": f"ws_v2_{len(records) + 1:04d}",
                        "source": {
                            "dataset": "OpenResearcher/OpenResearcher-Dataset",
                            "qid": row.get("qid"),
                            "step_index": step["step_index"],
                            "prefix_end_message_index": step["prefix_end_message_index"],
                        },
                        "teacher": {"model": args.model, "label_type": "two_stage_offline_teacher"},
                        "input": {
                            "question": question,
                            "global_intent_state_before": deepcopy(global_state),
                            "working_state_before": deepcopy(working_state),
                            "observed_messages": state_messages,
                            "current_loop_evidence_ids": sorted(seen_ids),
                            "available_cross_loop_memories": deepcopy(memories[-8:]),
                        },
                        "target": target,
                    }
                    records.append(record)
                    counts[action] += 1
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        f"[pilot-v2] saved={len(records)}/{total_target} counts={counts} "
                        f"qid={row.get('qid')} step={step['step_index']} action={action}",
                        flush=True,
                    )
                global_state = apply_delta(global_state, target["state_delta"]["operations"])
                memory = target["cross_loop_memory"]
                if isinstance(memory, dict):
                    memories.append(memory)
                working_state = target["working_state_after"]
                first_step = False
                if action == "SWITCH_LOOP":
                    loop_number += 1
                    seen_ids = set()
                if action == "READY_TO_ANSWER":
                    break
    preview = args.output.with_suffix(".preview.md")
    write_preview(preview, records, count=args.preview_count)
    report = {
        "requested": quotas,
        "generated": counts,
        "valid_samples": len(records),
        "processed_decision_points": processed,
        "skipped_long_trajectories": skipped_long_trajectories,
        "errors": len(errors),
        "model": args.model,
        "api_requests": boundary_client.request_count + state_client.request_count,
        "prompt_tokens": boundary_client.total_prompt_tokens + state_client.total_prompt_tokens,
        "completion_tokens": (
            boundary_client.total_completion_tokens + state_client.total_completion_tokens
        ),
        "total_tokens": boundary_client.total_tokens + state_client.total_tokens,
        "output": str(args.output),
        "preview": str(preview),
        "error_examples": errors[:20],
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if len(records) != total_target:
        raise SystemExit(f"generated {len(records)}/{total_target} requested samples")


if __name__ == "__main__":
    main()
