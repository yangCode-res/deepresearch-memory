#!/usr/bin/env python3
"""Build causal Working-State samples from retrospective trajectory segments.

The first teacher pass sees a complete, compact trajectory and places Loop
boundaries retrospectively.  A second teacher pass replays each decision point
causally: it sees only the state carried from earlier decisions and the newly
observed messages.  Future messages and the retrospective segmentation are
never placed in the model-training input.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import glob
import json
import os
from pathlib import Path
import re
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
    compact_message,
    decision_steps,
    initial_global_state,
    initial_working_state,
    iter_rows,
    validate_label,
    write_preview,
)
from build_working_state_pilot_v2 import (  # noqa: E402
    UPDATE_KEYS,
    build_target,
    compact_memories,
    compact_observed_messages,
    semantic_contract,
    semantic_similarity,
)


SEGMENTATION_SYSTEM_PROMPT = """You retrospectively segment one complete deep-research trajectory into Loops.
Return valid JSON only. Do not reveal chain-of-thought and do not reproduce the final answer.

You may read messages after a decision point only to determine the correct boundary label. A Loop is one
coherent, independently decidable information subgoal with one observable completion test. A different query,
tool, source, URL, failed search, or access strategy is not a new Loop. End a Loop only when its information
subgoal is resolved, refuted, saturated, superseded, or the following work pursues a genuinely different
dependency. Use READY_TO_ANSWER at the earliest tool-result decision point whose causal prefix already contains
stable, sufficient support for the exact answer. If the original agent searched after that point, leave those
trailing decisions outside the segmentation. If sufficient support is never reached, the last action is
CONTINUE_CURRENT_LOOP and all decisions remain covered.

Do not imitate the original agent's choice to keep searching. Audit every decision independently. For example,
if decision 1's tool result explicitly states the exact requested answer and decisions 2-4 only locate, repeat,
or reformat the same evidence, decision 1 MUST be READY_TO_ANSWER and decisions 2-4 MUST be omitted. For each
included decision, cite causal_evidence_ids no later than that decision's tool result. These coordinates prove
what was actually knowable at that point; later messages may never be cited for an earlier decision.

Decision indices refer only to tool-result messages marked with decision_index. Decision annotations must start
at 0 and be consecutive. loop_number starts at 1; it stays unchanged after CONTINUE and increases by exactly one
after SWITCH. Subgoals and completion tests must remain byte-for-byte identical inside a Loop and must describe
information outcomes; never name a website, URL, query, browser operation, or tool. Later messages help locate
boundaries but must not be used to claim that evidence existed earlier than it did."""


CAUSAL_STATE_SYSTEM_PROMPT = """You create a causal Working-State label at one fixed decision point.
Return valid JSON only. Do not reveal chain-of-thought and do not reconsider fixed_action.

Unlike the retrospective segmenter, you have no future trajectory. Use only working_state_before,
global_intent_state_before, available memories, and newly_observed_messages. Do not invent facts from a Loop
contract. Every key_evidence item must include at least one supplied msg_NNNN coordinate. Search snippets are
provisional. Separate hypotheses from confirmed evidence. Directional fields and the decision reason must not
prescribe or name a query, source, website, URL, tool, search/open/view action, or next browser operation.

For CONTINUE_CURRENT_LOOP, durable_update is empty and loop_memory is null. For SWITCH_LOOP or READY_TO_ANSWER,
write only durable findings supported by the causal prefix and a compact handoff memory. Select only allowed
memory IDs. Every confirmed fact, rejected finding, key evidence, and durable finding must include a supplied
msg_NNNN coordinate. A terminal loop_memory must contain at least one evidence_id. The next Loop contract is
deliberately hidden from you so it cannot leak future content into the current Loop's evidence or memory."""


SEGMENTATION_KEYS = {"trajectory_summary", "decisions"}
DECISION_ANNOTATION_KEYS = {
    "decision_index",
    "loop_number",
    "current_subgoal",
    "completion_test",
    "action",
    "outcome",
    "boundary_basis",
    "boundary_reason",
    "causal_evidence_ids",
}
CAUSAL_KEYS = {
    "decision_reason",
    "progress",
    "durable_update",
    "loop_memory",
    "retrieval",
}


def segmentation_contract(decision_count: int) -> dict[str, Any]:
    return {
        "trajectory_summary": "brief description of the research progression",
        "decisions": [
            {
                "decision_index": f"consecutive integer from 0 to at most {decision_count - 1}",
                "loop_number": 1,
                "current_subgoal": "one independently decidable information objective",
                "completion_test": "observable evidence condition",
                "action": "CONTINUE_CURRENT_LOOP|SWITCH_LOOP|READY_TO_ANSWER",
                "outcome": "IN_PROGRESS|RESOLVED|REFUTED|BLOCKED|SUPERSEDED",
                "boundary_basis": (
                    "NONE|SUBGOAL_COMPLETED|SUBGOAL_CHANGED|CANDIDATE_CHANGED|"
                    "BLOCKED_OR_SATURATED|PHASE_TRANSITION|TASK_COMPLETE"
                ),
                "boundary_reason": "short explanation based only on this decision's causal prefix",
                "causal_evidence_ids": ["msg_NNNN at or before this decision's tool result"],
            }
        ],
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _message_number(message_id: str) -> int | None:
    match = re.fullmatch(r"msg_(\d{4})", str(message_id))
    return int(match.group(1)) if match else None


def validate_segmentation(
    raw: dict[str, Any],
    *,
    decision_count: int,
    decision_message_limits: list[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != SEGMENTATION_KEYS:
        raise ValueError("segmentation response has missing or extra fields")
    summary = _normalize_text(raw["trajectory_summary"])
    decisions = raw.get("decisions")
    if not summary or not isinstance(decisions, list) or not decisions:
        raise ValueError("segmentation requires a summary and at least one decision annotation")
    if len(decisions) > decision_count:
        raise ValueError("segmentation contains more decisions than the trajectory")
    if decision_message_limits is not None and len(decision_message_limits) != decision_count:
        raise ValueError("decision_message_limits must cover the complete trajectory")

    normalized: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for expected_index, item in enumerate(decisions):
        if not isinstance(item, dict) or set(item) != DECISION_ANNOTATION_KEYS:
            raise ValueError("a decision annotation has missing or extra fields")
        try:
            decision_index = int(item["decision_index"])
            loop_number = int(item["loop_number"])
        except (TypeError, ValueError) as exc:
            raise ValueError("loop_number and decision_index must be integers") from exc
        if decision_index != expected_index:
            raise ValueError("decision annotations must be consecutive starting at decision 0")
        expected_loop = 1 if previous is None else (
            previous["loop_number"] + 1
            if previous["action"] == "SWITCH_LOOP"
            else previous["loop_number"]
        )
        if loop_number != expected_loop:
            raise ValueError("loop_number must increase exactly once after SWITCH and otherwise stay fixed")
        subgoal = _normalize_text(item["current_subgoal"])
        completion_test = _normalize_text(item["completion_test"])
        reason = _normalize_text(item["boundary_reason"])
        if not subgoal or not completion_test or not reason:
            raise ValueError("decision Loop contract and boundary reason cannot be empty")
        if CONCRETE_ACTION_PATTERN.search("\n".join((subgoal, completion_test))):
            raise ValueError("Loop contracts cannot contain concrete tools, queries, URLs, or named sites")
        if previous is not None and loop_number == previous["loop_number"] and (
            subgoal != previous["current_subgoal"]
            or completion_test != previous["completion_test"]
        ):
            raise ValueError("subgoal and completion_test must remain identical inside one Loop")
        if previous is not None and loop_number != previous["loop_number"]:
            if semantic_similarity(previous["current_subgoal"], subgoal) >= 0.72:
                raise ValueError("a SWITCH cannot be a source/query rephrasing of the same subgoal")
        action = str(item["action"] or "").upper()
        outcome = str(item["outcome"] or "").upper()
        basis = str(item["boundary_basis"] or "").upper()
        if action not in ACTIONS:
            raise ValueError("invalid decision action")
        # action is the retrospective judgment.  outcome and boundary_basis are
        # dependent enum fields, so normalize them deterministically rather
        # than consuming a repair request for an internally inconsistent tuple.
        if action == "CONTINUE_CURRENT_LOOP":
            outcome, basis = "IN_PROGRESS", "NONE"
        elif action == "READY_TO_ANSWER":
            outcome, basis = "RESOLVED", "TASK_COMPLETE"
        else:
            if outcome == "IN_PROGRESS" or outcome not in {
                "RESOLVED", "REFUTED", "BLOCKED", "SUPERSEDED"
            }:
                outcome = "RESOLVED"
            if basis in {"", "NONE", "TASK_COMPLETE"}:
                basis = "SUBGOAL_COMPLETED" if outcome == "RESOLVED" else "SUBGOAL_CHANGED"
        message_limit = (
            decision_message_limits[decision_index]
            if decision_message_limits is not None
            else None
        )
        evidence_ids = []
        for value in item.get("causal_evidence_ids") or []:
            evidence_id = str(value)
            number = _message_number(evidence_id)
            if number is not None and (message_limit is None or number <= message_limit):
                evidence_ids.append(evidence_id)
        # Coordinates are audit hints, not the label itself.  A retrospective
        # teacher may accidentally cite a later message while placing a valid
        # boundary.  Clamp the hints causally instead of spending repair calls;
        # the second-pass teacher must still produce actual prefix-supported
        # evidence before the sample is accepted.
        if not evidence_ids and message_limit is not None:
            evidence_ids = [f"msg_{message_limit:04d}"]
        if not evidence_ids:
            raise ValueError("every decision annotation requires causal_evidence_ids")
        normalized_item = {
            "decision_index": decision_index,
            "loop_number": loop_number,
            "current_subgoal": subgoal,
            "completion_test": completion_test,
            "action": action,
            "outcome": outcome,
            "boundary_basis": basis,
            "boundary_reason": reason,
            "causal_evidence_ids": evidence_ids,
        }
        normalized.append(normalized_item)
        previous = normalized_item

    final = normalized[-1]
    if final["action"] == "SWITCH_LOOP":
        raise ValueError("the final included decision cannot SWITCH without a following Loop decision")
    if final["action"] != "READY_TO_ANSWER" and len(normalized) != decision_count:
        raise ValueError("only an early READY decision may exclude trailing decisions")
    if any(item["action"] == "READY_TO_ANSWER" for item in normalized[:-1]):
        raise ValueError("READY must be the final included decision")

    loops: list[dict[str, Any]] = []
    for item in normalized:
        if not loops or item["loop_number"] != loops[-1]["loop_number"]:
            loops.append(
                {
                    "loop_number": item["loop_number"],
                    "start_decision_index": item["decision_index"],
                    "end_decision_index": item["decision_index"],
                    "subgoal": item["current_subgoal"],
                    "completion_test": item["completion_test"],
                    "end_action": item["action"],
                    "outcome": item["outcome"],
                    "boundary_basis": item["boundary_basis"],
                    "boundary_reason": item["boundary_reason"],
                }
            )
        else:
            loops[-1].update(
                {
                    "end_decision_index": item["decision_index"],
                    "end_action": item["action"],
                    "outcome": item["outcome"],
                    "boundary_basis": item["boundary_basis"],
                    "boundary_reason": item["boundary_reason"],
                }
            )
    if any(item["end_action"] != "SWITCH_LOOP" for item in loops[:-1]):
        raise ValueError("every non-final derived Loop must end with SWITCH_LOOP")

    raw["trajectory_summary"] = summary
    raw["decisions"] = normalized
    return {**raw, "loops": loops}


def trajectory_view(
    messages: list[dict[str, Any]],
    *,
    assistant_limit: int = 1800,
    tool_limit: int = 2800,
    final_limit: int = 3500,
) -> tuple[list[dict[str, Any]], int]:
    """Return all post-user messages with tool observations marked as decisions."""

    user_pos = next((i for i, message in enumerate(messages) if message.get("role") == "user"), -1)
    if user_pos < 0:
        return [], 0
    result: list[dict[str, Any]] = []
    decision_index = 0
    for pos in range(user_pos + 1, len(messages)):
        message = messages[pos]
        role = str(message.get("role") or "unknown")
        limit = tool_limit if role == "tool" else assistant_limit
        if role == "assistant" and message.get("channel") == "final":
            limit = final_limit
        item = compact_message(message, pos + 1, text_limit=limit)
        if role == "tool":
            item["decision_index"] = decision_index
            decision_index += 1
        else:
            item["decision_index"] = None
        result.append(item)
    return result, decision_index


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
        current = payload if attempt == 0 else {
            "validation_error": error,
            "invalid_response": raw,
            "instruction": "Repair the complete JSON object while preserving the evidence-based judgment.",
            "original_input": payload,
        }
        raw = client.complete_json(system_prompt, json.dumps(current, ensure_ascii=False))
        try:
            return validator(raw)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(f"invalid after repair attempts: {error}")


def segment_trajectory(
    client: OpenAIChatJSONClient,
    *,
    question: str,
    messages: list[dict[str, Any]],
    decision_count: int,
    decision_message_limits: list[int] | None = None,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "complete_trajectory": messages,
        "decision_count": decision_count,
        "required_output": segmentation_contract(decision_count),
    }
    return call_and_repair(
        client,
        system_prompt=SEGMENTATION_SYSTEM_PROMPT,
        payload=payload,
        validator=lambda raw: validate_segmentation(
            raw,
            decision_count=decision_count,
            decision_message_limits=decision_message_limits,
        ),
    )


def loop_for_decision(segmentation: dict[str, Any], decision_index: int) -> dict[str, Any]:
    for loop in segmentation["loops"]:
        if loop["start_decision_index"] <= decision_index <= loop["end_decision_index"]:
            return loop
    raise ValueError(f"decision {decision_index} is not covered by the retrospective segmentation")


def boundary_for_decision(segmentation: dict[str, Any], decision_index: int) -> dict[str, Any]:
    decisions = segmentation["decisions"]
    current = decisions[decision_index]
    action = current["action"]
    next_decision = decisions[decision_index + 1] if action == "SWITCH_LOOP" else None
    return {
        "action": action,
        "reason": "",
        "current_subgoal": current["current_subgoal"],
        "current_completion_test": current["completion_test"],
        "next_subgoal": next_decision["current_subgoal"] if next_decision else "",
        "next_completion_test": next_decision["completion_test"] if next_decision else "",
        "outcome": current["outcome"],
        "boundary_basis": current["boundary_basis"],
        "confidence": 1.0,
        "progress": {},
    }


def progress_contract() -> dict[str, Any]:
    return {
        "progress_summary": "at most two evidence-grounded sentences",
        "resolved_aspects": [],
        "open_aspects": [],
        "key_evidence": ["claim supported by msg_NNNN"],
        "candidate_answer": "",
        "active_hypotheses": [],
        "failed_strategies": [],
        "evidence_gaps": [],
        "answer_stable": False,
        "evidence_sufficient": False,
        "expected_information_gain": "HIGH|MEDIUM|LOW",
    }


def causal_contract(action: str, loop_number: int) -> dict[str, Any]:
    semantic = semantic_contract(action, loop_number)
    return {
        "decision_reason": "short causal-prefix reason for the fixed action",
        "progress": progress_contract(),
        "durable_update": semantic["durable_update"],
        "loop_memory": semantic["loop_memory"],
        "retrieval": semantic["retrieval"],
    }


def validate_causal_raw(
    raw: dict[str, Any],
    *,
    boundary: dict[str, Any],
    working_before: dict[str, Any],
    loop_number: int,
    seen_message_ids: set[str],
    allowed_memory_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != CAUSAL_KEYS:
        raise ValueError("causal-state response has missing or extra fields")
    reason = _normalize_text(raw["decision_reason"])
    if not reason:
        raise ValueError("decision_reason cannot be empty")
    if CONCRETE_ACTION_PATTERN.search(reason):
        raise ValueError("decision_reason cannot prescribe a query, source, URL, or tool action")
    progress = raw.get("progress")
    if not isinstance(progress, dict) or set(progress) != UPDATE_KEYS:
        raise ValueError("causal progress has missing or extra fields")
    for name in (
        "resolved_aspects",
        "open_aspects",
        "key_evidence",
        "active_hypotheses",
        "failed_strategies",
        "evidence_gaps",
    ):
        if not isinstance(progress[name], list):
            raise ValueError(f"progress.{name} must be a list")
        progress[name] = [_normalize_text(item)[:500] for item in progress[name] if _normalize_text(item)][:8]
    for item in progress["key_evidence"]:
        ids = set(re.findall(r"\bmsg_\d{4}\b", item))
        if not ids:
            raise ValueError("every key_evidence item must cite at least one msg_NNNN ID")
        if not ids <= seen_message_ids:
            raise ValueError("key_evidence cites a message outside the causal current-Loop prefix")
    for name in ("answer_stable", "evidence_sufficient"):
        if not isinstance(progress[name], bool):
            raise ValueError(f"progress.{name} must be boolean")
    gain = str(progress["expected_information_gain"] or "").upper()
    if gain not in GAIN_LEVELS:
        raise ValueError("invalid progress.expected_information_gain")
    progress["expected_information_gain"] = gain
    action = boundary["action"]
    if action == "READY_TO_ANSWER":
        progress["answer_stable"] = True
        progress["evidence_sufficient"] = True
        progress["expected_information_gain"] = "LOW"
    else:
        # These fields are dependent on the fixed retrospective action.  Do not
        # spend repair calls asking a second teacher to reproduce that decision.
        progress["answer_stable"] = False
        progress["evidence_sufficient"] = False
        if progress["expected_information_gain"] == "LOW":
            progress["expected_information_gain"] = "MEDIUM"

    operations = raw.get("durable_update")
    if not isinstance(operations, dict) or set(operations) != DELTA_FIELDS:
        raise ValueError("durable_update has missing or extra fields")
    if action == "CONTINUE_CURRENT_LOOP":
        if raw.get("loop_memory") is not None:
            raise ValueError("CONTINUE must not create loop_memory")
        if any(operations[name] for name in DELTA_FIELDS):
            raise ValueError("CONTINUE must have an empty durable_update")
    else:
        memory = raw.get("loop_memory")
        if not isinstance(memory, dict):
            raise ValueError("a terminal Loop requires loop_memory")
        evidence_ids = set(str(item) for item in (memory.get("evidence_ids") or []))
        if not evidence_ids or not evidence_ids <= seen_message_ids:
            raise ValueError("terminal loop_memory requires causal-prefix evidence_ids")
        for field in ("add_confirmed_facts", "add_rejected_hypotheses"):
            values = operations.get(field)
            if not isinstance(values, list):
                raise ValueError(f"durable_update.{field} must be a list")
            for item in values:
                ids = set(re.findall(r"\bmsg_\d{4}\b", str(item)))
                if not ids or not ids <= seen_message_ids:
                    raise ValueError(f"every durable_update.{field} item must cite a causal-prefix msg ID")
        findings = memory.get("durable_findings")
        if not isinstance(findings, list):
            raise ValueError("loop_memory.durable_findings must be a list")
        for item in findings:
            ids = set(re.findall(r"\bmsg_\d{4}\b", str(item)))
            if not ids or not ids <= seen_message_ids:
                raise ValueError("every loop_memory.durable_findings item must cite a causal-prefix msg ID")

    boundary = deepcopy(boundary)
    boundary["reason"] = reason
    boundary["progress"] = progress
    semantic_raw = {
        "durable_update": raw["durable_update"],
        "loop_memory": raw["loop_memory"],
        "retrieval": raw["retrieval"],
        "next_loop_setup": (
            {"evidence_gaps": [boundary["next_subgoal"]]}
            if action == "SWITCH_LOOP"
            else None
        ),
    }
    target = build_target(
        semantic_raw,
        boundary=boundary,
        working_before=working_before,
        loop_number=loop_number,
        seen_message_ids=seen_message_ids,
        allowed_memory_ids=allowed_memory_ids,
    )
    return target


def causal_teacher_payload(
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
    """Build the second-pass prompt without exposing future/next-Loop content."""

    allowed_memory_ids = {str(memory["memory_id"]) for memory in prior_memories}
    return {
        "question": question,
        "global_intent_state_before": global_state,
        "working_state_before": working_state,
        "newly_observed_messages": observed_messages,
        "current_loop_evidence_ids": sorted(seen_message_ids),
        "available_cross_loop_memories": compact_memories(prior_memories),
        "fixed_action": boundary["action"],
        "current_loop_contract": {
            "subgoal": boundary["current_subgoal"],
            "completion_test": boundary["current_completion_test"],
            "outcome": boundary["outcome"],
            "boundary_basis": boundary["boundary_basis"],
        },
        "allowed_memory_ids": sorted(allowed_memory_ids),
        "required_output": causal_contract(boundary["action"], loop_number),
    }


def write_causal_target(
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
    allowed_memory_ids = {str(memory["memory_id"]) for memory in prior_memories}
    if boundary["action"] != "CONTINUE_CURRENT_LOOP":
        allowed_memory_ids.add(f"memory_loop_{loop_number:03d}")
    payload = causal_teacher_payload(
        question=question,
        global_state=global_state,
        working_state=working_state,
        prior_memories=prior_memories,
        observed_messages=observed_messages,
        boundary=boundary,
        loop_number=loop_number,
        seen_message_ids=seen_message_ids,
    )
    return call_and_repair(
        client,
        system_prompt=CAUSAL_STATE_SYSTEM_PROMPT,
        payload=payload,
        validator=lambda raw: validate_causal_raw(
            raw,
            boundary=boundary,
            working_before=working_state,
            loop_number=loop_number,
            seen_message_ids=seen_message_ids,
            allowed_memory_ids=allowed_memory_ids,
        ),
    )


def audit_record_causality(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prefix_end = int(record["source"]["prefix_end_message_index"])
    input_data = record["input"]
    observed = input_data.get("observed_messages") or []
    if any(int(message.get("index") or 0) > prefix_end for message in observed):
        errors.append("training input contains a message after prefix_end_message_index")
    input_text = json.dumps(input_data, ensure_ascii=False)
    for message_id in re.findall(r"\bmsg_(\d{4})\b", input_text):
        if int(message_id) > prefix_end:
            errors.append("training input contains a future evidence ID")
            break
    target_text = json.dumps(record["target"], ensure_ascii=False)
    for message_id in re.findall(r"\bmsg_(\d{4})\b", target_text):
        if int(message_id) > prefix_end:
            errors.append("training target cites a future message ID")
            break
    forbidden_input_keys = {"complete_trajectory", "retrospective_segmentation", "next_loop_contract"}
    if forbidden_input_keys & set(input_data):
        errors.append("training input exposes retrospective or future-only fields")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-decision-points", type=int, default=3)
    parser.add_argument("--max-decision-points", type=int, default=10)
    parser.add_argument("--max-trajectory-chars", type=int, default=70000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model", default=os.getenv("CL_GISM_DATA_MODEL", "mimo-v2.5"))
    parser.add_argument("--base-url", default=os.getenv("CL_GISM_CONTROLLER_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("CL_GISM_CONTROLLER_API_KEY"))
    parser.add_argument("--preview-count", type=int, default=20)
    return parser.parse_args()


def _usage(*clients: OpenAIChatJSONClient) -> dict[str, int]:
    return {
        "api_requests": sum(client.request_count for client in clients),
        "prompt_tokens": sum(client.total_prompt_tokens for client in clients),
        "completion_tokens": sum(client.total_completion_tokens for client in clients),
        "total_tokens": sum(client.total_tokens for client in clients),
    }


def main() -> None:
    args = parse_args()
    if not args.api_key or not args.base_url:
        raise SystemExit("controller API configuration is required")
    paths = glob.glob(args.input_glob)
    if not paths:
        raise SystemExit("no input parquet files matched")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    segment_path = args.output.with_suffix(".segments.jsonl")
    partial_usage_path = args.output.with_suffix(".usage.partial.json")
    segment_client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=300,
        max_tokens=3072,
    )
    state_client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=300,
        max_tokens=3072,
    )
    records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    action_counts = {action: 0 for action in sorted(ACTIONS)}
    skipped_length = 0
    skipped_chars = 0
    excluded_trailing = 0
    causal_violations: list[dict[str, Any]] = []

    def checkpoint(extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "generated_samples": len(records),
            "segmented_trajectories": len(segment_records),
            **_usage(segment_client, state_client),
        }
        if extra:
            payload.update(extra)
        partial_usage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[retrospective] model={args.model} target={args.num_samples} "
        f"trajectory_decisions={args.min_decision_points}..{args.max_decision_points}",
        flush=True,
    )
    with args.output.open("w", encoding="utf-8") as output, segment_path.open("w", encoding="utf-8") as segments_out:
        for row in iter_rows(paths, seed=args.seed):
            if len(records) >= args.num_samples:
                break
            if str(row.get("status") or "").lower() not in {"success", "completed", "ok"}:
                continue
            messages = row.get("messages") or []
            steps = decision_steps(messages)
            if not (args.min_decision_points <= len(steps) <= args.max_decision_points):
                skipped_length += 1
                continue
            full_view, decision_count = trajectory_view(messages)
            trajectory_chars = len(json.dumps(full_view, ensure_ascii=False))
            if trajectory_chars > args.max_trajectory_chars:
                skipped_chars += 1
                continue
            question = str(row.get("question") or "").strip()
            try:
                segmentation = segment_trajectory(
                    segment_client,
                    question=question,
                    messages=full_view,
                    decision_count=decision_count,
                    decision_message_limits=[
                        int(step["prefix_end_message_index"])
                        for step in steps
                    ],
                )
            except Exception as exc:
                errors.append({"stage": "segmentation", "qid": row.get("qid"), "error": str(exc)})
                checkpoint()
                print(f"[retrospective] skip segmentation qid={row.get('qid')}: {exc}", flush=True)
                continue

            usable_end = int(segmentation["loops"][-1]["end_decision_index"])
            trailing = decision_count - usable_end - 1
            excluded_trailing += trailing
            segment_record = {
                "source": {
                    "dataset": "OpenResearcher/OpenResearcher-Dataset",
                    "qid": row.get("qid"),
                },
                "question": question,
                "message_count": len(messages),
                "decision_count": decision_count,
                "trajectory_chars": trajectory_chars,
                "usable_end_decision_index": usable_end,
                "excluded_trailing_decisions": trailing,
                "segmentation": segmentation,
            }
            segment_records.append(segment_record)
            segments_out.write(json.dumps(segment_record, ensure_ascii=False) + "\n")
            segments_out.flush()
            checkpoint()
            print(
                f"[retrospective] segmented qid={row.get('qid')} decisions={decision_count} "
                f"loops={len(segmentation['loops'])} trailing_excluded={trailing}",
                flush=True,
            )

            global_state = initial_global_state(question)
            working_state = initial_working_state()
            memories: list[dict[str, Any]] = []
            loop_number = 1
            current_loop_evidence_ids: set[str] = set()
            for decision_index, step in enumerate(steps[: usable_end + 1]):
                if len(records) >= args.num_samples:
                    break
                observed_messages = compact_observed_messages(
                    step["observed_messages"], tool_limit=4000, assistant_limit=1200
                )
                current_loop_evidence_ids.update(message["message_id"] for message in observed_messages)
                boundary = boundary_for_decision(segmentation, decision_index)
                if int(loop_number) != int(loop_for_decision(segmentation, decision_index)["loop_number"]):
                    raise RuntimeError("controller replay drifted from retrospective Loop numbering")
                try:
                    target = write_causal_target(
                        state_client,
                        question=question,
                        global_state=global_state,
                        working_state=working_state,
                        prior_memories=memories,
                        observed_messages=observed_messages,
                        boundary=boundary,
                        loop_number=loop_number,
                        seen_message_ids=current_loop_evidence_ids,
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "causal_state",
                            "qid": row.get("qid"),
                            "step_index": decision_index,
                            "error": str(exc),
                        }
                    )
                    checkpoint()
                    print(
                        f"[retrospective] abandon qid={row.get('qid')} step={decision_index}: {exc}",
                        flush=True,
                    )
                    break
                record = {
                    "sample_id": f"ws_retro_{len(records) + 1:04d}",
                    "source": {
                        "dataset": "OpenResearcher/OpenResearcher-Dataset",
                        "qid": row.get("qid"),
                        "step_index": decision_index,
                        "prefix_end_message_index": step["prefix_end_message_index"],
                        "retrospective_loop_number": loop_number,
                    },
                    "teacher": {
                        "model": args.model,
                        "label_type": "retrospective_boundary_causal_state_v1",
                        "future_visible_only_to_boundary_segmenter": True,
                    },
                    "input": {
                        "question": question,
                        "global_intent_state_before": deepcopy(global_state),
                        "working_state_before": deepcopy(working_state),
                        "observed_messages": observed_messages,
                        "current_loop_evidence_ids": sorted(current_loop_evidence_ids),
                        "available_cross_loop_memories": deepcopy(memories[-8:]),
                    },
                    "target": target,
                }
                violations = audit_record_causality(record)
                if violations:
                    causal_violations.append({"sample_id": record["sample_id"], "errors": violations})
                    errors.append(
                        {
                            "stage": "causality_audit",
                            "qid": row.get("qid"),
                            "step_index": decision_index,
                            "error": "; ".join(violations),
                        }
                    )
                    print(f"[retrospective] reject causal leakage {record['sample_id']}: {violations}", flush=True)
                    break

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                action = target["loop_decision"]["action"]
                action_counts[action] += 1
                global_state = apply_delta(global_state, target["state_delta"]["operations"])
                memory = target["cross_loop_memory"]
                if isinstance(memory, dict):
                    memories.append(memory)
                working_state = target["working_state_after"]
                if action == "SWITCH_LOOP":
                    loop_number += 1
                    current_loop_evidence_ids = set()
                checkpoint()
                print(
                    f"[retrospective] saved={len(records)}/{args.num_samples} qid={row.get('qid')} "
                    f"step={decision_index} loop={loop_number} action={action}",
                    flush=True,
                )
                if action == "READY_TO_ANSWER":
                    break

    preview = args.output.with_suffix(".preview.md")
    write_preview(preview, records, count=args.preview_count)
    loop_counts = [len(item["segmentation"]["loops"]) for item in segment_records]
    report = {
        "requested_samples": args.num_samples,
        "valid_samples": len(records),
        "action_counts": action_counts,
        "unique_questions": len({str(record["source"]["qid"]) for record in records}),
        "segmented_trajectories": len(segment_records),
        "loops_per_trajectory": loop_counts,
        "average_loops_per_trajectory": (
            round(sum(loop_counts) / len(loop_counts), 3) if loop_counts else 0
        ),
        "excluded_trailing_decisions": excluded_trailing,
        "causality_violations": causal_violations,
        "errors": len(errors),
        "error_examples": errors[:20],
        "skipped_by_decision_length": skipped_length,
        "skipped_by_character_budget": skipped_chars,
        "model": args.model,
        **_usage(segment_client, state_client),
        "output": str(args.output),
        "segments": str(segment_path),
        "preview": str(preview),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint({"complete": len(records) == args.num_samples})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if len(records) != args.num_samples or causal_violations:
        raise SystemExit(f"generated {len(records)}/{args.num_samples} valid causal samples")


if __name__ == "__main__":
    main()
