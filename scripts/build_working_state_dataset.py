#!/usr/bin/env python3
"""Build supervised Working-State decisions from OpenResearcher trajectories.

One output record represents the control decision made after a browser/tool
observation and before the research model's next call.  Labels are produced by
an OpenAI-compatible teacher, validated, and checkpointed as JSONL.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import glob
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_gism.llm_planner import OpenAIChatJSONClient  # noqa: E402


ACTIONS = {"CONTINUE_CURRENT_LOOP", "SWITCH_LOOP", "READY_TO_ANSWER"}
GAIN_LEVELS = {"HIGH", "MEDIUM", "LOW"}
WORKING_STATE_FIELDS = {
    "loop_id",
    "status",
    "current_subgoal",
    "completion_test",
    "progress_summary",
    "resolved_aspects",
    "open_aspects",
    "key_evidence",
    "candidate_answer",
    "active_hypotheses",
    "failed_strategies",
    "next_direction",
    "evidence_gaps",
    "answer_stable",
    "evidence_sufficient",
    "expected_information_gain",
}
DELTA_FIELDS = {
    "add_confirmed_facts",
    "add_working_hypotheses",
    "add_rejected_hypotheses",
    "add_open_questions",
    "resolve_open_questions",
    "completed_subgoal",
}
CONCRETE_ACTION_PATTERN = re.compile(
    r"(^|[.!?;]\s*)(search|query|open|view|visit|browse|google|click|read|inspect|look\s+up|"
    r"use\s+(?:the\s+)?browser|搜索|查询|查找|打开|查看|访问|点击)\b|https?://|"
    r"\b(browser\.search|browser\.open|wikipedia|google|bing)\b",
    re.IGNORECASE,
)


def message_text(message: dict[str, Any]) -> str:
    """Flatten the dataset's nested message content without losing tool calls."""

    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
    if message.get("reasoning_content"):
        parts.append(str(message["reasoning_content"]))
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(parts).strip()


def compact_message(message: dict[str, Any], index: int, *, text_limit: int = 8000) -> dict[str, Any]:
    text = message_text(message)
    return {
        "message_id": f"msg_{index:04d}",
        "index": index,
        "role": str(message.get("role") or "unknown"),
        "name": message.get("name"),
        "recipient": message.get("recipient"),
        "channel": message.get("channel"),
        "text": text[:text_limit],
        "truncated": len(text) > text_limit,
    }


def decision_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deployment-aligned snapshots ending at each tool observation."""

    user_pos = next((i for i, message in enumerate(messages) if message.get("role") == "user"), -1)
    if user_pos < 0:
        return []
    steps: list[dict[str, Any]] = []
    start = user_pos + 1
    for pos in range(start, len(messages)):
        if messages[pos].get("role") != "tool":
            continue
        observed = [compact_message(messages[i], i + 1) for i in range(start, pos + 1)]
        if observed:
            steps.append(
                {
                    "step_index": len(steps),
                    "prefix_end_message_index": pos + 1,
                    "observed_messages": observed,
                }
            )
        start = pos + 1
    return steps


def initial_global_state(question: str) -> dict[str, Any]:
    return {
        "goal": question,
        "success_criteria": ["Answer the question with evidence sufficient to support the exact answer."],
        "immutable_constraints": [
            "Do not invent facts or citations.",
            "Keep hypotheses distinct from confirmed evidence.",
        ],
        "confirmed_facts": [],
        "working_hypotheses": [],
        "rejected_hypotheses": [],
        "open_questions": [question],
        "completed_subgoals": [],
    }


def initial_working_state() -> dict[str, Any]:
    return {
        "loop_id": "loop_001",
        "status": "IN_PROGRESS",
        "current_subgoal": "Identify a promising evidence path toward the requested answer",
        "completion_test": "A concrete candidate or dependency is supported by tool evidence",
        "progress_summary": "No tool evidence has been observed yet.",
        "resolved_aspects": [],
        "open_aspects": ["Find the first evidence-backed candidate or dependency."],
        "key_evidence": [],
        "candidate_answer": "",
        "active_hypotheses": [],
        "failed_strategies": [],
        "next_direction": "Seek evidence that can establish a concrete candidate or dependency.",
        "evidence_gaps": ["No external evidence has been collected."],
        "answer_stable": False,
        "evidence_sufficient": False,
        "expected_information_gain": "HIGH",
    }


def apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state)
    mappings = {
        "add_confirmed_facts": "confirmed_facts",
        "add_working_hypotheses": "working_hypotheses",
        "add_rejected_hypotheses": "rejected_hypotheses",
        "add_open_questions": "open_questions",
    }
    for source, target in mappings.items():
        for item in delta.get(source) or []:
            if item not in updated[target]:
                updated[target].append(item)
    resolved = set(delta.get("resolve_open_questions") or [])
    updated["open_questions"] = [item for item in updated["open_questions"] if item not in resolved]
    completed = str(delta.get("completed_subgoal") or "").strip()
    if completed and completed not in updated["completed_subgoals"]:
        updated["completed_subgoals"].append(completed)
    return updated


def _string_list(value: Any, name: str, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if len(result) > limit:
        raise ValueError(f"{name} contains more than {limit} items")
    return result


def validate_label(
    raw: dict[str, Any], *, seen_message_ids: set[str], allowed_memory_ids: set[str], loop_number: int
) -> dict[str, Any]:
    required = {
        "working_state_after",
        "loop_decision",
        "state_delta",
        "cross_loop_memory",
        "retrieval",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"response must contain exactly {sorted(required)}")

    working = raw["working_state_after"]
    if not isinstance(working, dict) or set(working) != WORKING_STATE_FIELDS:
        raise ValueError("working_state_after has missing or extra fields")
    for name in (
        "resolved_aspects",
        "open_aspects",
        "key_evidence",
        "active_hypotheses",
        "failed_strategies",
        "evidence_gaps",
    ):
        working[name] = _string_list(working[name], f"working_state_after.{name}")
    for name in ("answer_stable", "evidence_sufficient"):
        if not isinstance(working[name], bool):
            raise ValueError(f"working_state_after.{name} must be boolean")
    gain = str(working["expected_information_gain"]).upper()
    if gain not in GAIN_LEVELS:
        raise ValueError("invalid expected_information_gain")
    working["expected_information_gain"] = gain
    if not str(working["current_subgoal"]).strip() or not str(working["completion_test"]).strip():
        raise ValueError("working state requires a subgoal and completion test")
    direction = str(working["next_direction"] or "").strip()
    if not direction:
        raise ValueError("working_state_after.next_direction cannot be empty")
    policy_text = "\n".join(
        [direction, str(working["completion_test"]), *working["open_aspects"], *working["evidence_gaps"]]
    )
    if CONCRETE_ACTION_PATTERN.search(policy_text):
        raise ValueError(
            "directional Working State fields must state information objectives and stop conditions, "
            "not exact search/open/view actions, named websites, or tools"
        )
    cited_ids = set(
        re.findall(
            r"\bmsg_\d{4}\b",
            "\n".join(
                [working["progress_summary"], *working["key_evidence"], *working["resolved_aspects"]]
            ),
        )
    )
    if not cited_ids <= seen_message_ids:
        raise ValueError("working state cites unseen message IDs")
    if not working["evidence_sufficient"] and not (
        working["evidence_gaps"] or working["open_aspects"]
    ):
        raise ValueError("insufficient evidence requires an explicit evidence gap or open aspect")

    decision = raw["loop_decision"]
    if not isinstance(decision, dict):
        raise ValueError("loop_decision must be an object")
    action = str(decision.get("action") or "").upper()
    if action not in ACTIONS:
        raise ValueError("invalid loop_decision.action")
    decision["action"] = action
    if not str(decision.get("reason") or "").strip():
        raise ValueError("loop_decision.reason cannot be empty")
    outcome = str(decision.get("outcome") or "").upper()
    boundary_basis = str(decision.get("boundary_basis") or "").upper()
    next_subgoal = str(decision.get("next_subgoal") or "").strip()
    current_subgoal = str(decision.get("current_subgoal") or "").strip()
    if not current_subgoal:
        raise ValueError("loop_decision.current_subgoal cannot be empty")
    if action == "CONTINUE_CURRENT_LOOP" and (
        outcome != "IN_PROGRESS" or boundary_basis != "NONE" or next_subgoal
    ):
        raise ValueError(
            "CONTINUE_CURRENT_LOOP requires IN_PROGRESS, boundary_basis=NONE, and empty next_subgoal"
        )
    if action == "SWITCH_LOOP" and (
        outcome == "IN_PROGRESS"
        or boundary_basis == "NONE"
        or not next_subgoal
        or " ".join(current_subgoal.casefold().split())
        == " ".join(next_subgoal.casefold().split())
    ):
        raise ValueError("SWITCH_LOOP requires a terminal outcome, real boundary, and distinct next subgoal")
    if action == "READY_TO_ANSWER" and (
        outcome != "RESOLVED" or boundary_basis != "TASK_COMPLETE" or next_subgoal
    ):
        raise ValueError(
            "READY_TO_ANSWER requires RESOLVED, TASK_COMPLETE, and empty next_subgoal"
        )
    expected_loop_id = (
        f"loop_{loop_number + 1:03d}" if action == "SWITCH_LOOP" else f"loop_{loop_number:03d}"
    )
    if str(working["loop_id"]) != expected_loop_id:
        raise ValueError(f"working_state_after.loop_id must be {expected_loop_id} for {action}")
    if action == "READY_TO_ANSWER" and str(working["status"]).upper() != "COMPLETED":
        raise ValueError("READY_TO_ANSWER requires working_state_after.status=COMPLETED")

    delta = raw["state_delta"]
    if not isinstance(delta, dict) or set(delta) != {"mode", "summary", "operations"}:
        raise ValueError("state_delta must contain mode, summary, and operations")
    mode = str(delta["mode"]).upper()
    if mode not in {"NOOP", "APPLY"}:
        raise ValueError("invalid state_delta.mode")
    operations = delta["operations"]
    if not isinstance(operations, dict) or set(operations) != DELTA_FIELDS:
        raise ValueError("state_delta.operations has missing or extra fields")
    for name in DELTA_FIELDS - {"completed_subgoal"}:
        operations[name] = _string_list(operations[name], f"state_delta.operations.{name}", limit=6)
    operations["completed_subgoal"] = str(operations["completed_subgoal"] or "").strip()
    boundary = action != "CONTINUE_CURRENT_LOOP"
    if boundary != (mode == "APPLY"):
        raise ValueError("StateDelta must APPLY exactly when the current loop ends")
    if mode == "NOOP" and (any(operations[name] for name in DELTA_FIELDS)):
        raise ValueError("NOOP StateDelta must have empty operations")
    delta["mode"] = mode

    memory = raw["cross_loop_memory"]
    if boundary:
        if not isinstance(memory, dict):
            raise ValueError("a loop boundary requires cross_loop_memory")
        memory_id = str(memory.get("memory_id") or "")
        expected_memory_id = f"memory_loop_{loop_number:03d}"
        if memory_id != expected_memory_id:
            raise ValueError(f"cross_loop_memory.memory_id must be {expected_memory_id}")
        evidence_ids = set(_string_list(memory.get("evidence_ids"), "cross_loop_memory.evidence_ids"))
        if not evidence_ids <= seen_message_ids:
            raise ValueError("cross-loop memory cites unseen message IDs")
        for name in ("durable_findings", "rejected_leads", "unresolved_questions"):
            memory[name] = _string_list(memory.get(name), f"cross_loop_memory.{name}", limit=6)
        if not str(memory.get("summary") or "").strip():
            raise ValueError("cross_loop_memory.summary cannot be empty")
    elif memory is not None:
        raise ValueError("continuing a loop must not emit cross_loop_memory")

    retrieval = raw["retrieval"]
    if not isinstance(retrieval, dict):
        raise ValueError("retrieval must be an object")
    selected = _string_list(retrieval.get("relevant_memory_ids"), "retrieval.relevant_memory_ids", limit=4)
    if not set(selected) <= allowed_memory_ids:
        raise ValueError("retrieval selected an unknown memory ID")
    retrieval["relevant_memory_ids"] = selected
    if action == "READY_TO_ANSWER" and not (
        working["answer_stable"] and working["evidence_sufficient"] and gain == "LOW"
    ):
        raise ValueError("READY_TO_ANSWER requires stable, sufficient evidence and LOW information gain")
    return raw


SYSTEM_PROMPT = """You label deployment-time control decisions for a deep-research memory system.
Return one JSON object only. Do not reveal chain-of-thought. Use only evidence visible in observed_messages
or supplied prior memory. Never use the reference answer (it is deliberately absent).

A Loop is one coherent, locally decidable subgoal with an observable completion test. Query wording,
tool choice, source changes, and ordinary failed searches do not create a boundary. Continue while the
same subgoal and completion test remain. Switch only after the subgoal is resolved, refuted, saturated,
superseded, or replaced by a genuinely different dependency. READY_TO_ANSWER is also a terminal boundary.

Working State is loop-local and directional. Preserve evidence-bearing progress; separate confirmed facts
from hypotheses; record failed strategy families; describe an objective and stop condition, but never prescribe
an exact query, URL, browser tool, or mandatory action. A StateDelta contains only durable cross-loop changes
and is APPLY exactly at SWITCH_LOOP or READY_TO_ANSWER. Cross-loop memory must cite only supplied msg_NNNN IDs.
Select only supplied prior memory IDs. Search-result snippets are provisional evidence, not automatically
authoritative or primary sources; Wikipedia is never a primary source. Do not name a website in next_direction,
completion_test, open_aspects, or evidence_gaps. CONTINUE always uses outcome=IN_PROGRESS and
boundary_basis=NONE. Prefer not to end a loop over inventing unsupported progress."""


def output_contract(loop_number: int) -> dict[str, Any]:
    empty_ops = {name: ([] if name != "completed_subgoal" else "") for name in DELTA_FIELDS}
    return {
        "working_state_after": {
            "loop_id": (
                f"loop_{loop_number:03d} when continuing/answering; "
                f"loop_{loop_number + 1:03d} when switching"
            ),
            "status": "IN_PROGRESS|COMPLETED",
            "current_subgoal": "one primary subgoal",
            "completion_test": "observable condition",
            "progress_summary": "at most two sentences",
            "resolved_aspects": [],
            "open_aspects": [],
            "key_evidence": ["claim with msg_NNNN source coordinate"],
            "candidate_answer": "",
            "active_hypotheses": [],
            "failed_strategies": [],
            "next_direction": "directional objective and stop condition, no exact query/tool/URL",
            "evidence_gaps": [],
            "answer_stable": False,
            "evidence_sufficient": False,
            "expected_information_gain": "HIGH|MEDIUM|LOW",
        },
        "loop_decision": {
            "action": "CONTINUE_CURRENT_LOOP|SWITCH_LOOP|READY_TO_ANSWER",
            "reason": "short evidence-grounded reason",
            "current_subgoal": "the ending/current subgoal",
            "next_subgoal": "empty unless SWITCH_LOOP",
            "outcome": "IN_PROGRESS|RESOLVED|REFUTED|BLOCKED|SUPERSEDED",
            "boundary_basis": "NONE|SUBGOAL_COMPLETED|SUBGOAL_CHANGED|CANDIDATE_CHANGED|BLOCKED_OR_SATURATED|PHASE_TRANSITION|TASK_COMPLETE",
        },
        "state_delta": {"mode": "NOOP|APPLY", "summary": "", "operations": empty_ops},
        "cross_loop_memory": None,
        "retrieval": {
            "query": "semantic query over prior memories",
            "relevant_memory_ids": [],
            "reason": "why these prior memories matter",
        },
    }


def label_step(
    client: OpenAIChatJSONClient,
    *,
    question: str,
    global_state: dict[str, Any],
    working_state: dict[str, Any],
    prior_memories: list[dict[str, Any]],
    step: dict[str, Any],
    loop_number: int,
    current_loop_evidence_ids: set[str],
) -> dict[str, Any]:
    seen_ids = set(current_loop_evidence_ids)
    allowed_memory_ids = {str(memory["memory_id"]) for memory in prior_memories}
    payload = {
        "question": question,
        "global_intent_state_before": global_state,
        "working_state_before": working_state,
        "observed_messages": step["observed_messages"],
        "current_loop_evidence_ids": sorted(seen_ids),
        "available_cross_loop_memories": prior_memories[-8:],
        "required_output": output_contract(loop_number),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    raw: dict[str, Any] = {}
    error = ""
    for attempt in range(3):
        current = prompt if not attempt else json.dumps(
            {
                "validation_error": error,
                "invalid_response": raw,
                "instruction": "Repair the response. Return the complete JSON object only.",
                "original_input": payload,
            },
            ensure_ascii=False,
        )
        raw = client.complete_json(SYSTEM_PROMPT, current)
        try:
            return validate_label(
                raw,
                seen_message_ids=seen_ids,
                allowed_memory_ids=allowed_memory_ids,
                loop_number=loop_number,
            )
        except ValueError as exc:
            error = str(exc)
    raise ValueError(f"label invalid after repair attempts: {error}")


def iter_rows(paths: list[str], *, seed: int) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on server environment
        raise RuntimeError("pyarrow is required to read the OpenResearcher parquet files") from exc
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    for path in paths:
        parquet = pq.ParquetFile(path)
        groups = list(range(parquet.num_row_groups))
        rng.shuffle(groups)
        for group in groups:
            rows = parquet.read_row_group(group).to_pylist()
            rng.shuffle(rows)
            yield from rows


def write_preview(path: Path, records: list[dict[str, Any]], *, count: int) -> None:
    lines = ["# Working State dataset preview", "", f"Samples generated: {len(records)}", ""]
    for index, record in enumerate(records[:count], start=1):
        lines.extend(
            [
                f"## Sample {index}: qid={record['source']['qid']} step={record['source']['step_index']}",
                "",
                f"**Question:** {record['input']['question']}",
                "",
                "### Input: Working State before",
                "",
                "```json",
                json.dumps(record["input"]["working_state_before"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### New observed messages",
                "",
                "```json",
                json.dumps(record["input"]["observed_messages"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Target",
                "",
                "```json",
                json.dumps(record["target"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model", default=os.getenv("CL_GISM_DATA_MODEL", "mimo-v2.5"))
    parser.add_argument("--base-url", default=os.getenv("CL_GISM_CONTROLLER_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("CL_GISM_CONTROLLER_API_KEY"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key or not args.base_url:
        raise SystemExit("CL_GISM_CONTROLLER_API_KEY and CL_GISM_CONTROLLER_BASE_URL are required")
    paths = glob.glob(args.input_glob)
    if not paths:
        raise SystemExit(f"No parquet files matched: {args.input_glob}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
    )
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    print(f"[builder] model={args.model} files={len(paths)} target={args.num_samples}", flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in iter_rows(paths, seed=args.seed):
            if len(records) >= args.num_samples:
                break
            if str(row.get("status") or "").lower() not in {"success", "completed", "ok"}:
                continue
            messages = row.get("messages") or []
            steps = decision_steps(messages)[: args.max_steps_per_trajectory]
            if not steps:
                continue
            question = str(row.get("question") or "").strip()
            global_state = initial_global_state(question)
            working_state = initial_working_state()
            memories: list[dict[str, Any]] = []
            loop_number = 1
            current_loop_evidence_ids: set[str] = set()
            for step in steps:
                if len(records) >= args.num_samples:
                    break
                current_loop_evidence_ids.update(
                    message["message_id"] for message in step["observed_messages"]
                )
                try:
                    target = label_step(
                        client,
                        question=question,
                        global_state=global_state,
                        working_state=working_state,
                        prior_memories=memories,
                        step=step,
                        loop_number=loop_number,
                        current_loop_evidence_ids=current_loop_evidence_ids,
                    )
                except Exception as exc:  # keep harvesting other trajectories
                    errors.append(
                        {"qid": row.get("qid"), "step_index": step["step_index"], "error": str(exc)}
                    )
                    print(f"[builder] skip qid={row.get('qid')} step={step['step_index']}: {exc}", flush=True)
                    break
                record = {
                    "sample_id": f"ws_{len(records) + 1:06d}",
                    "source": {
                        "dataset": "OpenResearcher/OpenResearcher-Dataset",
                        "qid": row.get("qid"),
                        "step_index": step["step_index"],
                        "prefix_end_message_index": step["prefix_end_message_index"],
                    },
                    "teacher": {"model": args.model, "label_type": "offline_teacher"},
                    "input": {
                        "question": question,
                        "global_intent_state_before": deepcopy(global_state),
                        "working_state_before": deepcopy(working_state),
                        "observed_messages": step["observed_messages"],
                        "current_loop_evidence_ids": sorted(current_loop_evidence_ids),
                        "available_cross_loop_memories": deepcopy(memories[-8:]),
                    },
                    "target": target,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                global_state = apply_delta(global_state, target["state_delta"]["operations"])
                memory = target.get("cross_loop_memory")
                if isinstance(memory, dict):
                    memories.append(memory)
                working_state = target["working_state_after"]
                action = target["loop_decision"]["action"]
                if action == "SWITCH_LOOP":
                    loop_number += 1
                    current_loop_evidence_ids = set()
                if action == "READY_TO_ANSWER":
                    break
                print(
                    f"[builder] {len(records)}/{args.num_samples} qid={row.get('qid')} "
                    f"step={step['step_index']} action={action}",
                    flush=True,
                )
    preview = args.output.with_suffix(".preview.md")
    write_preview(preview, records, count=args.preview_count)
    report = {
        "requested_samples": args.num_samples,
        "valid_samples": len(records),
        "errors": len(errors),
        "model": args.model,
        "seed": args.seed,
        "output": str(args.output),
        "preview": str(preview),
        "error_examples": errors[:20],
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if len(records) != args.num_samples:
        raise SystemExit(f"Generated {len(records)} valid samples, expected {args.num_samples}")


if __name__ == "__main__":
    main()
