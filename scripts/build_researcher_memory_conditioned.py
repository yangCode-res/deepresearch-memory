#!/usr/bin/env python3
"""Build causally aligned, Memory-conditioned Researcher SFT samples.

Controller labels are replayed at tool-result boundaries.  For every replay
row, the assistant messages immediately preceding that row's tool observation
become a Researcher target.  The Researcher input uses the Controller state
that existed *before* those assistant messages, plus only earlier messages from
the current Loop and a causal pre-action Memory snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import glob
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(ROOT / "src"), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_working_state_dataset import apply_delta, compact_message  # noqa: E402
from curate_retrospective_training20 import (  # noqa: E402
    apply_contract_overrides,
    qid_of,
    read_jsonl,
    validate_pool,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "with", "requested", "evidence",
    "establish", "determine", "identify", "extract", "find", "verify",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="append", required=True, type=Path)
    parser.add_argument("--segments", action="append", required=True, type=Path)
    parser.add_argument("--controller-training", required=True, type=Path)
    parser.add_argument("--contract-overrides", type=Path)
    parser.add_argument("--raw-input-glob")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-history-messages", type=int, default=24)
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_PATTERN.findall(str(value or ""))
        if token.casefold() not in STOPWORDS and len(token) > 1
    }


def memory_text(memory: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(memory.get("summary") or ""),
            *[str(item) for item in memory.get("durable_findings") or []],
            *[str(item) for item in memory.get("unresolved_questions") or []],
        ]
    )


def retrieval_text(working_state: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(working_state.get("current_subgoal") or ""),
            str(working_state.get("completion_test") or ""),
            *[str(item) for item in working_state.get("open_aspects") or []],
            *[str(item) for item in working_state.get("evidence_gaps") or []],
        ]
    )


def select_causal_memories(
    available: list[dict[str, Any]], working_state: dict[str, Any], *, limit: int = 2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank only Memories available before the Researcher action."""

    query_tokens = tokens(retrieval_text(working_state))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, memory in enumerate(available):
        mem_tokens = tokens(memory_text(memory))
        overlap = len(query_tokens & mem_tokens)
        score = overlap / max(1, len(query_tokens))
        unresolved = bool(memory.get("unresolved_questions"))
        if unresolved:
            score += 0.15
        # Recent Loop memories are preferred for handoff continuity, but an
        # unrelated distractor is not selected solely because it is recent.
        score += index * 1e-6
        scored.append((score, overlap, memory))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for score, overlap, memory in scored:
        audit.append(
            {
                "memory_id": str(memory["memory_id"]),
                "score": round(score, 6),
                "token_overlap": overlap,
            }
        )
        if len(selected) >= limit:
            continue
        if overlap > 0 or memory.get("unresolved_questions"):
            selected.append(deepcopy(memory))
    return selected, audit


def assistant_targets(observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(message) for message in observed if message.get("role") == "assistant"]


def tool_observation(observed: list[dict[str, Any]]) -> dict[str, Any] | None:
    tools = [message for message in observed if message.get("role") == "tool"]
    return deepcopy(tools[-1]) if tools else None


def parse_tool_call(messages: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    calls = [
        message
        for message in messages
        if str(message.get("recipient") or "").startswith("browser.")
    ]
    if not calls:
        return None, "assistant target has no native browser recipient"
    message = calls[-1]
    try:
        arguments = json.loads(str(message.get("text") or ""))
    except json.JSONDecodeError:
        return None, "browser arguments are not valid JSON"
    if not isinstance(arguments, dict):
        return None, "browser arguments must be a JSON object"
    return {
        "message_id": message["message_id"],
        "name": str(message["recipient"]),
        "arguments": arguments,
    }, None


def tool_fingerprint(tool_call: dict[str, Any]) -> str:
    return f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True, ensure_ascii=False)}"


def history_tool_fingerprints(history: Iterable[dict[str, Any]]) -> set[str]:
    fingerprints: set[str] = set()
    for message in history:
        call, error = parse_tool_call([message])
        if call is not None and error is None:
            fingerprints.add(tool_fingerprint(call))
    return fingerprints


def repeats_memory_known_fact(
    tool_call: dict[str, Any] | None,
    selected_memories: list[dict[str, Any]],
    working_state: dict[str, Any],
) -> bool:
    if not tool_call or tool_call["name"] != "browser.search" or not selected_memories:
        return False
    query = str(tool_call["arguments"].get("query") or "")
    query_tokens = tokens(query)
    if not query_tokens:
        return False
    memory_tokens = set().union(*(tokens(memory_text(memory)) for memory in selected_memories))
    subgoal_tokens = tokens(retrieval_text(working_state))
    distinctive = subgoal_tokens - memory_tokens
    return (
        len(query_tokens & memory_tokens) >= 2
        and bool(distinctive)
        and not bool(query_tokens & distinctive)
    )


def audit_tool_action(
    *,
    assistant_messages: list[dict[str, Any]],
    tool_call: dict[str, Any] | None,
    parse_error: str | None,
    loop_history: list[dict[str, Any]],
    selected_memories: list[dict[str, Any]],
    working_state: dict[str, Any],
) -> dict[str, Any]:
    exact_duplicate = bool(
        tool_call
        and tool_fingerprint(tool_call) in history_tool_fingerprints(loop_history)
    )
    repeats_memory = repeats_memory_known_fact(tool_call, selected_memories, working_state)
    has_reasoning = any(
        not message.get("recipient") and bool(str(message.get("text") or "").strip())
        for message in assistant_messages
    )
    if parse_error:
        decision = "REJECT"
        reasons = [parse_error]
    elif exact_duplicate:
        decision = "REWRITE"
        reasons = ["exact tool call already occurred in the current Loop"]
    else:
        decision = "KEEP"
        reasons = []
    return {
        "decision": decision,
        "reasons": reasons,
        "tool_call_valid": parse_error is None,
        "has_reasoning_message": has_reasoning,
        "exact_duplicate_tool_call_in_loop": exact_duplicate,
        "repeats_selected_memory_fact": repeats_memory,
        "review_warnings": (
            ["search overlaps a known Memory fact; verify that it serves the current evidence gap"]
            if repeats_memory
            else []
        ),
    }


def load_full_replay(
    pools: list[Path],
    segments: list[Path],
    overrides_path: Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    pool_by_source: dict[tuple[str, int], dict[str, Any]] = {}
    for path in pools:
        for row in read_jsonl(path):
            pool_by_source[(qid_of(row), int(row["source"]["step_index"]))] = row
    segment_by_qid: dict[str, dict[str, Any]] = {}
    for path in segments:
        for row in read_jsonl(path):
            segment_by_qid[qid_of(row)] = row
    if overrides_path:
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        apply_contract_overrides(segment_by_qid, overrides)
    grouped, segment_map, violations = validate_pool(
        list(pool_by_source.values()), list(segment_by_qid.values())
    )
    if violations:
        raise ValueError(f"Controller replay has {len(violations)} violations: {violations[:5]}")
    return grouped, segment_map


def controller_training_refs(
    path: Path,
) -> tuple[list[str], dict[tuple[str, int], str]]:
    ordered: list[str] = []
    refs: dict[tuple[str, int], str] = {}
    for row in read_jsonl(path):
        qid = qid_of(row)
        if qid not in ordered:
            ordered.append(qid)
        refs[(qid, int(row["source"]["step_index"]))] = str(row["sample_id"])
    return ordered, refs


def build_tool_samples(
    grouped: dict[str, list[dict[str, Any]]],
    qids: list[str],
    controller_refs: dict[tuple[str, int], str],
    *,
    max_history_messages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for qid in qids:
        group = sorted(grouped[qid], key=lambda row: int(row["source"]["step_index"]))
        history_by_loop: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            step = int(row["source"]["step_index"])
            loop_id = str(row["input"]["working_state_before"]["loop_id"])
            history = history_by_loop[loop_id]
            observed = row["input"].get("observed_messages") or []
            targets = assistant_targets(observed)
            observation = tool_observation(observed)
            tool_call, parse_error = parse_tool_call(targets)
            available = row["input"].get("available_cross_loop_memories") or []
            selected, retrieval_audit = select_causal_memories(
                available, row["input"]["working_state_before"]
            )
            quality = audit_tool_action(
                assistant_messages=targets,
                tool_call=tool_call,
                parse_error=parse_error,
                loop_history=history,
                selected_memories=selected,
                working_state=row["input"]["working_state_before"],
            )
            sample_id = f"research_{len(candidates) + 1:06d}"
            preceding = f"decision_{step - 1:03d}" if step > 0 else None
            candidate = {
                "sample_id": sample_id,
                "source": {
                    "dataset": "OpenResearcher/OpenResearcher-Dataset",
                    "trajectory_id": f"openresearcher_qid_{qid}",
                    "qid": row["source"]["qid"],
                    "research_action_index": step,
                    "target_assistant_message_ids": [
                        message["message_id"] for message in targets
                    ],
                    "expected_tool_observation_id": (
                        observation["message_id"] if observation else None
                    ),
                },
                "alignment": {
                    "preceding_controller_decision_id": preceding,
                    "preceding_controller_training_sample_id": controller_refs.get(
                        (qid, step - 1)
                    ),
                    "observation_controller_decision_id": f"decision_{step:03d}",
                    "observation_controller_training_sample_id": controller_refs.get(
                        (qid, step)
                    ),
                    "state_ref": (
                        f"openresearcher_qid_{qid}:decision_{step - 1:03d}:after"
                        if step > 0
                        else f"openresearcher_qid_{qid}:initial_state"
                    ),
                    "memory_snapshot_id": f"openresearcher_qid_{qid}:before_action_{step:03d}",
                    "observed_through_before_action": (
                        max((int(message["index"]) for message in history), default=0)
                    ),
                },
                "input": {
                    "question": row["input"]["question"],
                    "global_intent_state": deepcopy(
                        row["input"]["global_intent_state_before"]
                    ),
                    "working_state": deepcopy(row["input"]["working_state_before"]),
                    "current_loop_messages": deepcopy(history[-max_history_messages:]),
                    "retrieval_query": retrieval_text(
                        row["input"]["working_state_before"]
                    ),
                    "retrieved_memories": selected,
                    "available_memory_ids": [
                        str(memory["memory_id"]) for memory in available
                    ],
                    "tools": ["browser.search", "browser.open", "browser.find"],
                },
                "target": {
                    "action_type": "TOOL_CALL",
                    "assistant_messages": targets,
                    "tool_call": tool_call,
                },
                "quality_gate": quality,
                "retrieval_audit": retrieval_audit,
            }
            candidates.append(candidate)
            alignment.append(
                {
                    "trajectory_id": candidate["source"]["trajectory_id"],
                    "researcher_sample_id": sample_id,
                    "controller_decision_id": preceding,
                    "preceding_controller_training_sample_id": controller_refs.get(
                        (qid, step - 1)
                    ),
                    "observation_controller_decision_id": f"decision_{step:03d}",
                    "observation_controller_training_sample_id": controller_refs.get(
                        (qid, step)
                    ),
                    "state_ref": candidate["alignment"]["state_ref"],
                    "memory_ids": [
                        str(memory["memory_id"]) for memory in selected
                    ],
                    "target_assistant_message_ids": candidate["source"][
                        "target_assistant_message_ids"
                    ],
                }
            )
            history.extend(deepcopy(observed))
    return candidates, alignment


def raw_matches_replay(raw: dict[str, Any], replay: list[dict[str, Any]]) -> bool:
    """Disambiguate same-qid trajectories by their exact observed messages."""

    raw_messages = raw.get("messages") or []
    expected: dict[int, dict[str, Any]] = {}
    for decision in replay:
        for message in decision["input"].get("observed_messages") or []:
            expected[int(message["index"])] = message
    if not expected or max(expected) > len(raw_messages):
        return False
    for index, message in expected.items():
        # Retrospective pools compacted assistant/tool messages to 1,200/4,000
        # characters. Reproduce that exact stored prefix instead of comparing
        # it with the default 8,000-character representation.
        text_limit = max(1, len(str(message.get("text") or "")))
        actual = compact_message(raw_messages[index - 1], index, text_limit=text_limit)
        for field in ("role", "recipient", "channel", "text"):
            if actual.get(field) != message.get(field):
                return False
    return True


def load_raw_selected(
    input_glob: str,
    grouped: dict[str, list[dict[str, Any]]],
    wanted: set[str],
) -> dict[str, dict[str, Any]]:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise ValueError(f"No raw parquet files matched: {input_glob}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on server environment
        raise RuntimeError("pyarrow is required to read raw trajectories") from exc
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        remaining = wanted - set(found)
        if not remaining:
            break
        parquet = pq.ParquetFile(path)
        qid_type = parquet.schema_arrow.field("qid").type
        filter_values: list[Any]
        if pa.types.is_integer(qid_type):
            filter_values = [int(qid) for qid in remaining]
        else:
            filter_values = sorted(remaining)
        table = pq.read_table(
            path,
            columns=["qid", "messages"],
            filters=[("qid", "in", filter_values)],
        )
        for row in table.to_pylist():
            qid = str(row.get("qid"))
            if qid not in remaining or not raw_matches_replay(row, grouped[qid]):
                continue
            row["_raw_source_path"] = path
            found[qid] = row
    return found


def raw_final_message(
    row: dict[str, Any], *, after_message_index: int = 0
) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            index + 1 > after_message_index
            and message.get("role") == "assistant"
            and message.get("channel") == "final"
        ):
            return compact_message(message, index + 1, text_limit=12000)
    return None


def append_final_samples(
    candidates: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    qids: list[str],
    raw_rows: dict[str, dict[str, Any]],
    controller_refs: dict[tuple[str, int], str],
    *,
    max_history_messages: int,
) -> None:
    for qid in qids:
        raw = raw_rows.get(qid)
        group = sorted(grouped[qid], key=lambda row: int(row["source"]["step_index"]))
        last = group[-1]
        if last["target"]["loop_decision"]["action"] != "READY_TO_ANSWER":
            continue
        final_message = (
            raw_final_message(
                raw,
                after_message_index=int(last["source"]["prefix_end_message_index"]),
            )
            if raw
            else None
        )
        if final_message is None:
            continue
        loop_id = str(last["input"]["working_state_before"]["loop_id"])
        loop_history: list[dict[str, Any]] = []
        for row in group:
            if str(row["input"]["working_state_before"]["loop_id"]) == loop_id:
                loop_history.extend(deepcopy(row["input"].get("observed_messages") or []))
        memories = deepcopy(last["input"].get("available_cross_loop_memories") or [])
        emitted = last["target"].get("cross_loop_memory")
        if isinstance(emitted, dict):
            memories.append(deepcopy(emitted))
        step = int(last["source"]["step_index"])
        sample_id = f"research_{len(candidates) + 1:06d}"
        candidate = {
            "sample_id": sample_id,
            "source": {
                "dataset": "OpenResearcher/OpenResearcher-Dataset",
                "trajectory_id": f"openresearcher_qid_{qid}",
                "qid": last["source"]["qid"],
                "research_action_index": step + 1,
                "target_assistant_message_ids": [final_message["message_id"]],
                "expected_tool_observation_id": None,
            },
            "alignment": {
                "preceding_controller_decision_id": f"decision_{step:03d}",
                "preceding_controller_training_sample_id": controller_refs.get(
                    (qid, step)
                ),
                "observation_controller_decision_id": None,
                "observation_controller_training_sample_id": None,
                "state_ref": f"openresearcher_qid_{qid}:decision_{step:03d}:after",
                "memory_snapshot_id": f"openresearcher_qid_{qid}:before_final_answer",
                "observed_through_before_action": int(
                    last["source"]["prefix_end_message_index"]
                ),
            },
            "input": {
                "question": last["input"]["question"],
                "global_intent_state": apply_delta(
                    last["input"]["global_intent_state_before"],
                    last["target"]["state_delta"]["operations"],
                ),
                "working_state": deepcopy(last["target"]["working_state_after"]),
                "current_loop_messages": loop_history[-max_history_messages:],
                "retrieval_query": last["input"]["question"],
                "retrieved_memories": memories[-8:],
                "available_memory_ids": [
                    str(memory["memory_id"]) for memory in memories[-8:]
                ],
                "tools": [],
            },
            "target": {
                "action_type": "FINAL_ANSWER",
                "assistant_messages": [final_message],
                "tool_call": None,
            },
            "quality_gate": {
                "decision": "KEEP",
                "reasons": [],
                "tool_call_valid": True,
                "has_reasoning_message": False,
                "exact_duplicate_tool_call_in_loop": False,
                "repeats_selected_memory_fact": False,
                "review_warnings": [],
            },
            "retrieval_audit": [],
        }
        candidates.append(candidate)
        alignment.append(
            {
                "trajectory_id": candidate["source"]["trajectory_id"],
                "researcher_sample_id": sample_id,
                "controller_decision_id": f"decision_{step:03d}",
                "preceding_controller_training_sample_id": controller_refs.get(
                    (qid, step)
                ),
                "observation_controller_decision_id": None,
                "observation_controller_training_sample_id": None,
                "state_ref": candidate["alignment"]["state_ref"],
                "memory_ids": candidate["input"]["available_memory_ids"],
                "target_assistant_message_ids": [final_message["message_id"]],
            }
        )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict[str, Any]], *, count: int) -> None:
    lines = ["# Memory-conditioned Researcher samples", ""]
    for row in rows[:count]:
        source = row["source"]
        lines.extend(
            [
                f"## {row['sample_id']} · QID {source['qid']}",
                "",
                f"- Action: `{row['target']['action_type']}`",
                f"- Quality: `{row['quality_gate']['decision']}`",
                f"- State: `{row['alignment']['state_ref']}`",
                f"- Memories: `{', '.join(row['input']['available_memory_ids']) or 'none'}`",
                f"- Subgoal: {row['input']['working_state'].get('current_subgoal', '')}",
                "",
                "```json",
                json.dumps(row, ensure_ascii=False, indent=2)[:10000],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def finalize_training_rows(
    candidates: list[dict[str, Any]], alignment: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign stable training ids without mutating candidate lineage."""

    training_rows: list[dict[str, Any]] = []
    candidate_to_training: dict[str, str] = {}
    for candidate in candidates:
        if candidate["quality_gate"]["decision"] != "KEEP":
            continue
        row = deepcopy(candidate)
        candidate_id = str(candidate["sample_id"])
        training_id = f"research_train_{len(training_rows) + 1:06d}"
        row["sample_id"] = training_id
        row["lineage"] = {"candidate_sample_id": candidate_id}
        candidate_to_training[candidate_id] = training_id
        training_rows.append(row)

    quality_by_candidate = {
        str(row["sample_id"]): str(row["quality_gate"]["decision"])
        for row in candidates
    }
    for record in alignment:
        candidate_id = str(record.pop("researcher_sample_id"))
        record["candidate_sample_id"] = candidate_id
        record["training_sample_id"] = candidate_to_training.get(candidate_id)
        record["quality_decision"] = quality_by_candidate[candidate_id]
    return training_rows


def validate_outputs(
    candidates: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> list[str]:
    violations: list[str] = []
    candidate_ids = [str(row["sample_id"]) for row in candidates]
    training_ids = [str(row["sample_id"]) for row in training_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        violations.append("candidate sample ids are not unique")
    if len(training_ids) != len(set(training_ids)):
        violations.append("training sample ids are not unique")
    if {str(row["candidate_sample_id"]) for row in alignment} != set(candidate_ids):
        violations.append("alignment does not cover every candidate exactly")

    aligned_training_ids = {
        str(row["training_sample_id"])
        for row in alignment
        if row.get("training_sample_id")
    }
    if aligned_training_ids != set(training_ids):
        violations.append("alignment training ids do not match final dataset")

    for row in training_rows:
        sample_id = str(row["sample_id"])
        observed_through = int(row["alignment"]["observed_through_before_action"])
        history = row["input"].get("current_loop_messages") or []
        targets = row["target"].get("assistant_messages") or []
        history_ids = {str(message["message_id"]) for message in history}
        target_ids = {str(message["message_id"]) for message in targets}
        if history_ids & target_ids:
            violations.append(f"{sample_id}: target leaks into input history")
        if any(int(message["index"]) > observed_through for message in history):
            violations.append(f"{sample_id}: input history crosses causal boundary")
        if any(int(message["index"]) <= observed_through for message in targets):
            violations.append(f"{sample_id}: target is not after causal boundary")
        selected_ids = {
            str(memory["memory_id"])
            for memory in row["input"].get("retrieved_memories") or []
        }
        available_ids = {
            str(memory_id) for memory_id in row["input"].get("available_memory_ids") or []
        }
        if not selected_ids <= available_ids:
            violations.append(f"{sample_id}: retrieved Memory was not available")
        action_type = row["target"]["action_type"]
        if action_type == "TOOL_CALL" and row["target"].get("tool_call") is None:
            violations.append(f"{sample_id}: TOOL_CALL has no parsed tool call")
        if action_type == "FINAL_ANSWER" and row["target"].get("tool_call") is not None:
            violations.append(f"{sample_id}: FINAL_ANSWER unexpectedly has a tool call")
    return violations


def main() -> None:
    args = parse_args()
    grouped, _ = load_full_replay(args.pool, args.segments, args.contract_overrides)
    qids, controller_refs = controller_training_refs(args.controller_training)
    missing = [qid for qid in qids if qid not in grouped]
    if missing:
        raise SystemExit(f"Full replay is missing selected qids: {missing}")
    candidates, alignment = build_tool_samples(
        grouped,
        qids,
        controller_refs,
        max_history_messages=args.max_history_messages,
    )
    raw_rows: dict[str, dict[str, Any]] = {}
    if args.raw_input_glob:
        raw_rows = load_raw_selected(args.raw_input_glob, grouped, set(qids))
        append_final_samples(
            candidates,
            alignment,
            grouped,
            qids,
            raw_rows,
            controller_refs,
            max_history_messages=args.max_history_messages,
        )
    kept = finalize_training_rows(candidates, alignment)
    violations = validate_outputs(candidates, kept, alignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output.with_suffix(".candidates.jsonl")
    alignment_path = args.output.with_suffix(".alignment.jsonl")
    write_jsonl(candidate_path, candidates)
    write_jsonl(args.output, kept)
    write_jsonl(alignment_path, alignment)
    write_preview(args.output.with_suffix(".preview.md"), kept, count=args.preview_count)
    decisions = Counter(row["quality_gate"]["decision"] for row in candidates)
    action_types = Counter(row["target"]["action_type"] for row in kept)
    tools = Counter(
        row["target"]["tool_call"]["name"]
        for row in kept
        if row["target"]["tool_call"]
    )
    memory_counts = Counter(len(row["input"]["retrieved_memories"]) for row in kept)
    samples_by_qid = Counter(str(row["source"]["qid"]) for row in kept)
    final_answer_qids = {
        str(row["source"]["qid"])
        for row in kept
        if row["target"]["action_type"] == "FINAL_ANSWER"
    }
    paired_controller_ids = {
        str(row["alignment"]["observation_controller_training_sample_id"])
        for row in candidates
        if row["alignment"].get("observation_controller_training_sample_id")
    }
    kept_paired_controller_ids = {
        str(row["alignment"]["observation_controller_training_sample_id"])
        for row in kept
        if row["alignment"].get("observation_controller_training_sample_id")
    }
    kept_controller_reference_ids = {
        str(controller_id)
        for row in kept
        for controller_id in (
            row["alignment"].get("preceding_controller_training_sample_id"),
            row["alignment"].get("observation_controller_training_sample_id"),
        )
        if controller_id
    }
    report = {
        "valid": not violations,
        "violations": violations[:20],
        "controller_trajectories": len(qids),
        "controller_training_samples": len(controller_refs),
        "controller_samples_paired_in_candidates": len(paired_controller_ids),
        "controller_samples_paired_in_training": len(kept_paired_controller_ids),
        "controller_samples_referenced_in_training": len(
            kept_controller_reference_ids
        ),
        "unpaired_controller_training_sample_ids": sorted(
            set(controller_refs.values()) - paired_controller_ids
        ),
        "controller_training_sample_ids_excluded_by_quality_gate": sorted(
            paired_controller_ids - kept_paired_controller_ids
        ),
        "candidate_samples": len(candidates),
        "training_samples": len(kept),
        "quality_decisions": dict(decisions),
        "action_types": dict(action_types),
        "tool_counts": dict(tools),
        "retrieved_memory_count_distribution": dict(memory_counts),
        "memory_conditioned_samples": sum(
            bool(row["input"]["retrieved_memories"]) for row in kept
        ),
        "memory_overlap_review_warnings": sum(
            bool(row["quality_gate"].get("review_warnings")) for row in kept
        ),
        "samples_by_qid_min": min(samples_by_qid.values(), default=0),
        "samples_by_qid_max": max(samples_by_qid.values(), default=0),
        "raw_trajectories_found": len(raw_rows),
        "raw_source_paths": sorted(
            {str(row.get("_raw_source_path")) for row in raw_rows.values()}
        ),
        "missing_raw_qids": sorted(set(qids) - set(raw_rows)),
        "missing_post_boundary_final_answer_qids": sorted(set(qids) - final_answer_qids),
        "output": str(args.output),
        "candidates": str(candidate_path),
        "alignment": str(alignment_path),
    }
    args.output.with_suffix(".qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(f"Researcher dataset has {len(violations)} validation violations")


if __name__ == "__main__":
    main()
