#!/usr/bin/env python3
"""Rewrite weak retrospective Loop contracts without changing their boundaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
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
from build_working_state_retrospective import (  # noqa: E402
    scrub_future_literals,
)


SYSTEM_PROMPT = """You repair information-outcome contracts for an already fixed deep-research Loop segmentation.
Return valid JSON only. Do not reveal chain-of-thought. Do not change Loop count, numbering, ranges, or actions.

For every Loop, write:
- subgoal: the specific, independently decidable information objective pursued in that Loop.
- completion_test: the observable information condition that tells the controller the subgoal is resolved.

The contract must be useful to a controller before the Loop runs. Describe the unknown relation or attribute to
establish, not a search query, website, source, page, tool operation, or browser action. Do not write generic
placeholders such as "information dependency N", "remaining evidence", "requested value", or "evidence resolves".
Do not include an answer literal unless it is already present in activation_evidence. A verification Loop may name
a candidate already present in activation_evidence. Phrase completion tests as "Evidence establishes..." rather
than referring to a source or page. Keep each field concise and distinct from adjacent Loops."""


WEAK_PATTERN = re.compile(
    r"information dependency\s*\d*|evidence resolves|remaining evidence|"
    r"the requested value|the available evidence",
    re.IGNORECASE,
)

REPAIR_FORBIDDEN_PATTERN = re.compile(
    r"https?://|\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)\b|\bbrowser\.|"
    r"\b(?:search|query|open|view|visit|click|browse|"
    r"url|webpage|wikipedia|google|bing|search\s+result|snippet|doc\s*\d+)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-qids", nargs="*", default=[])
    parser.add_argument("--model", default=os.getenv("CL_GISM_DATA_MODEL", "mimo-v2.5-pro"))
    parser.add_argument("--base-url", default=os.getenv("CL_GISM_CONTROLLER_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("CL_GISM_CONTROLLER_API_KEY"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qid_of(row: dict[str, Any]) -> str:
    return str(row["source"]["qid"])


def normalize_repair_text(value: Any) -> str:
    """Turn source-centric completion wording into an information condition."""

    text = " ".join(str(value or "").strip().split())
    replacements = (
        (r"\banother source or the same source\b", "corroborating evidence"),
        (r"\ba separate source\b", "independent evidence"),
        (r"\banother source\b", "corroborating evidence"),
        (
            r"\b(?:an?\s+)?(?:primary|authoritative|independent|additional|multiple|corroborating|reliable|credible|trustworthy)\s+sources?\b",
            "evidence",
        ),
        (r"\b(?:a|the) source\b", "evidence"),
        (r"\bsource\b", "evidence"),
        (r"\bsources\b", "evidence"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def compact_activation(group: list[dict[str, Any]], start: int) -> str:
    question = str(group[0]["input"].get("question") or "")
    if start == 0:
        return question
    parts = [question]
    for row in group[:start]:
        parts.extend(str(message.get("text") or "") for message in row["input"].get("observed_messages") or [])
    return "\n".join(parts)[-14000:]


def validate_repair(
    raw: dict[str, Any],
    *,
    original_loops: list[dict[str, Any]],
    activation_texts: dict[int, str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"loops"} or not isinstance(raw["loops"], list):
        raise ValueError("response must contain exactly one loops array")
    if len(raw["loops"]) != len(original_loops):
        raise ValueError("repair must preserve Loop count")
    repaired: list[dict[str, Any]] = []
    for original, candidate in zip(original_loops, raw["loops"]):
        if not isinstance(candidate, dict) or set(candidate) != {
            "loop_number", "subgoal", "completion_test"
        }:
            raise ValueError("each repaired Loop must contain loop_number, subgoal, and completion_test")
        loop_number = int(candidate["loop_number"])
        if loop_number != int(original["loop_number"]):
            raise ValueError("repair must preserve Loop numbering")
        subgoal = normalize_repair_text(candidate["subgoal"]).rstrip(" .;；。")
        visible = activation_texts[loop_number]
        subgoal = scrub_future_literals(subgoal, visible_text=visible)
        subgoal = re.sub(
            r"\b(?:the\s+)?requested value\s+film\b",
            "the specified film",
            subgoal,
            flags=re.IGNORECASE,
        )
        subgoal = re.sub(r"\bthe\s+the\s+specified\b", "the specified", subgoal, flags=re.IGNORECASE)
        completion = f"Evidence is sufficient to {subgoal[:1].lower() + subgoal[1:]}"
        combined = f"{subgoal}\n{completion}"
        if not subgoal or not completion:
            raise ValueError("repaired contracts cannot be empty")
        if WEAK_PATTERN.search(combined):
            raise ValueError("repaired contracts still contain generic or future-value placeholders")
        if REPAIR_FORBIDDEN_PATTERN.search(combined):
            raise ValueError("repaired contracts contain tool, source, or browsing language")
        repaired.append(
            {
                "loop_number": loop_number,
                "subgoal": subgoal,
                "completion_test": completion,
            }
        )
    return {"loops": repaired}


def main() -> None:
    args = parse_args()
    if not args.api_key or not args.base_url:
        raise SystemExit("controller API configuration is required")
    pool = read_jsonl(args.pool)
    segments = read_jsonl(args.segments)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        grouped[qid_of(row)].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["source"]["step_index"]))
    include = {str(qid) for qid in args.include_qids}
    client = OpenAIChatJSONClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=300,
        max_tokens=2048,
    )
    output = deepcopy(segments)
    repaired_qids: list[str] = []
    for segment_index, segment in enumerate(segments):
        qid = qid_of(segment)
        if include and qid not in include:
            continue
        group = grouped.get(qid)
        if not group:
            raise SystemExit(f"pool is missing qid {qid}")
        loops = segment["segmentation"]["loops"]
        activation_texts = {
            int(loop["loop_number"]): compact_activation(group, int(loop["start_decision_index"]))
            for loop in loops
        }
        payload = {
            "question": segment["question"],
            "trajectory_summary": segment["segmentation"]["trajectory_summary"],
            "fixed_loops": [
                {
                    "loop_number": loop["loop_number"],
                    "start_decision_index": loop["start_decision_index"],
                    "end_decision_index": loop["end_decision_index"],
                    "end_action": loop["end_action"],
                    "existing_subgoal": loop["subgoal"],
                    "boundary_reason": loop["boundary_reason"],
                    "activation_evidence": activation_texts[int(loop["loop_number"])],
                }
                for loop in loops
            ],
            "required_output": {
                "loops": [
                    {
                        "loop_number": loop["loop_number"],
                        "subgoal": "specific information objective",
                        "completion_test": "observable information condition",
                    }
                    for loop in loops
                ]
            },
        }
        repaired: dict[str, Any] | None = None
        repair_input = payload
        for attempt in range(3):
            raw = client.complete_json(SYSTEM_PROMPT, json.dumps(repair_input, ensure_ascii=False))
            try:
                repaired = validate_repair(
                    raw,
                    original_loops=loops,
                    activation_texts=activation_texts,
                )
                break
            except ValueError as exc:
                print(
                    "[contract-repair] validation failure "
                    f"qid={qid} attempt={attempt + 1}: {exc}; raw={json.dumps(raw, ensure_ascii=False)}",
                    flush=True,
                )
                repair_input = {
                    "validation_error": str(exc),
                    "invalid_response": raw,
                    "instruction": "Rewrite the complete JSON object and remove the invalid phrasing.",
                    "original_input": payload,
                }
        if repaired is None:
            raise ValueError(f"contract repair failed after 3 attempts for qid={qid}")
        updated = deepcopy(segment)
        by_number = {int(item["loop_number"]): item for item in repaired["loops"]}
        for loop in updated["segmentation"]["loops"]:
            contract = by_number[int(loop["loop_number"])]
            loop["subgoal"] = contract["subgoal"]
            loop["completion_test"] = contract["completion_test"]
        output[segment_index] = updated
        repaired_qids.append(qid)
        write_jsonl(args.output, output)
        print(f"[contract-repair] repaired qid={qid} loops={len(loops)}", flush=True)

    write_jsonl(args.output, output)
    report = {
        "valid": True,
        "model": args.model,
        "repaired_qids": repaired_qids,
        "api_requests": client.request_count,
        "prompt_tokens": client.total_prompt_tokens,
        "completion_tokens": client.total_completion_tokens,
        "total_tokens": client.total_tokens,
        "output": str(args.output),
    }
    args.output.with_suffix(".repair.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
