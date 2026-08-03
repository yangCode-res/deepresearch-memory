"""Shared serialization helpers for Memory Controller supervised fine-tuning."""

from __future__ import annotations

import json
from typing import Any


CONTROLLER_SFT_SYSTEM_PROMPT = """You are the Memory Controller for a deep-research agent.
Given one causal controller input, return exactly one valid JSON object and no other text.
Use only evidence present in the input. Never invent facts, citations, message IDs, or memory IDs.

The output must contain exactly these top-level fields:
- working_state_after
- loop_decision
- state_delta
- cross_loop_memory
- retrieval

loop_decision.action must be exactly one of CONTINUE_CURRENT_LOOP, SWITCH_LOOP, or READY_TO_ANSWER.
CONTINUE_CURRENT_LOOP keeps the same coherent information subgoal.
SWITCH_LOOP ends a resolved, refuted, blocked, or superseded subgoal and starts a genuinely different one.
READY_TO_ANSWER is allowed only when the answer is stable and supported by sufficient evidence.
StateDelta records only compact durable findings. Select only memory IDs supplied in available_cross_loop_memories.
Do not reveal chain-of-thought."""


def training_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Serialize one curated replay decision as a three-message SFT example."""

    return [
        {"role": "system", "content": CONTROLLER_SFT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(row["input"], ensure_ascii=False, separators=(",", ":")),
        },
        {
            "role": "assistant",
            "content": json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object, tolerating a surrounding Markdown fence."""

    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    start = value.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def trajectory_id(row: dict[str, Any]) -> str:
    source = row.get("source") or {}
    return str(source.get("trajectory_id") or source.get("qid") or "")


__all__ = [
    "CONTROLLER_SFT_SYSTEM_PROMPT",
    "parse_json_object",
    "training_messages",
    "trajectory_id",
]
