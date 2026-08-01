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

This is behavioral retrospective segmentation, not counterfactual optimization. Read the following assistant
message(s) after every tool result to determine what research phase the recorded agent actually entered next.
A Loop is one coherent, independently decidable information subgoal with one observable completion test. A
different query, tool, source, URL, failed search, or another verification of the same exact claim is not a new
Loop. Mark CONTINUE when the following agent work continues the same information objective. Mark SWITCH only
when the following agent work pursues a genuinely different, independently decidable dependency. Mark READY
only when the following agent content transitions to the final answer without another research tool call.

A completed source/entity localization stage followed by extraction of the requested fact from that identified
artifact IS a Loop boundary: locating the correct artifact and establishing the requested fact are separately
decidable dependencies. In contrast, once a fact-extraction or verification objective is active, opening another
source or repeating the same verification remains inside that Loop unless a different factual dependency begins.

Do not label an earlier point READY merely because hindsight reveals that its evidence could have been enough.
If the recorded agent performs later verification of the same claim, those decisions remain in the same Loop.

Decision indices refer only to tool-result messages marked with decision_index. Return contiguous Loop ranges
that start at decision 0, cover every decision exactly once, and never overlap. Every non-final Loop ends with
SWITCH_LOOP; the final Loop ends with READY_TO_ANSWER when the recorded trajectory transitions to a final answer.
Subgoals and completion tests must describe information outcomes; never name a website, URL, query, browser
operation, or tool. A Loop contract must not contain an answer value, date, number, name, or quoted phrase that
first appears in a later tool result; describe the unknown information to establish instead. The ranges are
retrospective annotations kept outside model-training input."""


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


SEGMENTATION_KEYS = {"trajectory_summary", "loops"}
LOOP_KEYS = {
    "loop_number",
    "start_decision_index",
    "end_decision_index",
    "subgoal",
    "completion_test",
    "end_action",
    "outcome",
    "boundary_basis",
    "boundary_reason",
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
        "loops": [
            {
                "loop_number": 1,
                "start_decision_index": 0,
                "end_decision_index": f"integer from 0 to {decision_count - 1}",
                "subgoal": "one independently decidable information objective",
                "completion_test": "observable evidence condition",
                "end_action": "SWITCH_LOOP|READY_TO_ANSWER|CONTINUE_CURRENT_LOOP",
                "outcome": "IN_PROGRESS|RESOLVED|REFUTED|BLOCKED|SUPERSEDED",
                "boundary_basis": (
                    "NONE|SUBGOAL_COMPLETED|SUBGOAL_CHANGED|CANDIDATE_CHANGED|"
                    "BLOCKED_OR_SATURATED|PHASE_TRANSITION|TASK_COMPLETE"
                ),
                "boundary_reason": "short explanation of how following agent work relates to the current Loop",
            }
        ],
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


GLOBAL_CONCRETE_PATTERN = re.compile(
    r"https?://|\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)\b|"
    r"\b(browser\.(?:search|open)|wikipedia|google|bing|search(?:ing)?|query(?:ing)?|"
    r"open(?:s|ed|ing)?|view(?:s|ed|ing)?|visit(?:s|ed|ing)?|browse|click(?:s|ed|ing)?|"
    r"read(?:ing)?|inspect(?:ed|ing)?|look(?:ing)?\s+up|source|website|webpage|"
    r"search\s+result|snippet|doc\s*\d+|url)\b|"
    r"搜索|查询|查找|打开|查看|访问|点击|来源|搜索结果|片段",
    re.IGNORECASE,
)

DURABLE_OPERATION_PATTERN = re.compile(
    r"https?://|\bbrowser\.|\bdoc\s*\d+\b|\bsearch\s+results?\b|"
    r"\bopened?\s+in\s+(?:the\s+)?browser\b|\bnext\s+step\b",
    re.IGNORECASE,
)

GENERIC_CONTRACT_PATTERN = re.compile(
    r"\binformation dependency\s*\d*\b|\bevidence resolves\b",
    re.IGNORECASE,
)

VERIFICATION_ONLY_SUBGOAL_PATTERN = re.compile(
    r"^(?:(?:independently|additionally)\s+)?"
    r"(?:verify|confirm|corroborate|validate|double-check|ensure)\b",
    re.IGNORECASE,
)

VERIFICATION_ONLY_QUALIFIER_PATTERN = re.compile(
    r"\b(?:sources?|references?|corroborat\w*|consisten\w*|accurac\w*|"
    r"same\s+(?:claim|answer|fact)|only\s+relevant\s+mention|re-?check\w*)\b",
    re.IGNORECASE,
)


def sanitize_contract_text(value: Any, *, fallback: str) -> str:
    """Convert tool/source phrasing into an information-objective contract."""

    text = _normalize_text(value).rstrip(" .;；。")
    text = re.sub(r"https?://\S+", "the relevant evidence", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:www\.)?[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)\b",
        "an available source",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(browser\.(?:search|open)|wikipedia|google|bing)\b",
        "the available evidence",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:search|establish)\s+for\s+additional\s+references\s+to\s+corroborate\s+(.+)$",
        r"Corroborate \1 with additional independent evidence",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^find\s+an?\s+independent\s+reference\s+confirming\s+(.+)$",
        r"Independently verify \1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^locate\s+a\s+worked\s+solution\s+for\s+(.+)$",
        r"Establish \1 using a worked calculation",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:find|locate|obtain|consult|check|use)\s+"
        r"(?:(?:an?|another|a\s+second|independent|reliable|authoritative)\s+)?source\s*"
        r"(?:that\s+|which\s+|confirming\s+|stating\s+|for\s+|about\s+)?",
        "Establish evidence that ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:view|open|inspect|read)\s+(?:a\s+)?(?:specific\s+)?"
        r"(?:doc(?:ument)?\s*\d+|search\s+result|snippet)\b",
        "establish the remaining evidence",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:an?\s+)?(?:website|webpage)\s+(?:is\s+)?opened\s+that\s+(?:contains|states)",
        "evidence directly states",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(^|[.!?;]\s*)(search|query|open|view|visit|browse|click|read|inspect|look\s+up|"
        r"use\s+(?:the\s+)?browser)\b",
        r"\1Establish",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(^|[。！？；]\s*)(搜索|查询|查找|打开|查看|访问|点击)", r"\1确定", text)
    text = _normalize_text(text).rstrip(" .;；。")
    fallback = _normalize_text(fallback).rstrip(" .;；。")
    if re.match(r"^Establish (?:for\b|results\b)", text, flags=re.IGNORECASE):
        return fallback
    return fallback if not text or GLOBAL_CONCRETE_PATTERN.search(text) else text


def subgoal_fallback(question: str, loop_number: int) -> str:
    if loop_number == 1 and re.search(
        r"\b(article|study|paper|report|document|titled|according\s+to)\b|文章|研究|论文|报告|题为",
        question,
        flags=re.IGNORECASE,
    ):
        return "Identify and access the specific artifact referenced by the user"
    return f"Establish information dependency {loop_number} required by the user question"


def completion_fallback(question: str, loop_number: int) -> str:
    if loop_number == 1 and re.search(
        r"\b(article|study|paper|report|document|titled|according\s+to)\b|文章|研究|论文|报告|题为",
        question,
        flags=re.IGNORECASE,
    ):
        return "The referenced artifact is uniquely identified and accessible in the observed evidence"
    return f"Evidence resolves information dependency {loop_number}"


def scrub_future_literals(value: str, *, visible_text: str) -> str:
    """Remove answer-like literals unavailable when a Loop contract becomes active."""

    visible = visible_text.casefold()
    text = value

    def scrub_quote(match: re.Match[str]) -> str:
        phrase = _normalize_text(match.group(2))
        return match.group(0) if phrase.casefold() in visible else "the requested value"

    text = re.sub(r"(['\"])([^'\"]{3,80})\1", scrub_quote, text)
    month = (
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    )

    def scrub_date(match: re.Match[str]) -> str:
        phrase = _normalize_text(match.group(0))
        return phrase if phrase.casefold() in visible else "the requested date or period"

    text = re.sub(
        rf"\b{month}(?:\s+(?:and|to|through|[-–])\s+{month})?(?:\s+\d{{4}})?\b",
        scrub_date,
        text,
        flags=re.IGNORECASE,
    )

    def scrub_number(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.casefold() in visible else "the requested value"

    text = re.sub(r"(?<!\w)\d[\d,.:-]*%?(?![\w%])", scrub_number, text)
    return _normalize_text(text)


def validate_segmentation(
    raw: dict[str, Any],
    *,
    decision_count: int,
    decision_message_limits: list[int] | None = None,
    trajectory_message_limit: int | None = None,
    has_final_answer: bool | None = None,
    question: str = "",
    trajectory_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != SEGMENTATION_KEYS:
        raise ValueError("segmentation response has missing or extra fields")
    summary = _normalize_text(raw["trajectory_summary"])
    loops = raw.get("loops")
    if not summary or not isinstance(loops, list) or not loops:
        raise ValueError("segmentation requires a summary and at least one Loop")
    if decision_message_limits is not None and len(decision_message_limits) != decision_count:
        raise ValueError("decision_message_limits must cover the complete trajectory")

    normalized: list[dict[str, Any]] = []
    expected_start = 0
    for position, item in enumerate(loops, start=1):
        if not isinstance(item, dict) or set(item) != LOOP_KEYS:
            raise ValueError("a Loop has missing or extra fields")
        try:
            loop_number = int(item["loop_number"])
            start = int(item["start_decision_index"])
            end = int(item["end_decision_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Loop numbers and decision indices must be integers") from exc
        if loop_number != position:
            raise ValueError("Loop numbers must be consecutive starting at 1")
        if start != expected_start or end < start or end >= decision_count:
            raise ValueError("Loops must contiguously cover decision points from decision 0")
        subgoal = sanitize_contract_text(
            item["subgoal"],
            fallback=subgoal_fallback(question, loop_number),
        )
        completion_test = sanitize_contract_text(
            item["completion_test"],
            fallback=completion_fallback(question, loop_number),
        )
        if decision_message_limits is not None:
            visible_end = decision_message_limits[0 if start == 0 else start - 1]
            visible_parts = [question]
            for message in trajectory_messages or []:
                if int(message.get("index") or 0) <= visible_end:
                    visible_parts.append(str(message.get("text") or ""))
            visible_text = "\n".join(visible_parts)
            subgoal = scrub_future_literals(subgoal, visible_text=visible_text)
            completion_test = scrub_future_literals(completion_test, visible_text=visible_text)
        reason = _normalize_text(item["boundary_reason"])
        if not subgoal or not completion_test or not reason:
            raise ValueError("Loop contract and boundary reason cannot be empty")
        if GENERIC_CONTRACT_PATTERN.search(f"{subgoal}\n{completion_test}"):
            raise ValueError("Loop contracts must name the specific information objective")
        action = str(item["end_action"] or "").upper()
        outcome = str(item["outcome"] or "").upper()
        basis = str(item["boundary_basis"] or "").upper()
        if action not in ACTIONS:
            raise ValueError("invalid Loop end_action")
        is_last = position == len(loops)
        if not is_last and action != "SWITCH_LOOP":
            raise ValueError("every non-final Loop must end with SWITCH_LOOP")
        if is_last and action == "SWITCH_LOOP":
            raise ValueError("the final Loop cannot end with SWITCH_LOOP")
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
        normalized.append({
            "loop_number": loop_number,
            "start_decision_index": start,
            "end_decision_index": end,
            "subgoal": subgoal,
            "completion_test": completion_test,
            "end_action": action,
            "outcome": outcome,
            "boundary_basis": basis,
            "boundary_reason": reason,
        })
        expected_start = end + 1

    final = normalized[-1]
    if final["end_decision_index"] != decision_count - 1:
        raise ValueError("Loop ranges must cover every tool-result decision")
    if has_final_answer is True and final["end_action"] != "READY_TO_ANSWER":
        raise ValueError("a successful trajectory ending in a final answer requires READY at its last decision")
    if has_final_answer is False and final["end_action"] == "READY_TO_ANSWER":
        raise ValueError("READY requires a following final-answer transition")
    for left, right in zip(normalized, normalized[1:]):
        if (
            VERIFICATION_ONLY_SUBGOAL_PATTERN.search(right["subgoal"])
            and VERIFICATION_ONLY_QUALIFIER_PATTERN.search(right["subgoal"])
        ):
            raise ValueError(
                "same-claim verification must remain in the preceding Loop rather than form a new Loop"
            )
        if semantic_similarity(left["subgoal"], right["subgoal"]) >= 0.72:
            raise ValueError("adjacent Loops appear to be source/query rephrasings of the same subgoal")

    raw["trajectory_summary"] = summary
    raw["loops"] = normalized
    return raw


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


def decision_lookahead_view(
    messages: list[dict[str, Any]], *, assistant_limit: int = 1000
) -> list[dict[str, Any]]:
    """Expose the recorded next assistant work explicitly to the segmenter."""

    tool_positions = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    result: list[dict[str, Any]] = []
    for decision_index, tool_pos in enumerate(tool_positions):
        next_tool_pos = tool_positions[decision_index + 1] if decision_index + 1 < len(tool_positions) else len(messages)
        following: list[dict[str, Any]] = []
        for pos in range(tool_pos + 1, next_tool_pos):
            if messages[pos].get("role") == "assistant":
                following.append(compact_message(messages[pos], pos + 1, text_limit=assistant_limit))
        result.append(
            {
                "decision_index": decision_index,
                "tool_message_id": f"msg_{tool_pos + 1:04d}",
                "following_assistant_messages_before_next_tool": following,
            }
        )
    return result


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
    decision_lookahead: list[dict[str, Any]] | None = None,
    has_final_answer: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "complete_trajectory": messages,
        "decision_lookahead": decision_lookahead or [],
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
            trajectory_message_limit=max((int(item["index"]) for item in messages), default=None),
            has_final_answer=has_final_answer,
            question=question,
            trajectory_messages=messages,
        ),
    )


def loop_for_decision(segmentation: dict[str, Any], decision_index: int) -> dict[str, Any]:
    for loop in segmentation["loops"]:
        if loop["start_decision_index"] <= decision_index <= loop["end_decision_index"]:
            return loop
    raise ValueError(f"decision {decision_index} is not covered by the retrospective segmentation")


def boundary_for_decision(segmentation: dict[str, Any], decision_index: int) -> dict[str, Any]:
    loops = segmentation["loops"]
    current = loop_for_decision(segmentation, decision_index)
    at_end = decision_index == current["end_decision_index"]
    action = current["end_action"] if at_end else "CONTINUE_CURRENT_LOOP"
    next_loop = loops[current["loop_number"]] if action == "SWITCH_LOOP" else None
    return {
        "action": action,
        "reason": "",
        "current_subgoal": current["subgoal"],
        "current_completion_test": current["completion_test"],
        "next_subgoal": next_loop["subgoal"] if next_loop else "",
        "next_completion_test": next_loop["completion_test"] if next_loop else "",
        "outcome": current["outcome"] if at_end else "IN_PROGRESS",
        "boundary_basis": current["boundary_basis"] if at_end else "NONE",
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
    if not isinstance(raw, dict):
        raise ValueError("causal-state response must be an object")
    action = boundary["action"]
    default_reasons = {
        "CONTINUE_CURRENT_LOOP": "The recorded next work remains within the same information subgoal.",
        "SWITCH_LOOP": "The current information subgoal ends before a distinct dependency begins.",
        "READY_TO_ANSWER": "The recorded trajectory transitions from research to the final answer.",
    }
    reason = default_reasons[action]

    latest_id = max(seen_message_ids) if seen_message_ids else "msg_0000"

    def replace_unseen_ids(value: Any) -> str:
        text = _normalize_text(value)
        return re.sub(
            r"\bmsg_\d{4}\b",
            lambda match: match.group(0) if match.group(0) in seen_message_ids else latest_id,
            text,
        )

    def cited(value: Any) -> str:
        text = replace_unseen_ids(value)
        if text and not re.search(r"\bmsg_\d{4}\b", text):
            text = f"{text} [{latest_id}]"
        return text

    progress_raw = raw.get("progress") if isinstance(raw.get("progress"), dict) else {}
    progress = {
        "progress_summary": replace_unseen_ids(
            progress_raw.get("progress_summary")
            or f"The latest observation at {latest_id} was incorporated into the current Loop."
        ),
        "resolved_aspects": progress_raw.get("resolved_aspects") or [],
        "open_aspects": progress_raw.get("open_aspects") or [],
        "key_evidence": progress_raw.get("key_evidence") or [],
        "candidate_answer": _normalize_text(progress_raw.get("candidate_answer")),
        "active_hypotheses": progress_raw.get("active_hypotheses") or [],
        "failed_strategies": progress_raw.get("failed_strategies") or [],
        "evidence_gaps": progress_raw.get("evidence_gaps") or [],
        "answer_stable": bool(progress_raw.get("answer_stable")),
        "evidence_sufficient": bool(progress_raw.get("evidence_sufficient")),
        "expected_information_gain": str(
            progress_raw.get("expected_information_gain") or "HIGH"
        ).upper(),
    }
    for name in (
        "resolved_aspects",
        "open_aspects",
        "key_evidence",
        "active_hypotheses",
        "failed_strategies",
        "evidence_gaps",
    ):
        values = progress[name] if isinstance(progress[name], list) else []
        transform = cited if name == "key_evidence" else replace_unseen_ids
        progress[name] = [transform(item)[:500] for item in values if _normalize_text(item)][:8]
    progress["open_aspects"] = [
        sanitize_contract_text(item, fallback=boundary["current_subgoal"])
        for item in progress["open_aspects"]
    ]
    progress["evidence_gaps"] = [
        sanitize_contract_text(item, fallback=boundary["current_completion_test"])
        for item in progress["evidence_gaps"]
    ]
    gain = str(progress["expected_information_gain"] or "").upper()
    if gain not in GAIN_LEVELS:
        gain = "HIGH"
    progress["expected_information_gain"] = gain
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
        if action == "CONTINUE_CURRENT_LOOP":
            gain_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            previous_gain = str(
                working_before.get("expected_information_gain") or "HIGH"
            ).upper()
            if previous_gain in gain_rank and (
                gain_rank[progress["expected_information_gain"]] > gain_rank[previous_gain]
            ):
                progress["expected_information_gain"] = previous_gain

    operations_raw = raw.get("durable_update") if isinstance(raw.get("durable_update"), dict) else {}
    operations: dict[str, Any] = {}
    for name in DELTA_FIELDS:
        if name == "completed_subgoal":
            operations[name] = _normalize_text(operations_raw.get(name))
        else:
            values = operations_raw.get(name) if isinstance(operations_raw.get(name), list) else []
            transform = cited if name in {"add_confirmed_facts", "add_rejected_hypotheses"} else replace_unseen_ids
            operations[name] = [transform(item) for item in values if _normalize_text(item)][:6]
    if action == "CONTINUE_CURRENT_LOOP":
        # CONTINUE's persistence policy is deterministic.  Preserve the model's
        # loop-local progress, but never spend a repair request on extra durable
        # fields that the controller must discard anyway.
        operations = {
            name: ([] if name != "completed_subgoal" else "")
            for name in DELTA_FIELDS
        }
        raw["durable_update"] = operations
        raw["loop_memory"] = None
    else:
        operations["completed_subgoal"] = boundary["current_subgoal"]
        operations["add_confirmed_facts"] = [
            item
            for item in operations["add_confirmed_facts"]
            if not DURABLE_OPERATION_PATTERN.search(item)
        ]
        memory_raw = raw.get("loop_memory") if isinstance(raw.get("loop_memory"), dict) else {}
        evidence_ids = [
            str(item)
            for item in memory_raw.get("evidence_ids") or []
            if str(item) in seen_message_ids
        ] or [latest_id]
        findings_raw = memory_raw.get("durable_findings")
        findings = [
            cited(item)
            for item in (findings_raw if isinstance(findings_raw, list) else [])
            if _normalize_text(item)
        ][:6]
        if not findings:
            findings = list(progress["key_evidence"][:4]) or [
                f"The closing Loop incorporated evidence from {latest_id}."
            ]
        memory = {
            "memory_id": f"memory_loop_{loop_number:03d}",
            "summary": _normalize_text(memory_raw.get("summary"))
            or progress["progress_summary"],
            "durable_findings": findings,
            "rejected_leads": [
                cited(item)
                for item in (
                    memory_raw.get("rejected_leads")
                    if isinstance(memory_raw.get("rejected_leads"), list)
                    else []
                )
                if _normalize_text(item)
            ][:6],
            "unresolved_questions": [
                replace_unseen_ids(item)
                for item in (
                    memory_raw.get("unresolved_questions")
                    if isinstance(memory_raw.get("unresolved_questions"), list)
                    else []
                )
                if _normalize_text(item)
            ][:6],
            "evidence_ids": evidence_ids[:8],
        }
        raw["loop_memory"] = memory
        raw["durable_update"] = operations

    boundary = deepcopy(boundary)
    boundary["reason"] = reason
    boundary["progress"] = progress
    retrieval_raw = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
    selected_ids = [
        str(item)
        for item in retrieval_raw.get("relevant_memory_ids") or []
        if str(item) in allowed_memory_ids
    ][:4]
    semantic_raw = {
        "durable_update": raw["durable_update"],
        "loop_memory": raw["loop_memory"],
        "retrieval": {
            "query": _normalize_text(retrieval_raw.get("query")) or boundary["current_subgoal"],
            "relevant_memory_ids": selected_ids,
            "reason": _normalize_text(retrieval_raw.get("reason"))
            or "No prior cross-loop memory was required.",
        },
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
    if action != "CONTINUE_CURRENT_LOOP":
        target["state_delta"]["summary"] = summarize_delta(
            target["state_delta"]["operations"]
        )
    return target


def summarize_delta(operations: dict[str, Any]) -> str:
    parts: list[str] = []
    completed = _normalize_text(operations.get("completed_subgoal"))
    if completed:
        parts.append(f"Completed subgoal: {completed}.")
    confirmed = operations.get("add_confirmed_facts") or []
    if confirmed:
        parts.append("Confirmed: " + " ".join(str(item) for item in confirmed[:2]))
    rejected = operations.get("add_rejected_hypotheses") or []
    if rejected:
        parts.append("Rejected: " + " ".join(str(item) for item in rejected[:2]))
    open_questions = operations.get("add_open_questions") or []
    if open_questions:
        parts.append("Open: " + " ".join(str(item) for item in open_questions[:2]))
    return _normalize_text(" ".join(parts))


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
        max_attempts=1,
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
    parser.add_argument("--target-trajectories", type=int, default=0)
    parser.add_argument("--min-loops-per-trajectory", type=int, default=1)
    parser.add_argument("--min-continues-per-trajectory", type=int, default=0)
    parser.add_argument("--unique-qids", action="store_true")
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
    completed_trajectory_qids: list[str] = []
    attempted_qids: set[str] = set()
    skipped_by_loop_count = 0
    skipped_by_continue_count = 0
    target_mode = args.target_trajectories > 0

    def target_reached() -> bool:
        return (
            len(completed_trajectory_qids) >= args.target_trajectories
            if target_mode
            else len(records) >= args.num_samples
        )

    def checkpoint(extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "generated_samples": len(records),
            "segmented_trajectories": len(segment_records),
            "completed_trajectory_qids": completed_trajectory_qids,
            **_usage(segment_client, state_client),
        }
        if extra:
            payload.update(extra)
        partial_usage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[retrospective] model={args.model} target_samples={args.num_samples} "
        f"target_trajectories={args.target_trajectories} "
        f"trajectory_decisions={args.min_decision_points}..{args.max_decision_points}",
        flush=True,
    )
    with args.output.open("w", encoding="utf-8") as output, segment_path.open("w", encoding="utf-8") as segments_out:
        for row in iter_rows(paths, seed=args.seed):
            if target_reached():
                break
            if str(row.get("status") or "").lower() not in {"success", "completed", "ok"}:
                continue
            messages = row.get("messages") or []
            qid = str(row.get("qid"))
            if args.unique_qids and qid in attempted_qids:
                continue
            steps = decision_steps(messages)
            if not (args.min_decision_points <= len(steps) <= args.max_decision_points):
                skipped_length += 1
                continue
            full_view, decision_count = trajectory_view(messages)
            lookahead_view = decision_lookahead_view(messages)
            has_final_answer = any(
                message.get("role") == "assistant"
                and message.get("channel") == "final"
                and bool(str(message.get("content") or "").strip())
                for message in messages
            )
            trajectory_chars = len(json.dumps(full_view, ensure_ascii=False))
            if trajectory_chars > args.max_trajectory_chars:
                skipped_chars += 1
                continue
            if args.unique_qids:
                attempted_qids.add(qid)
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
                    decision_lookahead=lookahead_view,
                    has_final_answer=has_final_answer,
                )
            except Exception as exc:
                errors.append({"stage": "segmentation", "qid": row.get("qid"), "error": str(exc)})
                checkpoint()
                print(f"[retrospective] skip segmentation qid={row.get('qid')}: {exc}", flush=True)
                continue

            usable_end = int(segmentation["loops"][-1]["end_decision_index"])
            trailing = decision_count - usable_end - 1
            loop_count = len(segmentation["loops"])
            switch_count = max(0, loop_count - 1)
            ready_count = int(segmentation["loops"][-1]["end_action"] == "READY_TO_ANSWER")
            continue_count = decision_count - switch_count - ready_count
            if loop_count < args.min_loops_per_trajectory:
                skipped_by_loop_count += 1
                checkpoint()
                print(
                    f"[retrospective] skip qid={row.get('qid')} loops={loop_count} "
                    f"required={args.min_loops_per_trajectory}",
                    flush=True,
                )
                continue
            if continue_count < args.min_continues_per_trajectory:
                skipped_by_continue_count += 1
                checkpoint()
                print(
                    f"[retrospective] skip qid={row.get('qid')} continues={continue_count} "
                    f"required={args.min_continues_per_trajectory}",
                    flush=True,
                )
                continue
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
            if not target_mode:
                segment_records.append(segment_record)
                segments_out.write(json.dumps(segment_record, ensure_ascii=False) + "\n")
                segments_out.flush()
                excluded_trailing += trailing
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
            trajectory_records: list[dict[str, Any]] = []
            trajectory_failed = False
            trajectory_reached_ready = False
            for decision_index, step in enumerate(steps[: usable_end + 1]):
                if not target_mode and len(records) >= args.num_samples:
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
                    trajectory_failed = True
                    break
                record = {
                    "sample_id": (
                        f"ws_retro_pending_{len(trajectory_records) + 1:04d}"
                        if target_mode
                        else f"ws_retro_{len(records) + 1:04d}"
                    ),
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
                    if not target_mode:
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
                    trajectory_failed = True
                    break

                action = target["loop_decision"]["action"]
                if target_mode:
                    trajectory_records.append(record)
                else:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    records.append(record)
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
                    f"[retrospective] {'buffered' if target_mode else 'saved'}="
                    f"{len(trajectory_records) if target_mode else len(records)} "
                    f"qid={row.get('qid')} "
                    f"step={decision_index} loop={loop_number} action={action}",
                    flush=True,
                )
                if action == "READY_TO_ANSWER":
                    trajectory_reached_ready = True
                    break

            if target_mode:
                trajectory_complete = (
                    not trajectory_failed
                    and trajectory_reached_ready
                    and len(trajectory_records) == usable_end + 1
                )
                if not trajectory_complete:
                    print(
                        f"[retrospective] discard incomplete replay qid={row.get('qid')} "
                        f"buffered={len(trajectory_records)}/{usable_end + 1}",
                        flush=True,
                    )
                    continue
                for record in trajectory_records:
                    record["sample_id"] = f"ws_retro_{len(records) + 1:04d}"
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    action_counts[record["target"]["loop_decision"]["action"]] += 1
                output.flush()
                segment_records.append(segment_record)
                segments_out.write(json.dumps(segment_record, ensure_ascii=False) + "\n")
                segments_out.flush()
                excluded_trailing += trailing
                completed_trajectory_qids.append(qid)
                checkpoint()
                print(
                    f"[retrospective] committed trajectory {len(completed_trajectory_qids)}/"
                    f"{args.target_trajectories} qid={row.get('qid')} samples={len(trajectory_records)}",
                    flush=True,
                )

    preview = args.output.with_suffix(".preview.md")
    write_preview(preview, records, count=args.preview_count)
    loop_counts = [len(item["segmentation"]["loops"]) for item in segment_records]
    report = {
        "requested_samples": args.num_samples,
        "requested_trajectories": args.target_trajectories,
        "completed_trajectory_qids": completed_trajectory_qids,
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
        "skipped_by_loop_count": skipped_by_loop_count,
        "skipped_by_continue_count": skipped_by_continue_count,
        "model": args.model,
        **_usage(segment_client, state_client),
        "output": str(args.output),
        "segments": str(segment_path),
        "preview": str(preview),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    complete = target_reached()
    checkpoint({"complete": complete})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not complete or causal_violations:
        expected = (
            f"{args.target_trajectories} trajectories"
            if target_mode
            else f"{args.num_samples} samples"
        )
        raise SystemExit(f"generated {len(records)} samples but did not complete {expected}")


if __name__ == "__main__":
    main()
