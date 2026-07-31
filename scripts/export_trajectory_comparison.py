#!/usr/bin/env python3
"""Export baseline and CL-GISM trajectories as human-readable Markdown."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import re
from typing import Any


def load_record(directory: Path, qid: int) -> dict[str, Any]:
    for filename in sorted(directory.glob("*.jsonl")):
        for line in filename.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row["qid"]) == qid:
                return row
    raise FileNotFoundError(f"qid={qid} not found below {directory}")


def text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def exact_answer(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    final = text(messages[-1].get("content", "")) if messages else ""
    match = re.search(r"Exact Answer:\s*(.*?)(?:\n|$)", final, re.I)
    return match.group(1).strip(" *") if match else final[-300:]


def normalize_answer(value: Any) -> str:
    """Normalize presentation-only differences before comparing short answers."""
    normalized = str(value or "").replace("\u202f", " ").replace("\u00a0", " ")
    normalized = re.sub(r"[*_`]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def assistant_rounds(messages: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], list[dict[str, Any]]]]:
    rounds: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    current: tuple[int, dict[str, Any], list[dict[str, Any]]] | None = None
    number = 0
    for message in messages:
        if message.get("role") == "assistant":
            number += 1
            current = (number, message, [])
            rounds.append(current)
        elif message.get("role") == "tool" and current is not None:
            current[2].append(message)
    return rounds


def tool_signature(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    function = calls[0].get("function") or {}
    name = str(function.get("name") or "")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass
    return f"{name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def render_message(message: dict[str, Any]) -> str:
    payload = {key: value for key, value in message.items() if value not in (None, "", [], {})}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_trajectory(
    title: str,
    record: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> str:
    messages = record.get("messages") or []
    rounds = assistant_rounds(messages)
    trace_rounds = {int(item["round_number"]): item for item in (trace or {}).get("rounds", [])}
    lines = [
        f"# {title}",
        "",
        "## Task summary / 任务摘要",
        "",
        f"- QID: `{record.get('qid')}`",
        f"- Gold answer / 标准答案: `{record.get('answer')}`",
        f"- Predicted answer / 模型答案: `{exact_answer(record)}`",
        f"- Status / 状态: `{record.get('status')}`",
        f"- Latency / 推理耗时: `{record.get('latency_s')}` seconds",
        f"- Assistant rounds / 研究轮数: `{len(rounds)}`",
        f"- Canonical messages / 完整消息数: `{len(messages)}`",
        "",
        "## Original question / 原始问题",
        "",
        record.get("question", ""),
        "",
    ]
    if trace:
        lines.extend(
            [
                "## CL-GISM summary / Memory 摘要",
                "",
                f"- Completed loops: `{len(trace.get('completed_loops', []))}`",
                f"- State version: `{trace.get('state', {}).get('state_version')}`",
                f"- StateDelta count: `{len(trace.get('state_deltas', []))}`",
                "",
            ]
        )
    initial_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        initial_messages.append(message)
    lines.extend(["## Initial messages / 初始输入消息", ""])
    if initial_messages:
        for index, message in enumerate(initial_messages, start=1):
            lines.extend(
                [
                    f"### Initial message {index}: {message.get('role', 'unknown')}",
                    "",
                    "```json",
                    render_message(message),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "The result record does not retain separate system/user message objects; "
                "the original user question is preserved above.",
                "",
            ]
        )
    lines.extend(["## Complete round-by-round trajectory / 完整逐轮轨迹", ""])
    for number, assistant, tools in rounds:
        lines.extend([f"### Round {number} / 第 {number} 轮", ""])
        if number in trace_rounds:
            item = trace_rounds[number]
            lines.extend(
                [
                    "#### Memory control metadata / Memory 控制信息",
                    "",
                    "```json",
                    json.dumps(item, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        lines.extend(["#### Assistant / 研究 Agent", "", "```json", render_message(assistant), "```", ""])
        for index, tool in enumerate(tools, start=1):
            lines.extend(
                [
                    f"#### Tool result {index} / 工具结果 {index}",
                    "",
                    "```json",
                    render_message(tool),
                    "```",
                    "",
                ]
            )
    if trace:
        lines.extend(
            [
                "## Completed Loop memories / 已封存 Loop",
                "",
                "```json",
                json.dumps(trace.get("completed_loops", []), ensure_ascii=False, indent=2),
                "```",
                "",
                "## StateDelta history / 状态更新历史",
                "",
                "```json",
                json.dumps(trace.get("state_deltas", []), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Final Global Intent State / 最终全局状态",
                "",
                "```json",
                json.dumps(trace.get("state", {}), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def round_index(record: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for number, assistant, tools in assistant_rounds(record.get("messages") or []):
        reasoning = str(assistant.get("reasoning_content") or assistant.get("content") or "")
        output.append(
            {
                "round": number,
                "intent_preview": reasoning.replace("\n", " ")[:180],
                "tool_call": tool_signature(assistant),
                "tool_results": len(tools),
            }
        )
    return output


def render_comparison(base: dict[str, Any], memory: dict[str, Any], trace: dict[str, Any]) -> str:
    base_index = round_index(base)
    memory_index = round_index(memory)
    base_tools = Counter(item["tool_call"] for item in base_index if item["tool_call"])
    memory_tools = Counter(item["tool_call"] for item in memory_index if item["tool_call"])
    repeated = [(call, count) for call, count in memory_tools.items() if count > 1]
    trace_rounds = trace.get("rounds", [])
    rows = []
    for item in memory_index:
        meta = trace_rounds[item["round"] - 1] if item["round"] <= len(trace_rounds) else {}
        rows.append(
            "| {round} | {status} | {switch} | {state} | {retrieved} | {tool} | {preview} |".format(
                round=item["round"],
                status=meta.get("task_status", ""),
                switch="Y" if meta.get("loop_switched") else "",
                state=meta.get("state_version", ""),
                retrieved=len(meta.get("retrieved_memory_ids", [])),
                tool=item["tool_call"].replace("|", "\\|")[:120],
                preview=item["intent_preview"].replace("|", "\\|")[:140],
            )
        )
    return "\n".join(
        [
            "# Baseline vs CL-GISM V2 trajectory index / 轨迹人工校准索引",
            "",
            "## Outcome comparison / 结果对比",
            "",
            "| Metric | Baseline | CL-GISM V2 |",
            "|---|---:|---:|",
            f"| Correct answer | {normalize_answer(exact_answer(base)) == normalize_answer(base.get('answer'))} | {normalize_answer(exact_answer(memory)) == normalize_answer(memory.get('answer'))} |",
            f"| Predicted answer | {exact_answer(base)} | {exact_answer(memory)} |",
            f"| Research rounds | {len(base_index)} | {len(memory_index)} |",
            f"| Latency seconds | {base.get('latency_s'):.2f} | {memory.get('latency_s'):.2f} |",
            f"| Tool calls | {sum(base_tools.values())} | {sum(memory_tools.values())} |",
            f"| Completed loops | — | {len(trace.get('completed_loops', []))} |",
            f"| StateDelta count | — | {len(trace.get('state_deltas', []))} |",
            "",
            "## Exact repeated CL-GISM tool calls / 完全重复的工具调用",
            "",
            *(f"- `{count}×` {call}" for call, count in sorted(repeated, key=lambda x: -x[1])),
            "" if repeated else "- None detected by exact signature / 未发现参数完全相同的重复调用",
            "",
            "## CL-GISM round index / CL-GISM 逐轮索引",
            "",
            "| Round | Task status | Loop switch | State v | Memories | Tool call | Intent preview |",
            "|---:|---|:---:|---:|---:|---|---|",
            *rows,
            "",
            "## Manual calibration checklist / 人工校准建议",
            "",
            "- 标记每个条件 A–E 的证据首次出现轮次。",
            "- 检查后续需要该证据时，对应 Memory ID 是否被召回。",
            "- 标记因为证据或引用丢失而发生的重复搜索。",
            "- 检查每次 Loop switch 是否真的开始了新子任务。",
            "- 检查每个 StateDelta 是否保留了结论、日期、来源和引用需求。",
            "- 对照两条完整轨迹，区分随机搜索差异与 Memory 导致的差异。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_record(args.baseline_dir, args.qid)
    memory = load_record(args.memory_dir, args.qid)
    trace = json.loads(args.trace_file.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"baseline_qid_{args.qid}_trajectory.md").write_text(
        render_trajectory("Baseline complete trajectory / Baseline 完整轨迹", baseline), encoding="utf-8"
    )
    (args.output_dir / f"cl_gism_v2_qid_{args.qid}_trajectory.md").write_text(
        render_trajectory("CL-GISM V2 complete trajectory / CL-GISM V2 完整轨迹", memory, trace),
        encoding="utf-8",
    )
    (args.output_dir / f"comparison_qid_{args.qid}.md").write_text(
        render_comparison(baseline, memory, trace), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
