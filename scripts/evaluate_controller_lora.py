#!/usr/bin/env python3
"""Greedy held-out evaluation for a Controller LoRA adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cl_gism.controller_sft import parse_json_object  # noqa: E402


EXPECTED_KEYS = {
    "working_state_after", "loop_decision", "state_delta", "cross_loop_memory", "retrieval"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def memory_ids(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    retrieval = value.get("retrieval")
    if not isinstance(retrieval, dict):
        return set()
    return {str(item) for item in retrieval.get("relevant_memory_ids") or []}


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.validation.open(encoding="utf-8")
        if line.strip()
    ][: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    counters = Counter()
    predictions: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predicted = parse_json_object(generated)
        gold = json.loads(messages[-1]["content"])
        counters["samples"] += 1
        if predicted is not None:
            counters["valid_json"] += 1
            if set(predicted) == EXPECTED_KEYS:
                counters["valid_top_level_schema"] += 1
            predicted_action = str((predicted.get("loop_decision") or {}).get("action") or "")
            gold_action = str(gold["loop_decision"]["action"])
            if predicted_action == gold_action:
                counters["action_correct"] += 1
            if memory_ids(predicted) == memory_ids(gold):
                counters["retrieval_set_exact"] += 1
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "gold_action": gold["loop_decision"]["action"],
                "prediction_text": generated,
                "prediction": predicted,
            }
        )

    total = max(1, counters["samples"])
    metrics = {
        **dict(counters),
        "json_valid_rate": counters["valid_json"] / total,
        "top_level_schema_rate": counters["valid_top_level_schema"] / total,
        "action_accuracy": counters["action_correct"] / total,
        "retrieval_set_exact_rate": counters["retrieval_set_exact"] / total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
