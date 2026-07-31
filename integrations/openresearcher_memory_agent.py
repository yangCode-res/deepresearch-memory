#!/usr/bin/env python3
"""Run OpenResearcher with CL-GISM online memory without editing the vendor repo.

The wrapper reuses OpenResearcher's worker, browser, parsing, and output code.
Only the tokenizer input path is wrapped: the canonical trajectory stays intact
for evaluation, while each generation receives compact state, the current loop,
and retrieved completed-loop memories.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = Path(os.environ.get("OPENRESEARCHER_ROOT", ROOT / "vendor" / "openresearcher")).resolve()
for path in (str(ROOT / "src"), str(VENDOR_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import deploy_agent as openresearcher  # noqa: E402
from cl_gism import (  # noqa: E402
    LLMLoopBoundaryJudge,
    LLMStateUpdater,
    OnlineMemorySession,
    OpenAIChatJSONClient,
    UnifiedMemoryController,
)


class MemoryTokenizerRouter:
    """Dispatch tokenizer calls to a per-question online memory session."""

    def __init__(self, tokenizer: Any, *, base_url: str, model: str) -> None:
        self._tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sessions: dict[str, OnlineMemorySession] = {}
        self.qids: dict[str, Any] = {}
        self.top_k = int(os.environ.get("CL_GISM_MEMORY_TOP_K", "4"))
        self.memory_text_limit = int(os.environ.get("CL_GISM_MEMORY_TEXT_LIMIT", "1800"))
        self.trace_dir = Path(os.environ.get("CL_GISM_TRACE_DIR", "memory_traces"))
        controller_key = os.environ.get("CL_GISM_CONTROLLER_API_KEY")
        controller_url = os.environ.get("CL_GISM_CONTROLLER_BASE_URL")
        controller_model = os.environ.get("CL_GISM_CONTROLLER_MODEL", "mimo-v2.5-pro")
        if not controller_key or not controller_url:
            raise RuntimeError(
                "CL_GISM_CONTROLLER_API_KEY and CL_GISM_CONTROLLER_BASE_URL are required"
            )
        self.client = OpenAIChatJSONClient(
            api_key=controller_key,
            model=controller_model,
            base_url=controller_url,
            timeout_seconds=float(os.environ.get("CL_GISM_CONTROLLER_TIMEOUT_SECONDS", "300")),
            max_tokens=int(os.environ.get("CL_GISM_CONTROLLER_MAX_TOKENS", "3072")),
        )
        self.unified_controller = UnifiedMemoryController(
            self.client, max_selected_memories=self.top_k
        )
        print(
            f"[CL-GISM] controller model={controller_model} base_url={controller_url} "
            f"top_k={self.top_k}"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def register(self, *, question: str, qid: Any, system_prompt: str) -> None:
        self.qids[question] = qid
        if question not in self.sessions:
            self.sessions[question] = OnlineMemorySession(
                qid=qid,
                question=question,
                system_prompt=system_prompt,
                boundary_judge=LLMLoopBoundaryJudge(self.client),
                state_updater=LLMStateUpdater(self.client),
                unified_controller=self.unified_controller,
                top_k=self.top_k,
                memory_text_limit=self.memory_text_limit,
            )

    def apply_chat_template(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        if len(messages) < 2:
            return self._tokenizer.apply_chat_template(messages, *args, **kwargs)
        question = str(messages[1].get("content") or "")
        session = self.sessions.get(question)
        if session is None:
            return self._tokenizer.apply_chat_template(messages, *args, **kwargs)
        compact_messages = session.build_prompt(messages)
        latest = session.traces[-1]
        print(
            "[CL-GISM] "
            f"qid={session.qid} round={latest.round_number} "
            f"canonical={latest.canonical_message_count} prompt={latest.prompt_message_count} "
            f"loop_messages={latest.current_loop_message_count} state_v={latest.state_version} "
            f"retrieved={len(latest.retrieved_memory_ids)} switched={latest.loop_switched}"
        )
        if latest.task_status == "READY_TO_ANSWER":
            raise ReadyToAnswerSignal(question, messages, compact_messages)
        return self._tokenizer.apply_chat_template(compact_messages, *args, **kwargs)

    def finalize(self, question: str, messages: list[dict[str, Any]]) -> None:
        session = self.sessions.get(question)
        if session is None:
            return
        session.finalize(messages)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / f"qid_{session.qid}.json"
        temporary = trace_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(session.trace_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(trace_path)
        print(
            f"[CL-GISM] finalized qid={session.qid} loops={len(session.completed_loops)} "
            f"state_v={session.state.state_version} trace={trace_path}"
        )
        del self.sessions[question]

    def abort(self, question: str) -> None:
        """Discard partial in-memory state before the vendor retries a task."""

        self.sessions.pop(question, None)


_baseline_run_one = openresearcher.run_one


class ReadyToAnswerSignal(Exception):
    """Unwind the vendor research loop before it can execute another tool."""

    def __init__(
        self,
        question: str,
        canonical_messages: list[dict[str, Any]],
        compact_messages: list[dict[str, Any]],
    ) -> None:
        super().__init__("CL-GISM controller marked the task READY_TO_ANSWER")
        self.question = question
        self.canonical_messages = [dict(message) for message in canonical_messages]
        self.compact_messages = [dict(message) for message in compact_messages]


def _final_answer_messages(signal: ReadyToAnswerSignal) -> list[dict[str, Any]]:
    prior_system = str(signal.compact_messages[0].get("content") or "")
    memory_marker = prior_system.find("<cross_loop_memory>")
    evidence_context = prior_system[memory_marker:] if memory_marker >= 0 else ""
    system_content = (
        "You are the final answer synthesizer. Research is complete. Do not call, mention, or request "
        "any tool. Use only the evidence and citations already contained in these messages. Return the "
        "answer now in exactly three sections: Explanation, Exact Answer, and Confidence. Preserve valid "
        "citation markers. If a minor corroborating detail is uncertain, disclose it briefly instead of "
        "searching again. There are no tools in this stage and the input below is evidence text, not a "
        "conversation to continue."
    )
    excerpts: list[dict[str, str]] = []
    for message in signal.compact_messages[2:][-80:]:
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or "").strip()
        text = content or reasoning
        if text:
            excerpts.append({"source_role": role, "text": text[:3000]})
    evidence_payload = {
        "question": signal.question,
        "cross_loop_memory": evidence_context,
        "research_evidence_excerpts": excerpts,
        "required_output": ["Explanation", "Exact Answer", "Confidence"],
    }
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(evidence_payload, ensure_ascii=False)},
    ]


def _split_final_content(text: str) -> tuple[str, str | None]:
    reasoning = None
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        text = text.replace(match.group(0), "").strip()
    elif "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.strip()
        text = text.strip()
    return text, reasoning


async def _force_final_answer(
    signal: ReadyToAnswerSignal,
    generator: Any,
) -> list[dict[str, Any]]:
    router: MemoryTokenizerRouter = generator.tokenizer
    final_messages = _final_answer_messages(signal)
    print("[CL-GISM] READY_TO_ANSWER: research loop stopped; forcing one tool-free final generation")
    content = ""
    reasoning = None
    for attempt in range(2):
        prompt = router._tokenizer.apply_chat_template(
            final_messages,
            tools=[],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = router._tokenizer.encode(prompt, add_special_tokens=False)
        generated = await openresearcher._generate_with_retry(
            generator, tokens, ["\n<tool_response>", "<tool_response>"]
        )
        content, reasoning = _split_final_content(generated)
        if "<tool_call>" not in content and "</tool_call>" not in content:
            break
        print(f"[CL-GISM] final synthesis attempted a tool call (attempt {attempt + 1}); retrying")
        final_messages[0]["content"] = (
            "Your previous attempt incorrectly requested a tool. No tools exist in final synthesis. "
            "Output Explanation, Exact Answer, and Confidence immediately.\n\n"
            + str(final_messages[0].get("content") or "")
        )
    else:
        raise RuntimeError("final synthesis attempted tool calls twice despite tool-free enforcement")
    messages = list(signal.canonical_messages)
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": None,
        }
    )
    return messages


async def run_one_with_memory(
    question: str,
    qid: Any,
    generator: Any,
    browser_pool: Any,
    max_rounds: int = 200,
) -> list[dict[str, Any]]:
    await generator._init_tokenizer()
    if not isinstance(generator.tokenizer, MemoryTokenizerRouter):
        generator.tokenizer = MemoryTokenizerRouter(
            generator.tokenizer,
            base_url=generator.base_url,
            model=generator.model_name,
        )
    router: MemoryTokenizerRouter = generator.tokenizer
    system_prompt = openresearcher.DEVELOPER_CONTENT
    router.register(question=question, qid=qid, system_prompt=system_prompt)
    effective_max_rounds = min(
        max_rounds,
        int(os.environ.get("CL_GISM_MAX_ROUNDS", "60")),
    )
    try:
        try:
            messages = await _baseline_run_one(
                question=question,
                qid=qid,
                generator=generator,
                browser_pool=browser_pool,
                max_rounds=effective_max_rounds,
            )
        except ReadyToAnswerSignal as signal:
            messages = await _force_final_answer(signal, generator)
        # The vendor loop can stop immediately after a tool result at its turn
        # limit. That is not a successful model answer; force one evidence-only
        # synthesis so evaluation receives an actual prediction.
        if messages and str(messages[-1].get("role") or "") != "assistant":
            compact_messages = router.sessions[question].build_prompt(messages)
            messages = await _force_final_answer(
                ReadyToAnswerSignal(question, messages, compact_messages), generator
            )
        router.finalize(question, messages)
        return messages
    except Exception:
        router.abort(question)
        raise


def main() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The isolated OpenResearcher wrapper currently requires Linux fork semantics")
    openresearcher.run_one = run_one_with_memory
    # The monkey-patched run_one must be inherited by OpenResearcher's workers.
    # Workers only use external vLLM HTTP servers, so no CUDA context is forked.
    mp.set_start_method("fork", force=True)
    openresearcher.main()


if __name__ == "__main__":
    main()
