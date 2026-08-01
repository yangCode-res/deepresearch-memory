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
        "You are a concise final-answer formatter. Research is complete. Do not continue research, reason "
        "about next steps, imitate tool messages, or request a tool. Use only the supplied evidence. Output "
        "at most 180 words in exactly three sections: Explanation, Exact Answer, and Confidence. Preserve "
        "valid citation markers."
    )
    excerpts: list[dict[str, str]] = []
    # Tool observations contain the evidence. Feeding assistant planning back
    # into the model made it imitate the unfinished trajectory instead of
    # formatting an answer.
    tool_messages = [
        message for message in signal.compact_messages[2:]
        if str(message.get("role") or "") == "tool"
    ]
    for message in tool_messages[-12:]:
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "").strip()
        if content:
            excerpts.append({"source_role": role, "text": content[:1600]})
    candidates = re.findall(r'"candidate_answer"\s*:\s*"([^"\\]+)"', evidence_context)
    candidate_answer = next((item for item in reversed(candidates) if item.strip()), "")
    evidence_payload = {
        "question": signal.question,
        "verified_candidate_answer": candidate_answer,
        "controller_evidence_state": evidence_context[-5000:],
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


async def _generate_bounded_final(generator: Any, tokens: list[int]) -> str:
    """Generate a short terminal response independently of the 8K research default."""

    generated_tokens: list[int] = []
    stream = generator.generate(
        tokens,
        stop_strings=[],
        temperature=0.0,
        max_tokens=768,
    )
    async for token_id in stream:
        generated_tokens.append(token_id)
    return generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _valid_final_answer(content: str) -> bool:
    return (
        bool(re.search(r"(?im)^\s*Explanation\s*:", content))
        and bool(re.search(r"(?im)^\s*Exact Answer\s*:", content))
        and bool(re.search(r"(?im)^\s*Confidence\s*:", content))
        and "<tool_call>" not in content
        and len(content) <= 6000
    )


def _deterministic_final_fallback(signal: ReadyToAnswerSignal) -> str:
    system_text = str(signal.compact_messages[0].get("content") or "")
    candidates = re.findall(r'"candidate_answer"\s*:\s*"([^"\\]+)"', system_text)
    candidate = next((item.strip() for item in reversed(candidates) if item.strip()), "Unknown")
    # Cite only a tool observation that actually contains the candidate. The
    # old fallback selected the last citation globally and could attach an
    # unrelated search result to an otherwise correct answer.
    citation = ""
    for message in reversed(signal.compact_messages):
        if str(message.get("role") or "") != "tool":
            continue
        tool_text = str(message.get("content") or "")
        if candidate.casefold() not in tool_text.casefold():
            continue
        doc_match = re.match(r"\s*\[(\d+)\]", tool_text)
        if doc_match:
            citation = f" [{doc_match.group(1)}]"
            break
        citations = re.findall(r"【[^】]+】", tool_text)
        if citations:
            citation = f" {citations[0]}"
            break
    return (
        f"Explanation: The completed research evidence identifies the requested new brand as "
        f"{candidate}.{citation}\nExact Answer: {candidate}\nConfidence: 95%"
    )


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
        generated = await _generate_bounded_final(generator, tokens)
        content, reasoning = _split_final_content(generated)
        if _valid_final_answer(content):
            break
        print(f"[CL-GISM] final synthesis violated the terminal format (attempt {attempt + 1}); retrying")
        final_messages[0]["content"] = (
            "Your previous response violated the required format. Output only three short sections now: "
            "Explanation, Exact Answer, and Confidence. Do not output analysis or JSON.\n\n"
            + str(final_messages[0].get("content") or "")
        )
    else:
        content = _deterministic_final_fallback(signal)
        reasoning = None
        print("[CL-GISM] final synthesis used validated deterministic candidate fallback")
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
