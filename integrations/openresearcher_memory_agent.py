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
        self.client = OpenAIChatJSONClient(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            model=model,
            base_url=self.base_url,
            timeout_seconds=float(os.environ.get("CL_GISM_CONTROLLER_TIMEOUT_SECONDS", "300")),
            max_tokens=int(os.environ.get("CL_GISM_CONTROLLER_MAX_TOKENS", "2048")),
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


_baseline_run_one = openresearcher.run_one


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
    messages = await _baseline_run_one(
        question=question,
        qid=qid,
        generator=generator,
        browser_pool=browser_pool,
        max_rounds=max_rounds,
    )
    router.finalize(question, messages)
    return messages


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
