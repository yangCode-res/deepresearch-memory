"""Optional LLM-backed memory ranking for CL-GISM.

The MVP uses lexical retrieval as a safe fallback, but this module lets an
OpenAI-compatible model decide which candidate memories should enter the next
context window.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .retrieval import MemoryHit, build_retrieval_query
from .schema import GlobalIntentState, TaskAnchor


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-terra"


def _truncate(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _compact_items(items: list[Any], limit: int = 5) -> list[str]:
    compacted: list[str] = []
    for item in items[:limit]:
        value = getattr(item, "value", item)
        text = str(value).strip()
        if text:
            compacted.append(text)
    return compacted


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = _strip_code_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.S)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a valid JSON object")


@dataclass
class RetrievalPlan:
    retrieval_query: str
    selected_memory_ids: list[str]
    reason: str = ""
    source: str = "lexical"
    model: str | None = None


class OpenAIChatJSONClient:
    """Very small OpenAI-compatible JSON caller built on stdlib HTTP."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls, *, model: str | None = None, base_url: str | None = None) -> "OpenAIChatJSONClient" | None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CL_GISM_OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=model or os.getenv("CL_GISM_OPENAI_MODEL") or DEFAULT_MODEL,
            base_url=base_url or os.getenv("CL_GISM_OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            timeout_seconds=float(os.getenv("CL_GISM_OPENAI_TIMEOUT_SECONDS", "60")),
            max_tokens=int(os.getenv("CL_GISM_OPENAI_MAX_TOKENS", "2048")),
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:  # pragma: no cover - network path
                detail = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI request failed with status {exc.code}: {detail}") from exc
            except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, ConnectionError) as exc:
                if attempt == 2:  # pragma: no cover - network path
                    detail = getattr(exc, "reason", str(exc))
                    raise RuntimeError(f"OpenAI request failed after 3 attempts: {detail}") from exc
                time.sleep(0.5 * (2**attempt))

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response did not contain any choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI response did not contain assistant text content")
        return _extract_json_object(content)


class LLMMemoryReranker:
    """Use a model to rerank lexical candidates for the next context window."""

    def __init__(
        self,
        client: OpenAIChatJSONClient | None,
        *,
        candidate_pool_size: int = 12,
        top_k: int = 5,
    ) -> None:
        self.client = client
        self.candidate_pool_size = candidate_pool_size
        self.top_k = top_k

    def _client_model(self) -> str | None:
        return getattr(self.client, "model", None) if self.client else None

    @classmethod
    def from_env(
        cls,
        *,
        candidate_pool_size: int = 12,
        top_k: int = 5,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "LLMMemoryReranker" | None:
        client = OpenAIChatJSONClient.from_env(model=model, base_url=base_url)
        if not client:
            return None
        return cls(client, candidate_pool_size=candidate_pool_size, top_k=top_k)

    def _format_state(self, state: GlobalIntentState) -> str:
        parts = [
            f"current_goal: {state.current_goal.value if state.current_goal else ''}",
            f"open_questions: {_compact_items(state.open_questions)}",
            f"working_hypotheses: {_compact_items(state.working_hypotheses)}",
            f"resolved_findings: {_compact_items(state.resolved_findings)}",
            f"next_actions: {_compact_items(state.next_actions)}",
        ]
        return "\n".join(parts)

    def _format_candidates(self, hits: list[MemoryHit]) -> str:
        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            lines.append(
                "\n".join(
                    [
                        f"{index}. id={hit.memory_id} type={hit.memory_type} score={hit.score:.4f}",
                        f"   text={_truncate(hit.text, 260)}",
                    ]
                )
            )
        return "\n".join(lines)

    def rerank(
        self,
        anchor: TaskAnchor,
        state: GlobalIntentState,
        current_event: str,
        candidates: list[MemoryHit],
        *,
        recent_messages: list[str] | None = None,
    ) -> tuple[RetrievalPlan, list[MemoryHit]]:
        fallback_query = build_retrieval_query(anchor, state, current_event)
        lexical_hits = candidates[: self.top_k]
        if not self.client or not candidates:
            return (
                RetrievalPlan(
                    retrieval_query=fallback_query,
                    selected_memory_ids=[hit.memory_id for hit in lexical_hits],
                    reason="lexical fallback",
                    source="lexical",
                    model=self._client_model(),
                ),
                lexical_hits,
            )

        system_prompt = (
            "You rank candidate memories for a deep-research agent.\n"
            "Return only valid JSON.\n"
            "Select the memories that are most useful for the next model call."
        )
        user_prompt = "\n".join(
            [
                f"task_id: {anchor.task_id}",
                f"goal: {anchor.original_goal}",
                f"current_event: {current_event}",
                "state:",
                self._format_state(state),
                f"recent_messages: {recent_messages or []}",
                "candidate_memories:",
                self._format_candidates(candidates),
                "",
                "Return JSON with exactly these keys:",
                "{",
                '  "retrieval_query": "short query for a follow-up lexical search",',
                '  "selected_memory_ids": ["id1", "id2"],',
                '  "reason": "one short sentence"',
                "}",
                f"Choose at most {self.top_k} ids and keep them ordered by usefulness.",
                "If none are useful, return an empty selected_memory_ids list.",
            ]
        )
        try:
            result = self.client.complete_json(system_prompt, user_prompt)
        except Exception as exc:  # pragma: no cover - network path
            return (
                RetrievalPlan(
                    retrieval_query=fallback_query,
                    selected_memory_ids=[hit.memory_id for hit in lexical_hits],
                    reason=f"lexical fallback after llm error: {exc.__class__.__name__}",
                    source="lexical",
                    model=self._client_model(),
                ),
                lexical_hits,
            )

        allowed_ids = [hit.memory_id for hit in candidates]
        requested_ids = result.get("selected_memory_ids") or []
        if isinstance(requested_ids, list):
            selected_ids = [str(memory_id) for memory_id in requested_ids if str(memory_id) in allowed_ids]
        else:
            selected_ids = []

        if not selected_ids:
            return (
                RetrievalPlan(
                    retrieval_query=str(result.get("retrieval_query") or fallback_query).strip() or fallback_query,
                    selected_memory_ids=[hit.memory_id for hit in lexical_hits],
                    reason=str(result.get("reason") or "lexical fallback after empty llm selection"),
                    source="lexical",
                    model=self._client_model(),
                ),
                lexical_hits,
            )

        selected_hits = [hit for memory_id in selected_ids for hit in candidates if hit.memory_id == memory_id]
        if len(selected_hits) > self.top_k:
            selected_hits = selected_hits[: self.top_k]
            selected_ids = [hit.memory_id for hit in selected_hits]

        return (
            RetrievalPlan(
                retrieval_query=str(result.get("retrieval_query") or fallback_query).strip() or fallback_query,
                selected_memory_ids=selected_ids,
                reason=str(result.get("reason") or ""),
                source="llm",
                model=self._client_model(),
            ),
            selected_hits,
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMMemoryReranker",
    "OpenAIChatJSONClient",
    "RetrievalPlan",
]
