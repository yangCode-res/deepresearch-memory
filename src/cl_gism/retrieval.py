"""Dependency-free lexical retrieval and context packing for the MVP."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from .schema import GlobalIntentState, LoopMemory, RawMemory, TaskAnchor


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


@dataclass
class MemoryDocument:
    memory_id: str
    memory_type: str
    text: str
    metadata: dict[str, Any]


@dataclass
class MemoryHit:
    memory_id: str
    memory_type: str
    score: float
    text: str
    metadata: dict[str, Any]


class LexicalMemoryIndex:
    """Small BM25-style index over RawMemory and LoopMemory records."""

    def __init__(self) -> None:
        self.documents: list[MemoryDocument] = []
        self._tokens: dict[str, list[str]] = {}

    def add_raw(self, memory: RawMemory) -> None:
        content = memory.content
        if isinstance(content, dict):
            text = content.get("text") or json.dumps(content, ensure_ascii=False)
        else:
            text = str(content)
        self._add(
            MemoryDocument(
                memory_id=memory.raw_id,
                memory_type="raw",
                text=text,
                metadata={"source_type": memory.source_type.value, "task_id": memory.task_id},
            )
        )

    def add_loop(self, memory: LoopMemory) -> None:
        # Put durable conclusions and tool observations before assistant actions.
        # Prompt packing truncates long memories, so evidence must appear first.
        text = "\n".join(
            [
                f"SUBGOAL: {memory.subgoal}",
                f"CONCLUSION: {memory.conclusion or ''}",
                f"TOOL_EVIDENCE: {json.dumps(memory.observations, ensure_ascii=False)}",
                f"CONTEXT: {memory.context or ''}",
                f"ACTIONS: {json.dumps(memory.actions, ensure_ascii=False)}",
            ]
        )
        self._add(
            MemoryDocument(
                memory_id=memory.loop_id,
                memory_type="loop",
                text=text,
                metadata={"task_id": memory.task_id, "status": memory.status.value},
            )
        )

    def _add(self, document: MemoryDocument) -> None:
        self.documents.append(document)
        self._tokens[document.memory_id] = tokenize(document.text)

    def lookup(self, memory_ids: list[str]) -> list[MemoryHit]:
        """Fetch specific memories in caller order for boundary handoff."""
        by_id = {document.memory_id: document for document in self.documents}
        hits: list[MemoryHit] = []
        for memory_id in memory_ids:
            document = by_id.get(memory_id)
            if document is not None:
                hits.append(
                    MemoryHit(
                        document.memory_id,
                        document.memory_type,
                        1.0,
                        document.text,
                        document.metadata,
                    )
                )
        return hits

    def search(self, query: str, top_k: int = 5, task_id: str | None = None) -> list[MemoryHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        n = len(self.documents)
        lengths = [len(self._tokens[d.memory_id]) for d in self.documents]
        avgdl = sum(lengths) / max(n, 1)
        df = {term: sum(term in self._tokens[d.memory_id] for d in self.documents) for term in set(query_tokens)}
        k1, b = 1.2, 0.75
        hits: list[MemoryHit] = []
        for document, dl in zip(self.documents, lengths):
            if task_id and document.metadata.get("task_id") != task_id:
                continue
            terms = self._tokens[document.memory_id]
            score = 0.0
            for term in query_tokens:
                tf = terms.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1))))
            if score > 0:
                source_type = document.metadata.get("source_type")
                if source_type == "tool":
                    score *= 1.3
                elif document.memory_type == "loop":
                    score *= 1.15
                elif source_type == "agent":
                    score *= 0.8
                hits.append(MemoryHit(document.memory_id, document.memory_type, score, document.text, document.metadata))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


def build_retrieval_query(anchor: TaskAnchor, state: GlobalIntentState, current_event: str = "") -> str:
    parts = [anchor.original_goal]
    if state.current_goal:
        parts.append(str(state.current_goal.value))
    parts.extend(str(item.value) for item in state.active_subgoals)
    parts.extend(str(item.value) for item in state.open_questions if item.status is not None)
    parts.extend(str(item.value) for item in state.next_actions)
    if current_event:
        parts.append(current_event)
    deduplicated: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = part.strip()
        if normalized and normalized not in seen:
            deduplicated.append(normalized)
            seen.add(normalized)
    return "\n".join(deduplicated)


def pack_context(
    anchor: TaskAnchor,
    state: GlobalIntentState,
    current_event: str,
    hits: list[MemoryHit],
    recent_messages: list[str] | None = None,
    retrieval_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact context that would be supplied to an LLM."""

    context = {
        "task_anchor": anchor.to_dict(),
        "global_state": state.to_dict(),
        "current_event": current_event,
        "recent_messages": recent_messages or [],
        "retrieved_memories": [
            {
                "memory_id": hit.memory_id,
                "memory_type": hit.memory_type,
                "score": round(hit.score, 4),
                "text": hit.text,
                "metadata": hit.metadata,
            }
            for hit in hits
        ],
    }
    if retrieval_plan is not None:
        context["retrieval_plan"] = retrieval_plan
    return context


__all__ = [
    "LexicalMemoryIndex",
    "MemoryDocument",
    "MemoryHit",
    "build_retrieval_query",
    "pack_context",
    "tokenize",
]
