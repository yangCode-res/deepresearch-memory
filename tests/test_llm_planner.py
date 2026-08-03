import json
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

from cl_gism import CLGISMEngine, HeuristicStateUpdater, LLMMemoryReranker, RuleBasedLoopBuilder, build_retrieval_query
from cl_gism.llm_planner import OpenAIChatJSONClient


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class LlmPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
        cls.row = json.loads(path.read_text())

    def test_llm_reranker_controls_context_order(self) -> None:
        engine = CLGISMEngine(loop_builder=RuleBasedLoopBuilder(), state_updater=HeuristicStateUpdater())
        task = engine.ingest_openresearcher_row(self.row)
        seed_query = build_retrieval_query(task.anchor, task.state, "release date")
        candidates = engine.index.search(seed_query, top_k=5, task_id=task.task_id)
        selected_ids = [candidates[1].memory_id, candidates[0].memory_id]
        reranker = LLMMemoryReranker(
            FakeClient(
                {
                    "retrieval_query": "release date evidence",
                    "selected_memory_ids": selected_ids,
                    "reason": "prioritize the strongest evidence",
                }
            ),
            candidate_pool_size=5,
            top_k=2,
        )
        engine.memory_reranker = reranker

        context = engine.build_llm_context(task.task_id, "release date", top_k=2)

        self.assertEqual(context["retrieval_plan"]["source"], "llm")
        self.assertEqual(context["retrieval_plan"]["selected_memory_ids"], selected_ids)
        self.assertEqual([item["memory_id"] for item in context["retrieved_memories"]], selected_ids)

    def test_llm_reranker_falls_back_on_invalid_selection(self) -> None:
        reranker = LLMMemoryReranker(
            FakeClient(
                {
                    "retrieval_query": "release date evidence",
                    "selected_memory_ids": ["not-a-real-id"],
                    "reason": "bad selection",
                }
            ),
            candidate_pool_size=5,
            top_k=2,
        )
        engine = CLGISMEngine(
            loop_builder=RuleBasedLoopBuilder(),
            state_updater=HeuristicStateUpdater(),
            memory_reranker=reranker,
        )
        task = engine.ingest_openresearcher_row(self.row)

        context = engine.build_llm_context(task.task_id, "release date", top_k=2)

        self.assertEqual(context["retrieval_plan"]["source"], "lexical")
        self.assertEqual(len(context["retrieved_memories"]), 2)

    def test_json_client_retries_http_429(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "Too many requests",
            {"Retry-After": "0"},
            BytesIO(b'{"error":{"message":"limited"}}'),
        )
        response = BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()
        )
        response.__enter__ = lambda item: item
        response.__exit__ = lambda *args: None
        client = OpenAIChatJSONClient(
            api_key="test",
            base_url="https://example.test/v1",
            max_attempts=2,
            retry_base_seconds=0,
        )
        with patch("urllib.request.urlopen", side_effect=[error, response]), patch(
            "time.sleep"
        ) as sleep:
            self.assertEqual(client.complete_json("system", "user"), {"ok": True})
        error.close()
        sleep.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
