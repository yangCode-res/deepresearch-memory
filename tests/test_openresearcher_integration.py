import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


INTEGRATION_PATH = (
    Path(__file__).resolve().parents[1] / "integrations" / "openresearcher_memory_agent.py"
)


def load_integration_module():
    async def unused_run_one(**kwargs):
        return []

    deploy_agent = SimpleNamespace(
        run_one=unused_run_one,
        DEVELOPER_CONTENT="research with tools",
    )
    spec = importlib.util.spec_from_file_location("test_memory_agent_integration", INTEGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"deploy_agent": deploy_agent}):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class FakeUnderlyingTokenizer:
    def __init__(self):
        self.messages = []
        self.tools = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.tools = kwargs.get("tools")
        return "final prompt"

    def encode(self, prompt, add_special_tokens=False):
        return [1, 2, 3]


class ForcedFinalAnswerTests(unittest.IsolatedAsyncioTestCase):
    def test_deterministic_fallback_cites_only_candidate_evidence(self):
        module = load_integration_module()
        signal = module.ReadyToAnswerSignal(
            "What is the brand?",
            [],
            [
                {
                    "role": "system",
                    "content": '<cross_loop_memory>{"candidate_answer":"Vakkorama"}</cross_loop_memory>',
                },
                {"role": "user", "content": "What is the brand?"},
                {"role": "tool", "content": "[12] Video Game Orchestra evidence"},
                {"role": "tool", "content": "[35] Cem Hakko established Vakkorama in 1982."},
            ],
        )

        content = module._deterministic_final_fallback(signal)

        self.assertIn("Exact Answer: Vakkorama", content)
        self.assertIn("[35]", content)
        self.assertNotIn("[12]", content)
    async def test_ready_signal_forces_one_tool_free_final_generation(self):
        module = load_integration_module()
        underlying = FakeUnderlyingTokenizer()
        router = SimpleNamespace(_tokenizer=underlying)
        generator = SimpleNamespace(tokenizer=router)
        signal = module.ReadyToAnswerSignal(
            "What is the answer?",
            [
                {"role": "system", "content": "research with tools"},
                {"role": "user", "content": "What is the answer?"},
            ],
            [
                {
                    "role": "system",
                    "content": (
                        "research with tools\n"
                        "<cross_loop_memory>verified evidence 【1†L1-L2】</cross_loop_memory>"
                    ),
                },
                {"role": "user", "content": "What is the answer?"},
                {"role": "tool", "content": "verified evidence 【1†L1-L2】"},
            ],
        )
        calls = 0

        async def fake_generate(generator_arg, tokens):
            nonlocal calls
            calls += 1
            return (
                "<think>Use the verified evidence.</think>"
                "Explanation: Evidence supports Alpha 【1†L1-L2】.\n"
                "Exact Answer: Alpha\nConfidence: 99%"
            )

        module._generate_bounded_final = fake_generate
        messages = await module._force_final_answer(signal, generator)

        self.assertEqual(calls, 1)
        self.assertEqual(underlying.tools, [])
        self.assertEqual(len(underlying.messages), 2)
        self.assertNotIn("research with tools", underlying.messages[0]["content"])
        self.assertIn("<cross_loop_memory>", underlying.messages[1]["content"])
        self.assertNotIn('"role": "tool"', underlying.messages[1]["content"])
        self.assertEqual(messages[-1]["tool_calls"], None)
        self.assertIn("Exact Answer: Alpha", messages[-1]["content"])
        self.assertEqual(messages[-1]["reasoning_content"], "Use the verified evidence.")


if __name__ == "__main__":
    unittest.main()
