import json
from pathlib import Path
import unittest

from cl_gism import LLMStateUpdater, RuleBasedLoopBuilder, parse_openresearcher_row


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class LlmStateUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
        cls.row = json.loads(path.read_text())

    def test_llm_state_updater_applies_planned_delta(self) -> None:
        parsed = parse_openresearcher_row(self.row)
        loop = RuleBasedLoopBuilder().build(parsed)[0]
        updater = LLMStateUpdater(
            FakeClient(
                {
                    "summary": "Record the tentative date and close the open question.",
                    "operations": [
                        {
                            "operation": "ADD",
                            "target": "working_hypotheses",
                            "value": "The film was likely released on September 19, 1997.",
                            "reason": "The loop surfaced a release-date excerpt.",
                            "item": {
                                "status": "tentative",
                                "confidence": 0.74,
                                "source_type": "agent",
                                "user_confirmed": False,
                            },
                        },
                        {
                            "operation": "RESOLVE",
                            "target": "open_questions",
                            "reason": "The loop resolved the search question for this test.",
                        },
                    ],
                }
            )
        )

        state = updater.initialize(parsed.anchor)
        result = updater.update(parsed.anchor, state, loop)

        self.assertEqual(result.state.state_version, 2)
        self.assertEqual(len(result.state.working_hypotheses), 1)
        self.assertEqual(result.state.working_hypotheses[0].value, "The film was likely released on September 19, 1997.")
        self.assertTrue(all(item.status.value == "resolved" for item in result.state.open_questions))
        self.assertEqual(result.delta.generated_from_loop_id, loop.loop_id)
        self.assertEqual(result.delta.operations[0].target, "working_hypotheses")

    def test_llm_state_updater_falls_back_on_invalid_plan(self) -> None:
        parsed = parse_openresearcher_row(self.row)
        loop = RuleBasedLoopBuilder().build(parsed)[0]
        updater = LLMStateUpdater(
            FakeClient(
                {
                    "summary": "bad plan",
                    "operations": [
                        {
                            "operation": "ADD",
                            "target": "unknown_target",
                            "value": "bad",
                            "reason": "unsupported",
                        }
                    ],
                }
            )
        )

        state = updater.initialize(parsed.anchor)
        result = updater.update(parsed.anchor, state, loop)

        self.assertEqual(result.state.state_version, 2)
        self.assertGreaterEqual(len(result.delta.operations), 1)


if __name__ == "__main__":
    unittest.main()
