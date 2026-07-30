import unittest

from cl_gism import LLMLoopBuilder, LoopBoundaryDecision, parse_openresearcher_row


class LlmLoopBuilderTests(unittest.TestCase):
    def test_llm_loop_builder_splits_on_new_subtask(self) -> None:
        row = {
            "qid": 999,
            "question": "When was the film released?",
            "messages": [
                {"role": "user", "content": "When was the film released?"},
                {"role": "assistant", "content": "Search Wikipedia for the release date."},
                {
                    "role": "tool",
                    "content": "Wikipedia says the film premiered on May 14, 1997 and released in the US on September 19, 1997.",
                },
                {"role": "assistant", "content": "That suggests September 19, 1997."},
                {"role": "assistant", "content": "Now let's check another source for confirmation."},
                {"role": "tool", "content": "A second site says September 19, 1997."},
                {"role": "assistant", "content": "Confirmed."},
            ],
        }
        parsed = parse_openresearcher_row(row)
        class Judge:
            def judge(self, anchor, current_loop_events, candidate_next_event):
                if "another source for confirmation" in candidate_next_event.text:
                    return LoopBoundaryDecision(
                        split=True,
                        reason="the candidate begins a new confirmation phase",
                        confidence=0.93,
                        current_loop_subgoal="Search and inspect release dates",
                        next_loop_subgoal="Confirm the answer with a second source",
                    )
                return LoopBoundaryDecision(
                    split=False,
                    reason="same search thread",
                    confidence=0.88,
                    current_loop_subgoal="Search and inspect release dates",
                )

        builder = LLMLoopBuilder(judge=Judge())
        loops = builder.build(parsed)

        self.assertEqual(len(loops), 2)
        self.assertEqual(len(loops[0].evidence_ids), 3)
        self.assertEqual(len(loops[1].evidence_ids), 3)
        self.assertTrue(loops[0].subgoal.startswith("Search and inspect release dates"))


if __name__ == "__main__":
    unittest.main()
