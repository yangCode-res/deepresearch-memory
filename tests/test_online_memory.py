import unittest

from cl_gism import LLMLoopBoundaryJudge, LLMStateUpdater, OnlineMemorySession


class FakeController:
    model = "fake-controller"

    def complete_json(self, system_prompt, user_prompt):
        if "segment deep-research trajectories" in system_prompt:
            return {
                "split": True,
                "reason": "the assistant starts a new evidence thread",
                "confidence": 0.9,
                "current_loop_subgoal": "Find the first source",
                "next_loop_subgoal": "Confirm with a second source",
            }
        return {
            "summary": "retain the first source as a hypothesis",
            "operations": [
                {
                    "operation": "ADD",
                    "target": "working_hypotheses",
                    "value": "The first source suggests the answer is Alpha.",
                    "reason": "The completed loop found a candidate answer.",
                    "evidence_ids": [],
                    "item": {
                        "status": "tentative",
                        "confidence": 0.7,
                        "source_type": "agent",
                        "valid_time": None,
                        "contradicts": [],
                        "supersedes": [],
                        "user_confirmed": False,
                    },
                }
            ],
        }


class OnlineMemoryTests(unittest.TestCase):
    def test_cross_loop_prompt_keeps_current_loop_and_retrieves_old_loop(self):
        controller = FakeController()
        session = OnlineMemorySession(
            qid=42,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(controller),
            state_updater=LLMStateUpdater(controller),
            top_k=3,
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
        ]
        first_prompt = session.build_prompt(messages)
        self.assertEqual(len(first_prompt), 2)

        messages.extend(
            [
                {"role": "assistant", "content": "Search for Alpha."},
                {"role": "tool", "content": "The first source says Alpha."},
            ]
        )
        session.build_prompt(messages)

        messages.extend(
            [
                {"role": "assistant", "content": "Now confirm Alpha independently."},
                {"role": "tool", "content": "A second source also mentions Alpha."},
            ]
        )
        compact = session.build_prompt(messages)

        self.assertEqual(len(session.completed_loops), 1)
        self.assertEqual(session.state.state_version, 2)
        self.assertEqual(len(compact), 4)
        self.assertEqual(compact[-2]["content"], "Now confirm Alpha independently.")
        self.assertTrue(session.traces[-1].loop_switched)
        self.assertGreater(len(session.traces[-1].retrieved_memory_ids), 0)
        self.assertIn("cross_loop_memory", compact[0]["content"])


if __name__ == "__main__":
    unittest.main()
