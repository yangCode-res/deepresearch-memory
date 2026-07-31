import unittest

from cl_gism import (
    LLMLoopBoundaryJudge,
    LLMStateUpdater,
    OnlineMemorySession,
    UnifiedControlDecision,
)


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

    def test_unified_controller_ready_signal_forces_answer_prompt(self):
        class ReadyController:
            def decide(self, **kwargs):
                return UnifiedControlDecision(
                    task_status="READY_TO_ANSWER",
                    switch_loop=False,
                    reason="answer and citations are complete",
                    confidence=0.99,
                    state_delta={"mode": "NOOP", "summary": "", "operations": []},
                    retrieval_query="Alpha final evidence",
                )

        session = OnlineMemorySession(
            qid=43,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(FakeController()),
            state_updater=LLMStateUpdater(FakeController()),
            unified_controller=ReadyController(),
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
        ]
        initial = session.build_prompt(messages)
        self.assertNotIn("cross_loop_memory", initial[0]["content"])

        messages.extend([
            {"role": "assistant", "content": "Alpha is supported by 【1†L1-L3】."},
            {"role": "tool", "content": "[1] Alpha evidence\nL1: Alpha"},
        ])
        final_prompt = session.build_prompt(messages)
        self.assertEqual(session.traces[-1].task_status, "READY_TO_ANSWER")
        self.assertIn("Do not call tools", final_prompt[0]["content"])
        self.assertIn("Alpha is supported", final_prompt[-2]["content"])

    def test_invalid_controller_delta_archives_with_noop_instead_of_polluting_state(self):
        class SwitchingController:
            def __init__(self):
                self.calls = 0

            def decide(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return UnifiedControlDecision(
                        task_status="CONTINUE",
                        switch_loop=False,
                        state_delta={"mode": "NOOP", "summary": "", "operations": []},
                    )
                return UnifiedControlDecision(
                    task_status="SWITCH_LOOP",
                    switch_loop=True,
                    reason="genuinely new candidate",
                    confidence=0.9,
                    current_loop_subgoal="find candidate",
                    next_loop_subgoal="verify candidate",
                    state_delta={
                        "mode": "APPLY",
                        "summary": "invalid nested item",
                        "operations": [{
                            "operation": "ADD",
                            "target": "working_hypotheses",
                            "value": "Alpha",
                            "reason": "candidate found",
                            "item": {"status": "not-a-status"},
                        }],
                    },
                )

        session = OnlineMemorySession(
            qid=44,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(FakeController()),
            state_updater=LLMStateUpdater(FakeController()),
            unified_controller=SwitchingController(),
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "Search Alpha."},
            {"role": "tool", "content": "Alpha result."},
        ]
        session.build_prompt(messages)
        messages.extend([
            {"role": "assistant", "content": "Now verify Beta."},
            {"role": "tool", "content": "Beta result."},
        ])
        session.build_prompt(messages)

        self.assertEqual(len(session.completed_loops), 1)
        self.assertEqual(session.state.state_version, 1)
        self.assertEqual(session.state.working_hypotheses, [])
        self.assertIn("not-a-status", session.traces[-1].controller_error)


if __name__ == "__main__":
    unittest.main()
