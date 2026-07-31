import unittest

from cl_gism import (
    HeuristicStateUpdater,
    MemoryHit,
    TaskAnchor,
    UnifiedMemoryController,
)


class FakeClient:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "task_status": "SWITCH_LOOP",
            "loop": {
                "switch": True,
                "reason": "new verification phase",
                "confidence": 0.94,
                "current_loop_subgoal": "find candidate",
                "next_loop_subgoal": "verify candidate",
            },
            "state_delta": {
                "mode": "NOOP",
                "summary": "retain candidate",
                "operations": [],
            },
            "retrieval": {
                "query": "verify Alpha",
                "selected_memory_ids": ["loop_allowed"],
                "reason": "candidate evidence is useful",
            },
        }


class UnifiedControllerTests(unittest.TestCase):
    def test_one_call_controls_loop_state_and_memory_selection(self):
        anchor = TaskAnchor(task_id="task_1", original_goal="Find the answer")
        state = HeuristicStateUpdater().initialize(anchor)
        controller = UnifiedMemoryController(FakeClient(), max_selected_memories=2)
        decision = controller.decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[MemoryHit("loop_allowed", "loop", 1.0, "Alpha evidence", {})],
        )
        self.assertTrue(decision.switch_loop)
        self.assertEqual(decision.task_status, "SWITCH_LOOP")
        self.assertEqual(decision.retrieval_query, "verify Alpha")
        self.assertEqual(decision.selected_memory_ids, ["loop_allowed"])

    def test_ready_to_answer_is_terminal_without_loop_switch(self):
        class ReadyClient:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "task_status": "READY_TO_ANSWER",
                    "loop": {
                        "switch": False,
                        "reason": "answer and citations are complete",
                        "confidence": 0.98,
                        "current_loop_subgoal": "verify Alpha",
                        "next_loop_subgoal": "",
                    },
                    "state_delta": {"mode": "NOOP", "summary": "", "operations": []},
                    "retrieval": {
                        "query": "Alpha final evidence",
                        "selected_memory_ids": [],
                        "reason": "current loop already contains citations",
                    },
                }

        anchor = TaskAnchor(task_id="task_ready", original_goal="Find Alpha")
        state = HeuristicStateUpdater().initialize(anchor)
        decision = UnifiedMemoryController(ReadyClient()).decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
        )
        self.assertEqual(decision.task_status, "READY_TO_ANSWER")
        self.assertFalse(decision.switch_loop)

    def test_rejects_long_reasoning_transcript_as_state_value(self):
        class InvalidClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_prompt):
                self.calls += 1
                return {
                    "task_status": "SWITCH_LOOP",
                    "loop": {
                        "switch": True,
                        "reason": "new phase",
                        "confidence": 0.9,
                        "current_loop_subgoal": "search",
                        "next_loop_subgoal": "verify",
                    },
                    "state_delta": {
                        "mode": "APPLY",
                        "summary": "bad transcript",
                        "operations": [{
                            "operation": "ADD",
                            "target": "working_hypotheses",
                            "value": "x" * 801,
                            "reason": "too long",
                            "evidence_ids": [],
                            "target_item_ids": [],
                        }],
                    },
                    "retrieval": {"query": "verify", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="task_invalid", original_goal="Find Alpha")
        state = HeuristicStateUpdater().initialize(anchor)
        client = InvalidClient()
        with self.assertRaisesRegex(ValueError, "at most 800"):
            UnifiedMemoryController(client).decide(
                anchor=anchor,
                state=state,
                current_loop=[],
                latest_events=[],
                candidates=[],
            )
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
