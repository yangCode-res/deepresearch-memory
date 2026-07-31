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
        self.assertEqual(decision.retrieval_query, "verify Alpha")
        self.assertEqual(decision.selected_memory_ids, ["loop_allowed"])


if __name__ == "__main__":
    unittest.main()
