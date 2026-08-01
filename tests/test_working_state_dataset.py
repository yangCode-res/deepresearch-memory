from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_working_state_dataset.py"
SPEC = spec_from_file_location("build_working_state_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WorkingStateDatasetTest(unittest.TestCase):
    def test_decision_steps_end_at_tool_results(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"text": "plan"}]},
            {"role": "assistant", "recipient": "browser.search", "content": [{"text": '{"query":"x"}'}]},
            {"role": "tool", "name": "browser.search", "content": [{"text": "result 1"}]},
            {"role": "assistant", "content": [{"text": "interpretation"}]},
            {"role": "tool", "name": "browser.open", "content": [{"text": "result 2"}]},
            {"role": "assistant", "content": [{"text": "final answer"}]},
        ]
        steps = MODULE.decision_steps(messages)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["prefix_end_message_index"], 5)
        self.assertEqual(steps[1]["prefix_end_message_index"], 7)
        self.assertEqual(steps[1]["observed_messages"][0]["index"], 6)

    def test_apply_delta_is_deduplicated(self):
        state = MODULE.initial_global_state("question")
        delta = {name: [] for name in MODULE.DELTA_FIELDS}
        delta["completed_subgoal"] = "identify candidate"
        delta["add_confirmed_facts"] = ["fact", "fact"]
        updated = MODULE.apply_delta(state, delta)
        self.assertEqual(updated["confirmed_facts"], ["fact"])
        self.assertEqual(updated["completed_subgoals"], ["identify candidate"])

    def test_validate_continue_label(self):
        label = MODULE.output_contract(1)
        working = label["working_state_after"]
        working.update(
            {
                "loop_id": "loop_001",
                "status": "IN_PROGRESS",
                "current_subgoal": "identify candidate",
                "completion_test": "one candidate is evidenced",
                "open_aspects": ["direct support for the candidate"],
                "key_evidence": ["claim [msg_0005]"],
                "next_direction": "Establish the candidate from reliable evidence; stop once it is directly supported.",
                "expected_information_gain": "HIGH",
            }
        )
        label["loop_decision"].update(
            {
                "action": "CONTINUE_CURRENT_LOOP",
                "reason": "the candidate is not established",
                "next_subgoal": "",
                "outcome": "IN_PROGRESS",
                "boundary_basis": "NONE",
            }
        )
        label["state_delta"]["mode"] = "NOOP"
        validated = MODULE.validate_label(
            label, seen_message_ids={"msg_0005"}, allowed_memory_ids=set(), loop_number=1
        )
        self.assertEqual(validated["state_delta"]["mode"], "NOOP")


if __name__ == "__main__":
    unittest.main()
