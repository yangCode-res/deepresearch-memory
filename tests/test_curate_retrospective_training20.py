from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = spec_from_file_location(
    "curate_retrospective_training20",
    SCRIPTS / "curate_retrospective_training20.py",
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample(step: int, action: str, *, memories: list[str] | None = None):
    return {
        "source": {"qid": "qid-1", "step_index": step},
        "target": {
            "loop_decision": {"action": action},
            "retrieval": {"relevant_memory_ids": memories or []},
        },
    }


class RetrospectiveCuratorTest(unittest.TestCase):
    def test_choose_four_covers_both_sides_of_boundary(self):
        group = [
            sample(0, "CONTINUE_CURRENT_LOOP"),
            sample(1, "CONTINUE_CURRENT_LOOP"),
            sample(2, "SWITCH_LOOP"),
            sample(3, "CONTINUE_CURRENT_LOOP", memories=["memory_loop_001"]),
            sample(4, "CONTINUE_CURRENT_LOOP"),
            sample(5, "READY_TO_ANSWER"),
        ]

        chosen = MODULE.choose_four(group)
        by_role = {role: row for row, role in chosen}

        self.assertEqual(int(by_role["continue_before_switch"]["source"]["step_index"]), 1)
        self.assertEqual(int(by_role["continue_after_switch"]["source"]["step_index"]), 3)
        self.assertEqual(int(by_role["switch_boundary"]["source"]["step_index"]), 2)
        self.assertEqual(int(by_role["ready_boundary"]["source"]["step_index"]), 5)

    def test_choose_four_prefers_positive_post_switch_retrieval(self):
        group = [
            sample(0, "CONTINUE_CURRENT_LOOP"),
            sample(1, "SWITCH_LOOP"),
            sample(2, "CONTINUE_CURRENT_LOOP"),
            sample(3, "CONTINUE_CURRENT_LOOP", memories=["memory_loop_001"]),
            sample(4, "READY_TO_ANSWER"),
        ]

        chosen = MODULE.choose_four(group)
        after = next(row for row, role in chosen if role == "continue_after_switch")

        self.assertEqual(int(after["source"]["step_index"]), 3)
        self.assertTrue(MODULE.positive_retrieval_after_switch(group))

    def test_choose_four_rejects_incomplete_action_coverage(self):
        group = [
            sample(0, "CONTINUE_CURRENT_LOOP"),
            sample(1, "CONTINUE_CURRENT_LOOP"),
            sample(2, "READY_TO_ANSWER"),
        ]

        with self.assertRaisesRegex(ValueError, "SWITCH"):
            MODULE.choose_four(group)


if __name__ == "__main__":
    unittest.main()
