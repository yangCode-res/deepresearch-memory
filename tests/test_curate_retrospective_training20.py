from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from copy import deepcopy
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
    def test_completion_is_derived_from_specific_subgoal(self):
        self.assertEqual(
            MODULE.completion_from_subgoal("Identify the requested founding year"),
            "Evidence is sufficient to identify the requested founding year",
        )

    def test_contract_normalizer_keeps_website_as_answer_type(self):
        value = "Identify which website matches the described research-library features"
        self.assertEqual(
            MODULE.normalize_curated_contract(value, fallback="generic fallback"),
            value,
        )

    def test_contract_override_repairs_text_without_changing_boundary(self):
        segments = {
            "7547": {
                "segmentation": {
                    "loops": [
                        {
                            "loop_number": 2,
                            "start_decision_index": 2,
                            "end_decision_index": 10,
                            "subgoal": "Determine Joanna the requested values father-in-law",
                            "completion_test": "Find the fatherthe requested values biography",
                            "end_action": "READY_TO_ANSWER",
                        }
                    ]
                }
            }
        }
        applied = MODULE.apply_contract_overrides(
            segments,
            {
                "7547": {
                    "2": {
                        "subgoal": "Determine Joanna's father",
                        "completion_test": "Evidence establishes who Joanna's father was",
                    }
                }
            },
        )

        loop = segments["7547"]["segmentation"]["loops"][0]
        self.assertEqual(applied, ["7547"])
        self.assertEqual(loop["subgoal"], "Determine Joanna's father")
        self.assertEqual(loop["start_decision_index"], 2)
        self.assertEqual(loop["end_decision_index"], 10)
        self.assertEqual(loop["end_action"], "READY_TO_ANSWER")

    def test_state_item_artifact_falls_back_to_repaired_contract(self):
        fallback = "Evidence establishes who Joanna's father was"
        artifacts = [
            "Find the fatherthe requested values biography",
            "Extract the requested value from the paper",
            "Establish evidence that s (WHO, CDC) describe transmission",
            "The the available evidence page confirms the claim",
            "Evidence resolves information dependency 2",
        ]
        for artifact in artifacts:
            with self.subTest(artifact=artifact):
                self.assertEqual(
                    MODULE.normalize_state_item(artifact, fallback=fallback),
                    fallback,
                )

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

    def test_normalization_cleans_direction_and_preserves_replay_chain(self):
        before = {
            "loop_id": "loop_001",
            "current_subgoal": "initial goal",
            "completion_test": "initial test",
        }
        after = {
            "loop_id": "loop_001",
            "current_subgoal": "Find another source confirming the requested fact",
            "completion_test": "A webpage is opened that contains the requested fact",
            "open_aspects": ["Find another source confirming the requested fact"],
            "evidence_gaps": ["Need to view Doc 42"],
            "next_direction": "view Doc 42",
        }
        rows = [
            {
                "source": {"qid": "qid-1", "step_index": 0},
                "input": {"working_state_before": deepcopy(before)},
                "target": {
                    "loop_decision": {
                        "current_subgoal": "Find another source confirming the requested fact",
                        "next_subgoal": "",
                    },
                    "working_state_after": deepcopy(after),
                },
            },
            {
                "source": {"qid": "qid-1", "step_index": 1},
                "input": {"working_state_before": deepcopy(after)},
                "target": {
                    "loop_decision": {
                        "current_subgoal": "Find another source confirming the requested fact",
                        "next_subgoal": "",
                    },
                    "working_state_after": deepcopy(after),
                },
            },
        ]
        segments = {
            "qid-1": {
                "segmentation": {
                    "loops": [
                        {
                            "subgoal": "Find another source confirming the requested fact",
                            "completion_test": "A webpage is opened that contains the requested fact",
                        }
                    ]
                }
            }
        }

        MODULE.normalize_directional_contracts({"qid-1": rows}, segments)

        rendered = str(rows[0]["target"]["working_state_after"])
        self.assertFalse(MODULE.GLOBAL_CONCRETE_PATTERN.search(rendered))
        self.assertEqual(
            rows[0]["target"]["loop_decision"]["current_subgoal"],
            "Establish evidence that the requested fact",
        )
        self.assertEqual(
            rows[0]["target"]["working_state_after"],
            rows[1]["input"]["working_state_before"],
        )

    def test_normalization_scrubs_future_number_at_loop_activation(self):
        row = {
            "source": {"qid": "qid-2", "step_index": 0},
            "input": {
                "question": "According to the article, what percentage is reported?",
                "observed_messages": [{"text": "The correct article was identified."}],
                "working_state_before": {"loop_id": "loop_001"},
            },
            "target": {
                "loop_decision": {
                    "current_subgoal": "Extract the requested percentage",
                    "next_subgoal": "",
                },
                "working_state_after": {
                    "loop_id": "loop_001",
                    "current_subgoal": "Extract the requested percentage",
                    "completion_test": "Evidence verifies the exact 15% statistic",
                    "open_aspects": [],
                    "evidence_gaps": [],
                    "next_direction": "",
                },
            },
        }

        MODULE.normalize_directional_contracts({"qid-2": [row]}, {})

        self.assertNotIn("15%", row["target"]["working_state_after"]["completion_test"])


if __name__ == "__main__":
    unittest.main()
