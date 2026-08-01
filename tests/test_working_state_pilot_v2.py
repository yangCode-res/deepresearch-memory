from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = spec_from_file_location("build_working_state_pilot_v2", SCRIPTS / "build_working_state_pilot_v2.py")
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WorkingStatePilotV2Test(unittest.TestCase):
    def test_continue_must_preserve_committed_contract(self):
        raw = MODULE.boundary_contract()
        raw.update(
            {
                "action": "CONTINUE_CURRENT_LOOP",
                "reason": "same evidence goal remains open",
                "current_subgoal": "verify identity",
                "current_completion_test": "identity has direct support",
                "next_subgoal": "",
                "next_completion_test": "",
                "outcome": "IN_PROGRESS",
                "boundary_basis": "NONE",
                "confidence": 0.8,
            }
        )
        validated = MODULE.validate_boundary(
            raw,
            committed_subgoal="verify identity",
            committed_completion_test="identity has direct support",
            first_step=False,
        )
        self.assertEqual(validated["action"], "CONTINUE_CURRENT_LOOP")
        raw["current_subgoal"] = "verify release date"
        normalized = MODULE.validate_boundary(
            raw,
            committed_subgoal="verify identity",
            committed_completion_test="identity has direct support",
            first_step=False,
        )
        self.assertEqual(normalized["current_subgoal"], "verify identity")

    def test_switch_requires_distinct_next_contract(self):
        raw = MODULE.boundary_contract()
        raw.update(
            {
                "action": "SWITCH_LOOP",
                "reason": "identity resolved; date remains",
                "current_subgoal": "verify identity",
                "current_completion_test": "identity has direct support",
                "next_subgoal": "verify release date",
                "next_completion_test": "date has direct support",
                "outcome": "RESOLVED",
                "boundary_basis": "SUBGOAL_COMPLETED",
                "confidence": 0.9,
            }
        )
        validated = MODULE.validate_boundary(
            raw,
            committed_subgoal="verify identity",
            committed_completion_test="identity has direct support",
            first_step=False,
        )
        self.assertEqual(validated["next_subgoal"], "verify release date")

    def test_switch_rejects_source_strategy_rephrasing(self):
        raw = MODULE.boundary_contract()
        raw.update(
            {
                "action": "SWITCH_LOOP",
                "reason": "the first source failed",
                "current_subgoal": "Find a reliable source listing states bordering Lake Superior",
                "current_completion_test": "A source lists all bordering states",
                "next_subgoal": "Find a different accessible source listing states bordering Lake Superior",
                "next_completion_test": "A different source lists all bordering states",
                "outcome": "BLOCKED",
                "boundary_basis": "BLOCKED_OR_SATURATED",
                "confidence": 0.8,
            }
        )
        with self.assertRaisesRegex(ValueError, "rephrase the same"):
            MODULE.validate_boundary(
                raw,
                committed_subgoal=raw["current_subgoal"],
                committed_completion_test=raw["current_completion_test"],
                first_step=False,
            )


if __name__ == "__main__":
    unittest.main()
