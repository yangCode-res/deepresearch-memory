from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = spec_from_file_location(
    "repair_retrospective_segment_contracts",
    SCRIPTS / "repair_retrospective_segment_contracts.py",
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RetrospectiveContractRepairTest(unittest.TestCase):
    def test_source_wording_is_normalized_to_evidence(self):
        normalized = MODULE.normalize_repair_text(
            "A primary source confirms the claim and multiple independent sources verify the tenure."
        )
        self.assertNotRegex(normalized.casefold(), r"\bsources?\b")
        self.assertIn("evidence", normalized.casefold())

    def test_database_source_is_normalized_to_evidence(self):
        normalized = MODULE.normalize_repair_text(
            "Confirm the attribution with an authoritative film database source."
        )
        self.assertNotRegex(normalized.casefold(), r"\bsource\b")
        self.assertIn("database evidence", normalized.casefold())

    def test_validator_rejects_generic_contract(self):
        original = [{"loop_number": 1}]
        raw = {
            "loops": [
                {
                    "loop_number": 1,
                    "subgoal": "Establish information dependency 1",
                    "completion_test": "Evidence resolves information dependency 1",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "generic"):
            MODULE.validate_repair(
                raw,
                original_loops=original,
                activation_texts={1: "Which person held the office?"},
            )

    def test_validator_accepts_specific_information_contract(self):
        original = [{"loop_number": 1}]
        repaired = MODULE.validate_repair(
            {
                "loops": [
                    {
                        "loop_number": 1,
                        "subgoal": "Identify the officeholder whose tenure overlaps the specified period",
                        "completion_test": "Evidence establishes the officeholder and the overlapping tenure dates",
                    }
                ]
            },
            original_loops=original,
            activation_texts={1: "Who held the office during the specified period?"},
        )
        self.assertEqual(repaired["loops"][0]["loop_number"], 1)
        self.assertEqual(
            repaired["loops"][0]["completion_test"],
            "Evidence is sufficient to identify the officeholder whose tenure overlaps the specified period",
        )

    def test_validator_allows_website_as_the_requested_entity_type(self):
        repaired = MODULE.validate_repair(
            {
                "loops": [
                    {
                        "loop_number": 1,
                        "subgoal": "Identify which website matches the described research-library features",
                        "completion_test": "The matching website is identified",
                    }
                ]
            },
            original_loops=[{"loop_number": 1}],
            activation_texts={1: "Which website has the described research library?"},
        )
        self.assertIn("which website", repaired["loops"][0]["subgoal"])

    def test_validator_abstracts_a_future_film_year(self):
        repaired = MODULE.validate_repair(
            {
                "loops": [
                    {
                        "loop_number": 1,
                        "subgoal": "Identify the actor in the 1971 film",
                        "completion_test": "The actor is identified",
                    }
                ]
            },
            original_loops=[{"loop_number": 1}],
            activation_texts={1: "Identify the actor in the named film."},
        )
        self.assertEqual(repaired["loops"][0]["subgoal"], "Identify the actor in the specified film")


if __name__ == "__main__":
    unittest.main()
