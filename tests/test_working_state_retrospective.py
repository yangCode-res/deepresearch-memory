from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import copy
import sys
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = spec_from_file_location(
    "build_working_state_retrospective",
    SCRIPTS / "build_working_state_retrospective.py",
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decision(
    index: int,
    number: int,
    action: str,
    *,
    subgoal: str,
    outcome: str | None = None,
    basis: str | None = None,
):
    if outcome is None:
        outcome = "IN_PROGRESS" if action == "CONTINUE_CURRENT_LOOP" else "RESOLVED"
    if basis is None:
        basis = "NONE" if action == "CONTINUE_CURRENT_LOOP" else (
            "TASK_COMPLETE" if action == "READY_TO_ANSWER" else "SUBGOAL_COMPLETED"
        )
    return {
        "decision_index": index,
        "loop_number": number,
        "current_subgoal": subgoal,
        "completion_test": f"Evidence establishes whether to {subgoal.lower()}",
        "action": action,
        "outcome": outcome,
        "boundary_basis": basis,
        "boundary_reason": "The evidence objective has reached its retrospective boundary.",
        "boundary_message_ids": [f"msg_{6 + index * 3:04d}"],
    }


class RetrospectiveBuilderTest(unittest.TestCase):
    def test_full_trajectory_view_includes_final_after_last_tool(self):
        messages = [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reason"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "channel": "final", "content": "final answer"},
        ]
        view, count = MODULE.trajectory_view(messages)
        self.assertEqual(count, 1)
        self.assertEqual([item["role"] for item in view], ["assistant", "tool", "assistant"])
        self.assertEqual(view[-1]["text"], "final answer")
        self.assertEqual(view[1]["decision_index"], 0)
        lookahead = MODULE.decision_lookahead_view(messages)
        self.assertEqual(lookahead[0]["following_assistant_messages_before_next_tool"][0]["text"], "final answer")

    def test_segmentation_maps_decisions_to_loops(self):
        raw = {
            "trajectory_summary": "identify candidate, verify date, then answer",
            "decisions": [
                decision(0, 1, "CONTINUE_CURRENT_LOOP", subgoal="Identify the target work"),
                decision(1, 1, "SWITCH_LOOP", subgoal="Identify the target work"),
                decision(2, 2, "CONTINUE_CURRENT_LOOP", subgoal="Establish the requested date"),
                decision(
                    3,
                    2,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                ),
            ],
        }
        segmented = MODULE.validate_segmentation(
            raw,
            decision_count=4,
            decision_message_limits=[6, 9, 12, 15],
            has_final_answer=True,
        )
        self.assertEqual(len(segmented["loops"]), 2)
        self.assertEqual(segmented["loops"][0]["end_decision_index"], 1)
        self.assertEqual(
            MODULE.boundary_for_decision(segmented, 0)["action"],
            "CONTINUE_CURRENT_LOOP",
        )
        boundary = MODULE.boundary_for_decision(segmented, 1)
        self.assertEqual(boundary["action"], "SWITCH_LOOP")
        self.assertEqual(boundary["next_subgoal"], "Establish the requested date")
        self.assertEqual(
            MODULE.boundary_for_decision(segmented, 3)["action"],
            "READY_TO_ANSWER",
        )

    def test_segmentation_rejects_gap(self):
        raw = {
            "trajectory_summary": "two stages",
            "decisions": [
                decision(0, 1, "SWITCH_LOOP", subgoal="Identify the target work"),
                decision(
                    2,
                    2,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "consecutive"):
            MODULE.validate_segmentation(raw, decision_count=2)

    def test_segmentation_must_cover_all_decisions(self):
        raw = {
            "trajectory_summary": "unfinished research",
            "decisions": [
                decision(
                    0,
                    1,
                    "CONTINUE_CURRENT_LOOP",
                    subgoal="Establish the requested date",
                ),
                decision(
                    1,
                    1,
                    "CONTINUE_CURRENT_LOOP",
                    subgoal="Establish the requested date",
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "annotate every"):
            MODULE.validate_segmentation(raw, decision_count=3)

    def test_segmentation_allows_retrospective_lookahead_coordinate(self):
        raw = {
            "trajectory_summary": "one decision",
            "decisions": [
                decision(
                    0,
                    1,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                )
            ],
        }
        raw["decisions"][0]["boundary_message_ids"] = ["msg_0009"]
        segmented = MODULE.validate_segmentation(
            raw,
            decision_count=1,
            decision_message_limits=[6],
            trajectory_message_limit=9,
            has_final_answer=True,
        )
        self.assertEqual(segmented["decisions"][0]["boundary_message_ids"], ["msg_0009"])

    def test_segmentation_rejects_ready_before_final_decision(self):
        raw = {
            "trajectory_summary": "agent keeps researching after the first observation",
            "decisions": [
                decision(0, 1, "READY_TO_ANSWER", subgoal="Establish the requested date"),
                decision(1, 1, "READY_TO_ANSWER", subgoal="Establish the requested date"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "final tool-result"):
            MODULE.validate_segmentation(raw, decision_count=2, has_final_answer=True)

    def test_segmentation_normalizes_action_dependent_enums(self):
        raw = {
            "trajectory_summary": "answer becomes available immediately",
            "decisions": [
                decision(
                    0,
                    1,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                    outcome="IN_PROGRESS",
                    basis="NONE",
                )
            ],
        }
        segmented = MODULE.validate_segmentation(raw, decision_count=1)
        item = segmented["decisions"][0]
        self.assertEqual(item["outcome"], "RESOLVED")
        self.assertEqual(item["boundary_basis"], "TASK_COMPLETE")

    def test_causal_payload_hides_next_loop_contract(self):
        boundary = {
            "action": "SWITCH_LOOP",
            "current_subgoal": "Identify the target work",
            "current_completion_test": "The target identity is established",
            "next_subgoal": "SECRET FUTURE SUBGOAL",
            "next_completion_test": "SECRET FUTURE TEST",
            "outcome": "RESOLVED",
            "boundary_basis": "SUBGOAL_COMPLETED",
        }
        payload = MODULE.causal_teacher_payload(
            question="question",
            global_state={},
            working_state={},
            prior_memories=[],
            observed_messages=[],
            boundary=boundary,
            loop_number=1,
            seen_message_ids=set(),
        )
        rendered = str(payload)
        self.assertNotIn("SECRET FUTURE", rendered)
        self.assertNotIn("next_loop_contract", payload)

    def test_causality_audit_rejects_future_ids_anywhere_in_input_or_target(self):
        record = {
            "source": {"prefix_end_message_index": 7},
            "input": {
                "observed_messages": [{"index": 7}],
                "current_loop_evidence_ids": ["msg_0007"],
                "working_state_before": {"progress_summary": "future msg_0008"},
            },
            "target": {"summary": "supported by msg_0007"},
        }
        errors = MODULE.audit_record_causality(record)
        self.assertIn("training input contains a future evidence ID", errors)
        clean = copy.deepcopy(record)
        clean["input"]["working_state_before"]["progress_summary"] = "past msg_0006"
        clean["target"]["summary"] = "future msg_0009"
        errors = MODULE.audit_record_causality(clean)
        self.assertIn("training target cites a future message ID", errors)

    def test_validate_causal_raw_requires_key_evidence_citation(self):
        empty_ops = {
            name: ([] if name != "completed_subgoal" else "")
            for name in MODULE.DELTA_FIELDS
        }
        raw = {
            "decision_reason": "The evidence objective remains incomplete.",
            "progress": {
                "progress_summary": "A candidate exists.",
                "resolved_aspects": [],
                "open_aspects": ["The exact date remains unsupported."],
                "key_evidence": ["A candidate exists without a citation"],
                "candidate_answer": "",
                "active_hypotheses": [],
                "failed_strategies": [],
                "evidence_gaps": ["Direct support is missing."],
                "answer_stable": False,
                "evidence_sufficient": False,
                "expected_information_gain": "HIGH",
            },
            "durable_update": empty_ops,
            "loop_memory": None,
            "retrieval": {"query": "identity", "relevant_memory_ids": [], "reason": "none"},
        }
        boundary = {
            "action": "CONTINUE_CURRENT_LOOP",
            "reason": "",
            "current_subgoal": "Identify the target work",
            "current_completion_test": "The target identity is established",
            "next_subgoal": "",
            "next_completion_test": "",
            "outcome": "IN_PROGRESS",
            "boundary_basis": "NONE",
            "confidence": 1.0,
            "progress": {},
        }
        with self.assertRaisesRegex(ValueError, "key_evidence"):
            MODULE.validate_causal_raw(
                raw,
                boundary=boundary,
                working_before=MODULE.initial_working_state(),
                loop_number=1,
                seen_message_ids={"msg_0006"},
                allowed_memory_ids=set(),
            )

    def test_continue_mechanically_discards_durable_output(self):
        operations = {
            name: ([] if name != "completed_subgoal" else "")
            for name in MODULE.DELTA_FIELDS
        }
        operations["add_working_hypotheses"] = ["A provisional candidate"]
        raw = {
            "decision_reason": "The same information objective remains open.",
            "progress": {
                "progress_summary": "A candidate path is visible in msg_0006.",
                "resolved_aspects": [],
                "open_aspects": ["The exact answer remains unsupported."],
                "key_evidence": ["msg_0006 shows a candidate source."],
                "candidate_answer": "",
                "active_hypotheses": ["The candidate may contain the answer."],
                "failed_strategies": [],
                "evidence_gaps": ["Direct support is missing."],
                "answer_stable": False,
                "evidence_sufficient": False,
                "expected_information_gain": "HIGH",
            },
            "durable_update": operations,
            "loop_memory": {"unexpected": True},
            "retrieval": {"query": "candidate evidence", "relevant_memory_ids": [], "reason": "none"},
        }
        boundary = {
            "action": "CONTINUE_CURRENT_LOOP",
            "reason": "",
            "current_subgoal": "Identify the requested fact",
            "current_completion_test": "Direct evidence establishes the requested fact",
            "next_subgoal": "",
            "next_completion_test": "",
            "outcome": "IN_PROGRESS",
            "boundary_basis": "NONE",
            "confidence": 1.0,
            "progress": {},
        }
        target = MODULE.validate_causal_raw(
            raw,
            boundary=boundary,
            working_before=MODULE.initial_working_state(),
            loop_number=1,
            seen_message_ids={"msg_0006"},
            allowed_memory_ids=set(),
        )
        self.assertEqual(target["state_delta"]["mode"], "NOOP")
        self.assertIsNone(target["cross_loop_memory"])
        self.assertFalse(any(target["state_delta"]["operations"].values()))


if __name__ == "__main__":
    unittest.main()
