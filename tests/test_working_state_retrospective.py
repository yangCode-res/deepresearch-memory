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


def loop_record(
    number: int,
    start: int,
    end: int,
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
        "loop_number": number,
        "start_decision_index": start,
        "end_decision_index": end,
        "subgoal": subgoal,
        "completion_test": f"Evidence establishes whether to {subgoal.lower()}",
        "end_action": action,
        "outcome": outcome,
        "boundary_basis": basis,
        "boundary_reason": "The evidence objective has reached its retrospective boundary.",
    }


class RetrospectiveBuilderTest(unittest.TestCase):
    def test_excluded_qids_load_from_json_or_lines(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "qids.json"
            json_path.write_text('["100", 200]', encoding="utf-8")
            self.assertEqual(
                MODULE.load_excluded_qids(["300"], json_path),
                {"100", "200", "300"},
            )
            line_path = Path(directory) / "qids.txt"
            line_path.write_text("400\n500\n", encoding="utf-8")
            self.assertEqual(MODULE.load_excluded_qids([], line_path), {"400", "500"})

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
            "loops": [
                loop_record(1, 0, 1, "SWITCH_LOOP", subgoal="Identify the target work"),
                loop_record(
                    2,
                    2,
                    3,
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
            "loops": [
                loop_record(1, 0, 0, "SWITCH_LOOP", subgoal="Identify the target work"),
                loop_record(
                    2,
                    2,
                    2,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "contiguously"):
            MODULE.validate_segmentation(raw, decision_count=3)

    def test_segmentation_must_cover_all_decisions(self):
        raw = {
            "trajectory_summary": "unfinished research",
            "loops": [
                loop_record(
                    1,
                    0,
                    1,
                    "CONTINUE_CURRENT_LOOP",
                    subgoal="Establish the requested date",
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "cover every"):
            MODULE.validate_segmentation(raw, decision_count=3)

    def test_segmentation_requires_ready_when_final_answer_follows(self):
        raw = {
            "trajectory_summary": "one decision",
            "loops": [
                loop_record(
                    1,
                    0,
                    0,
                    "CONTINUE_CURRENT_LOOP",
                    subgoal="Establish the requested date",
                )
            ],
        }
        with self.assertRaisesRegex(ValueError, "requires READY"):
            MODULE.validate_segmentation(raw, decision_count=1, has_final_answer=True)

    def test_segmentation_rejects_nonfinal_ready_loop(self):
        raw = {
            "trajectory_summary": "agent keeps researching after the first observation",
            "loops": [
                loop_record(1, 0, 0, "READY_TO_ANSWER", subgoal="Establish the requested date"),
                loop_record(2, 1, 1, "READY_TO_ANSWER", subgoal="Establish the requested country"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "non-final Loop"):
            MODULE.validate_segmentation(raw, decision_count=2, has_final_answer=True)

    def test_segmentation_rejects_generic_loop_contract(self):
        raw = {
            "trajectory_summary": "generic placeholder",
            "loops": [
                loop_record(
                    1,
                    0,
                    0,
                    "READY_TO_ANSWER",
                    subgoal="Establish information dependency 1 required by the user question",
                )
            ],
        }
        with self.assertRaisesRegex(ValueError, "specific information objective"):
            MODULE.validate_segmentation(raw, decision_count=1, has_final_answer=True)

    def test_segmentation_derives_completion_from_specific_subgoal(self):
        raw = {
            "trajectory_summary": "specific objective with a weak completion placeholder",
            "loops": [
                loop_record(
                    1,
                    0,
                    0,
                    "READY_TO_ANSWER",
                    subgoal="Identify the requested founding year",
                )
            ],
        }
        raw["loops"][0]["completion_test"] = "Evidence resolves information dependency 1"
        segmented = MODULE.validate_segmentation(raw, decision_count=1, has_final_answer=True)
        self.assertEqual(
            segmented["loops"][0]["completion_test"],
            "Evidence is sufficient to identify the requested founding year",
        )

    def test_segmentation_rejects_same_claim_verification_loop(self):
        raw = {
            "trajectory_summary": "find the date and then recheck the same date",
            "loops": [
                loop_record(1, 0, 0, "SWITCH_LOOP", subgoal="Identify the requested death date"),
                loop_record(
                    2,
                    1,
                    1,
                    "READY_TO_ANSWER",
                    subgoal="Verify the requested death date with another reference",
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "same-claim verification"):
            MODULE.validate_segmentation(raw, decision_count=2, has_final_answer=True)

    def test_segmentation_allows_verification_of_a_distinct_relation(self):
        raw = {
            "trajectory_summary": "identify the officeholder and then establish the role relation",
            "loops": [
                loop_record(1, 0, 0, "SWITCH_LOOP", subgoal="Identify the officeholder"),
                loop_record(
                    2,
                    1,
                    1,
                    "READY_TO_ANSWER",
                    subgoal="Confirm that the identified officeholder is the principal leader of the church",
                ),
            ],
        }
        segmented = MODULE.validate_segmentation(raw, decision_count=2, has_final_answer=True)
        self.assertEqual(len(segmented["loops"]), 2)

    def test_segmentation_normalizes_action_dependent_enums(self):
        raw = {
            "trajectory_summary": "answer becomes available immediately",
            "loops": [
                loop_record(
                    1,
                    0,
                    0,
                    "READY_TO_ANSWER",
                    subgoal="Establish the requested date",
                    outcome="IN_PROGRESS",
                    basis="NONE",
                )
            ],
        }
        segmented = MODULE.validate_segmentation(raw, decision_count=1)
        item = segmented["loops"][0]
        self.assertEqual(item["outcome"], "RESOLVED")
        self.assertEqual(item["boundary_basis"], "TASK_COMPLETE")

    def test_segmentation_scrubs_future_literals(self):
        raw = {
            "trajectory_summary": "locate the article and then verify its answer",
            "loops": [
                loop_record(1, 0, 0, "SWITCH_LOOP", subgoal="Identify the target article"),
                loop_record(
                    2,
                    1,
                    1,
                    "READY_TO_ANSWER",
                    subgoal="Extract the requested survey months from the identified article",
                ),
            ],
        }
        raw["loops"][1]["completion_test"] = (
            "The 'May and June 2020' text is directly supported by evidence"
        )
        segmented = MODULE.validate_segmentation(
            raw,
            decision_count=2,
            decision_message_limits=[6, 9],
            has_final_answer=True,
            question="During which months was the survey conducted?",
            trajectory_messages=[
                {"index": 6, "text": "A candidate article was found."},
                {"index": 9, "text": "The survey occurred in May and June 2020."},
            ],
        )
        contract = " ".join(
            [segmented["loops"][1]["subgoal"], segmented["loops"][1]["completion_test"]]
        )
        self.assertNotIn("May and June 2020", contract)
        self.assertNotRegex(contract.casefold(), r"\b(searching|opening)\b")

    def test_contract_sanitizer_removes_domain_names(self):
        cleaned = MODULE.sanitize_contract_text(
            "Confirm the claim with reggaeville.com",
            fallback="Evidence confirms the requested claim",
        )
        self.assertNotIn("reggaeville.com", cleaned)

    def test_contract_sanitizer_removes_source_and_doc_operations(self):
        source_goal = MODULE.sanitize_contract_text(
            "Find another source confirming that resistance increases with temperature",
            fallback="Establish how resistance changes with temperature",
        )
        doc_gap = MODULE.sanitize_contract_text(
            "Need to view Doc 4323047 to extract a direct statement",
            fallback="Evidence must establish how resistance changes with temperature",
        )
        self.assertFalse(MODULE.GLOBAL_CONCRETE_PATTERN.search(source_goal))
        self.assertFalse(MODULE.GLOBAL_CONCRETE_PATTERN.search(doc_gap))

    def test_contract_sanitizer_rewrites_reference_strategy_as_information_goal(self):
        converted = MODULE.sanitize_contract_text(
            "Find an independent reference confirming the conversion factor",
            fallback="Evidence resolves the conversion factor",
        )
        self.assertEqual(converted, "Independently verify the conversion factor")

    def test_artifact_question_gets_specific_fallback_contract(self):
        question = "According to the study, on what date did the event occur?"
        self.assertIn("specific artifact", MODULE.subgoal_fallback(question, 1))
        self.assertIn("uniquely identified", MODULE.completion_fallback(question, 1))

    def test_future_percentage_is_not_masked_by_visible_line_numbers(self):
        visible = "L15: the target article is available"
        cleaned = MODULE.scrub_future_literals(
            "Evidence verifies the exact 15% statistic",
            visible_text=visible,
        )
        self.assertNotIn("15%", cleaned)
        self.assertIn("requested value", cleaned)

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

    def test_validate_causal_raw_adds_missing_key_evidence_citation(self):
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
        target = MODULE.validate_causal_raw(
            raw,
            boundary=boundary,
            working_before=MODULE.initial_working_state(),
            loop_number=1,
            seen_message_ids={"msg_0006"},
            allowed_memory_ids=set(),
        )
        self.assertIn("msg_0006", target["working_state_after"]["key_evidence"][0])

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

    def test_continue_information_gain_does_not_increase_inside_loop(self):
        operations = {
            name: ([] if name != "completed_subgoal" else "")
            for name in MODULE.DELTA_FIELDS
        }
        raw = {
            "decision_reason": "same objective",
            "progress": {
                "progress_summary": "Additional evidence arrived in msg_0009.",
                "resolved_aspects": [],
                "open_aspects": ["Confirmation remains."],
                "key_evidence": ["A candidate is supported by msg_0009."],
                "candidate_answer": "",
                "active_hypotheses": [],
                "failed_strategies": [],
                "evidence_gaps": ["Confirmation remains."],
                "answer_stable": False,
                "evidence_sufficient": False,
                "expected_information_gain": "HIGH",
            },
            "durable_update": operations,
            "loop_memory": None,
            "retrieval": {"query": "claim", "relevant_memory_ids": [], "reason": "none"},
        }
        boundary = {
            "action": "CONTINUE_CURRENT_LOOP",
            "reason": "",
            "current_subgoal": "Establish the requested claim",
            "current_completion_test": "Evidence confirms the requested claim",
            "next_subgoal": "",
            "next_completion_test": "",
            "outcome": "IN_PROGRESS",
            "boundary_basis": "NONE",
            "confidence": 1.0,
            "progress": {},
        }
        before = MODULE.initial_working_state()
        before["expected_information_gain"] = "MEDIUM"
        target = MODULE.validate_causal_raw(
            raw,
            boundary=boundary,
            working_before=before,
            loop_number=1,
            seen_message_ids={"msg_0009"},
            allowed_memory_ids=set(),
        )
        self.assertEqual(target["working_state_after"]["expected_information_gain"], "MEDIUM")


if __name__ == "__main__":
    unittest.main()
