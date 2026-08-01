import unittest
import json

from cl_gism import (
    HeuristicStateUpdater,
    MemoryHit,
    TaskAnchor,
    UnifiedMemoryController,
)
from cl_gism.unified_controller import _normalize_loop_progress


class FakeClient:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "task_status": "SWITCH_LOOP",
            "research_phase": "CANDIDATE_VERIFICATION",
            "loop": {
                "switch": True,
                "reason": "new verification phase",
                "confidence": 0.94,
                "current_loop_subgoal": "find candidate",
                "next_loop_subgoal": "verify candidate",
                "outcome": "RESOLVED",
                "boundary_basis": "PHASE_TRANSITION",
            },
            "state_delta": {
                "mode": "APPLY",
                "summary": "retain candidate",
                "operations": [{
                    "operation": "ADD",
                    "target": "working_hypotheses",
                    "value": "Alpha is the candidate",
                    "reason": "discovery identified a named candidate",
                    "evidence_ids": [],
                    "target_item_ids": [],
                }],
            },
            "retrieval": {
                "query": "verify Alpha",
                "selected_memory_ids": ["loop_allowed"],
                "reason": "candidate evidence is useful",
            },
        }


class UnifiedControllerTests(unittest.TestCase):
    def test_unsupported_lead_expires_and_action_instruction_is_dropped(self):
        progress = _normalize_loop_progress(
            {
                "promising_leads": [
                    {
                        "kind": "ENTITY",
                        "entity": "Civic Tower",
                        "source": "result 1",
                        "evidence": "a tower was mentioned",
                        "status": "ACTIVE",
                        "confidence": 0.8,
                    },
                    {
                        "kind": "ENTITY",
                        "entity": "Search for European tower brands",
                        "source": "assistant plan",
                        "status": "ACTIVE",
                        "confidence": 0.9,
                    },
                ]
            },
            None,
        )
        self.assertEqual([lead["entity"] for lead in progress["promising_leads"]], ["Civic Tower"])

        for _ in range(3):
            progress = _normalize_loop_progress({}, progress)

        self.assertEqual(progress["promising_leads"][0]["status"], "REJECTED")
        self.assertLessEqual(progress["promising_leads"][0]["confidence"], 0.2)

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
        self.assertEqual(decision.research_phase, "CANDIDATE_VERIFICATION")
        self.assertEqual(decision.retrieval_query, "verify Alpha")
        self.assertEqual(decision.selected_memory_ids, ["loop_allowed"])

    def test_ready_to_answer_is_terminal_without_loop_switch(self):
        class ReadyClient:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "task_status": "READY_TO_ANSWER",
                    "research_phase": "ANSWER_SYNTHESIS",
                    "loop": {
                        "switch": False,
                        "reason": "answer and citations are complete",
                        "confidence": 0.98,
                        "current_loop_subgoal": "verify Alpha",
                        "next_loop_subgoal": "",
                        "outcome": "RESOLVED",
                        "boundary_basis": "TASK_COMPLETE",
                        "progress": {
                            "completion_test": "identify Alpha with citable evidence",
                            "progress_summary": "Alpha is identified and supported",
                            "resolved_aspects": ["identity", "supporting source"],
                            "open_aspects": [],
                            "key_evidence": ["Alpha is confirmed 【1†L1-L3】"],
                            "candidate_answer": "Alpha",
                            "answer_stable": True,
                            "evidence_sufficient": True,
                            "confidence": 0.98,
                            "expected_information_gain": "LOW",
                        },
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
                    "research_phase": "CANDIDATE_VERIFICATION",
                    "loop": {
                        "switch": True,
                        "reason": "new phase",
                        "confidence": 0.9,
                        "current_loop_subgoal": "search",
                        "next_loop_subgoal": "verify",
                        "outcome": "RESOLVED",
                        "boundary_basis": "SUBGOAL_COMPLETED",
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

    def test_same_phase_allows_a_new_independently_decidable_subgoal(self):
        class SamePhaseSwitchClient:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "task_status": "SWITCH_LOOP",
                    "research_phase": "CANDIDATE_VERIFICATION",
                    "loop": {
                        "switch": True,
                        "reason": "the first claim is resolved; verify a separate claim next",
                        "confidence": 0.9,
                        "current_loop_subgoal": "verify Alpha's publication",
                        "next_loop_subgoal": "verify Alpha's appointment",
                        "outcome": "RESOLVED",
                        "boundary_basis": "SUBGOAL_COMPLETED",
                    },
                    "state_delta": {
                        "mode": "APPLY",
                        "summary": "publication verified",
                        "operations": [{
                            "operation": "ADD",
                            "target": "resolved_findings",
                            "value": "Alpha's publication is verified.",
                            "reason": "the current work unit met its completion test",
                            "evidence_ids": [],
                            "target_item_ids": [],
                        }],
                    },
                    "retrieval": {
                        "query": "Alpha appointment",
                        "selected_memory_ids": [],
                        "reason": "start the next verification unit",
                    },
                }

        anchor = TaskAnchor(task_id="task_phase", original_goal="Find Alpha")
        state = HeuristicStateUpdater().initialize(anchor)
        decision = UnifiedMemoryController(SamePhaseSwitchClient()).decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
            current_phase="CANDIDATE_VERIFICATION",
            current_loop_subgoal="verify Alpha's publication",
        )
        self.assertTrue(decision.switch_loop)
        self.assertEqual(decision.research_phase, "CANDIDATE_VERIFICATION")
        self.assertEqual(decision.boundary_basis, "SUBGOAL_COMPLETED")

    def test_rejects_switch_while_current_work_unit_is_in_progress(self):
        class InvalidBoundaryClient:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "task_status": "SWITCH_LOOP",
                    "research_phase": "CANDIDATE_VERIFICATION",
                    "loop": {
                        "switch": True,
                        "reason": "incorrect premature switch",
                        "confidence": 0.8,
                        "current_loop_subgoal": "verify publication",
                        "next_loop_subgoal": "verify appointment",
                        "outcome": "IN_PROGRESS",
                        "boundary_basis": "SUBGOAL_CHANGED",
                    },
                    "state_delta": {
                        "mode": "APPLY",
                        "summary": "premature",
                        "operations": [{
                            "operation": "ADD",
                            "target": "uncertainties",
                            "value": "Publication verification is incomplete.",
                            "reason": "the claim is not resolved",
                            "evidence_ids": [],
                            "target_item_ids": [],
                        }],
                    },
                    "retrieval": {"query": "next", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="task_boundary", original_goal="Find Alpha")
        state = HeuristicStateUpdater().initialize(anchor)
        with self.assertRaisesRegex(ValueError, "terminal outcome"):
            UnifiedMemoryController(InvalidBoundaryClient()).decide(
                anchor=anchor,
                state=state,
                current_loop=[],
                latest_events=[],
                candidates=[],
                current_phase="CANDIDATE_VERIFICATION",
            )

    def test_rejects_switch_to_the_same_subgoal(self):
        class SameSubgoalClient:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "task_status": "SWITCH_LOOP",
                    "research_phase": "DISCOVERY",
                    "loop": {
                        "switch": True,
                        "reason": "fake boundary",
                        "confidence": 0.8,
                        "current_loop_subgoal": "identify the city",
                        "next_loop_subgoal": " identify   the CITY ",
                        "outcome": "BLOCKED",
                        "boundary_basis": "BLOCKED_OR_SATURATED",
                    },
                    "state_delta": {
                        "mode": "APPLY",
                        "summary": "blocked",
                        "operations": [{
                            "operation": "ADD",
                            "target": "uncertainties",
                            "value": "City remains unknown.",
                            "reason": "search saturated",
                        }],
                    },
                    "retrieval": {"query": "city", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="same_subgoal", original_goal="Find the city")
        state = HeuristicStateUpdater().initialize(anchor)
        with self.assertRaisesRegex(ValueError, "genuinely different"):
            UnifiedMemoryController(SameSubgoalClient()).decide(
                anchor=anchor,
                state=state,
                current_loop=[],
                latest_events=[],
                candidates=[],
            )

    def test_prompt_uses_a_domain_general_work_unit_contract(self):
        class RecordingClient:
            def __init__(self):
                self.system_prompt = ""
                self.payload = {}

            def complete_json(self, system_prompt, user_prompt):
                self.system_prompt = system_prompt
                self.payload = json.loads(user_prompt)
                return {
                    "task_status": "CONTINUE",
                    "research_phase": "DISCOVERY",
                    "loop": {
                        "switch": False,
                        "reason": "the same diagnostic hypothesis is still being tested",
                        "confidence": 0.8,
                        "current_loop_subgoal": "test whether cache misses cause latency",
                        "next_loop_subgoal": "",
                        "outcome": "IN_PROGRESS",
                        "boundary_basis": "NONE",
                    },
                    "state_delta": {"mode": "NOOP", "summary": "", "operations": []},
                    "retrieval": {"query": "cache miss latency", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="task_diagnosis", original_goal="Diagnose API latency")
        state = HeuristicStateUpdater().initialize(anchor)
        client = RecordingClient()
        UnifiedMemoryController(client).decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
            current_phase="DISCOVERY",
            current_loop_subgoal="test whether cache misses cause latency",
        )

        self.assertIn("locally decidable subgoal", client.system_prompt)
        self.assertIn("diagnosing one cause", client.system_prompt)
        self.assertIn("orthogonal macro label", client.system_prompt)
        self.assertEqual(
            client.payload["committed_current_loop_subgoal"],
            "test whether cache misses cause latency",
        )

    def test_system_overrides_drift_from_the_committed_current_subgoal(self):
        class DriftingClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_prompt):
                self.calls += 1
                return {
                    "task_status": "CONTINUE",
                    "research_phase": "DISCOVERY",
                    "loop": {
                        "switch": False,
                        "reason": "incorrectly regressed to an older search thread",
                        "confidence": 0.7,
                        "current_loop_subgoal": "identify the city from its tower",
                        "next_loop_subgoal": "",
                        "outcome": "IN_PROGRESS",
                        "boundary_basis": "NONE",
                    },
                    "state_delta": {"mode": "NOOP", "summary": "", "operations": []},
                    "retrieval": {"query": "city tower", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="task_commitment", original_goal="Identify the brand")
        state = HeuristicStateUpdater().initialize(anchor)
        client = DriftingClient()
        decision = UnifiedMemoryController(client).decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
            current_phase="DISCOVERY",
            current_loop_subgoal="identify the brand from its naming history",
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            decision.current_loop_subgoal,
            "identify the brand from its naming history",
        )

    def test_loop_progress_persists_beyond_the_visible_event_window(self):
        class ProgressClient:
            def __init__(self):
                self.calls = 0
                self.inputs = []

            def complete_json(self, system_prompt, user_prompt):
                self.calls += 1
                self.inputs.append(json.loads(user_prompt))
                progress = (
                    {
                        "completion_test": "identify the brand",
                        "progress_summary": "The founder and original brand are verified.",
                        "resolved_aspects": ["founder", "original brand"],
                        "open_aspects": ["youth brand name"],
                        "key_evidence": ["Vakko was founded by Vitali Hakko 【1†L3-L7】"],
                        "candidate_answer": "",
                        "answer_stable": False,
                        "evidence_sufficient": False,
                        "confidence": 0.7,
                        "expected_information_gain": "MEDIUM",
                        "tried_strategies": [
                            {
                                "strategy": "broad country guessing",
                                "outcome": "no matching candidate",
                                "evidence_gain": "NONE",
                            }
                        ],
                        "rejected_hypotheses": ["Finland"],
                        "promising_leads": [
                            {
                                "kind": "ENTITY",
                                "entity": "Vitali Hakko",
                                "source": "result 3",
                                "reason": "matches the hat and scarf clues",
                                "status": "ACTIVE",
                                "confidence": 0.87,
                            }
                        ],
                        "prioritized_open_aspects": [
                            {
                                "aspect": "exact youth brand",
                                "priority": "ANSWER_CRITICAL",
                                "status": "open",
                                "best_next_action": "open result 3",
                            }
                        ],
                        "research_direction": {
                            "objective": "verify Vitali Hakko",
                            "must_investigate": [],
                            "rationale": "the result matches multiple clues",
                            "stop_condition": "confirm the child-created youth brand",
                        },
                        "avoid": ["more Finland tower queries"],
                    }
                    if self.calls == 1
                    else {}
                )
                return {
                    "task_status": "CONTINUE",
                    "research_phase": "DISCOVERY",
                    "loop": {
                        "switch": False,
                        "reason": "the same brand-identification unit continues",
                        "confidence": 0.8,
                        "current_loop_subgoal": "identify the brand",
                        "next_loop_subgoal": "",
                        "outcome": "IN_PROGRESS",
                        "boundary_basis": "NONE",
                        "progress": progress,
                    },
                    "state_delta": {"mode": "NOOP", "summary": "", "operations": []},
                    "retrieval": {"query": "youth brand", "selected_memory_ids": [], "reason": ""},
                }

        anchor = TaskAnchor(task_id="task_progress", original_goal="Identify the brand")
        state = HeuristicStateUpdater().initialize(anchor)
        client = ProgressClient()
        controller = UnifiedMemoryController(client)
        first = controller.decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
        )
        second = controller.decide(
            anchor=anchor,
            state=state,
            current_loop=[],
            latest_events=[],
            candidates=[],
            current_loop_subgoal="identify the brand",
            loop_progress=first.loop_progress,
            loop_rounds=40,
            stagnant_rounds=2,
        )

        self.assertEqual(
            client.inputs[1]["committed_loop_progress"]["key_evidence"],
            ["Vakko was founded by Vitali Hakko 【1†L3-L7】"],
        )
        self.assertEqual(second.loop_progress["open_aspects"], ["youth brand name"])
        self.assertEqual(
            second.loop_progress["promising_leads"][0]["entity"],
            "Vitali Hakko",
        )
        self.assertNotIn("must_investigate", second.loop_progress["research_direction"])
        self.assertEqual(client.inputs[1]["loop_runtime"]["rounds_in_current_loop"], 40)


if __name__ == "__main__":
    unittest.main()
