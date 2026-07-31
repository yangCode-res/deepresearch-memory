import unittest

from cl_gism import (
    LLMLoopBoundaryJudge,
    LLMStateUpdater,
    OnlineMemorySession,
    UnifiedControlDecision,
)


class FakeController:
    model = "fake-controller"

    def complete_json(self, system_prompt, user_prompt):
        if "segment deep-research trajectories" in system_prompt:
            return {
                "split": True,
                "reason": "the assistant starts a new evidence thread",
                "confidence": 0.9,
                "current_loop_subgoal": "Find the first source",
                "next_loop_subgoal": "Confirm with a second source",
            }
        return {
            "summary": "retain the first source as a hypothesis",
            "operations": [
                {
                    "operation": "ADD",
                    "target": "working_hypotheses",
                    "value": "The first source suggests the answer is Alpha.",
                    "reason": "The completed loop found a candidate answer.",
                    "evidence_ids": [],
                    "item": {
                        "status": "tentative",
                        "confidence": 0.7,
                        "source_type": "agent",
                        "valid_time": None,
                        "contradicts": [],
                        "supersedes": [],
                        "user_confirmed": False,
                    },
                }
            ],
        }


class OnlineMemoryTests(unittest.TestCase):
    def test_material_progress_ignores_narrative_lead_churn(self):
        first = {
            "open_aspects": ["identify the youth brand"],
            "resolved_aspects": [],
            "key_evidence": [],
            "promising_leads": [{
                "kind": "ENTITY",
                "entity": "Vitali Hakko",
                "source": "result 3",
                "reason": "matches the hat clue",
                "status": "ACTIVE",
                "confidence": 0.8,
            }],
        }
        rewritten = {
            **first,
            "open_aspects": ["find the exact brand name"],
            "promising_leads": [{
                **first["promising_leads"][0],
                "source": "result 7",
                "reason": "different narrative wording",
                "confidence": 0.9,
            }],
        }
        genuinely_new = {
            **rewritten,
            "promising_leads": [
                *rewritten["promising_leads"],
                {
                    "kind": "ENTITY",
                    "entity": "Cem Hakko",
                    "source": "result 8",
                    "status": "ACTIVE",
                },
            ],
        }

        self.assertEqual(
            OnlineMemorySession._progress_signature(first),
            OnlineMemorySession._progress_signature(rewritten),
        )
        self.assertNotEqual(
            OnlineMemorySession._progress_signature(rewritten),
            OnlineMemorySession._progress_signature(genuinely_new),
        )

    def test_cross_loop_prompt_keeps_current_loop_and_retrieves_old_loop(self):
        controller = FakeController()
        session = OnlineMemorySession(
            qid=42,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(controller),
            state_updater=LLMStateUpdater(controller),
            top_k=3,
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
        ]
        first_prompt = session.build_prompt(messages)
        self.assertEqual(len(first_prompt), 2)

        messages.extend(
            [
                {"role": "assistant", "content": "Search for Alpha."},
                {"role": "tool", "content": "The first source says Alpha."},
            ]
        )
        session.build_prompt(messages)

        messages.extend(
            [
                {"role": "assistant", "content": "Now confirm Alpha independently."},
                {"role": "tool", "content": "A second source also mentions Alpha."},
            ]
        )
        compact = session.build_prompt(messages)

        self.assertEqual(len(session.completed_loops), 1)
        self.assertEqual(session.state.state_version, 2)
        self.assertEqual(len(compact), 4)
        self.assertEqual(compact[-2]["content"], "Now confirm Alpha independently.")
        self.assertTrue(session.traces[-1].loop_switched)
        self.assertGreater(len(session.traces[-1].retrieved_memory_ids), 0)
        self.assertIn("cross_loop_memory", compact[0]["content"])

    def test_unified_controller_ready_signal_forces_answer_prompt(self):
        class ReadyController:
            def decide(self, **kwargs):
                return UnifiedControlDecision(
                    task_status="READY_TO_ANSWER",
                    research_phase="ANSWER_SYNTHESIS",
                    switch_loop=False,
                    reason="answer and citations are complete",
                    confidence=0.99,
                    current_loop_subgoal="verify Alpha",
                    loop_outcome="RESOLVED",
                    boundary_basis="TASK_COMPLETE",
                    loop_progress={
                        "completion_test": "identify Alpha",
                        "progress_summary": "Alpha is fully verified",
                        "resolved_aspects": ["answer identity"],
                        "open_aspects": [],
                        "key_evidence": ["Alpha is supported 【1†L1-L3】"],
                        "candidate_answer": "Alpha",
                        "answer_stable": True,
                        "evidence_sufficient": True,
                        "confidence": 0.99,
                        "expected_information_gain": "LOW",
                    },
                    state_delta={"mode": "NOOP", "summary": "", "operations": []},
                    retrieval_query="Alpha final evidence",
                )

        session = OnlineMemorySession(
            qid=43,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(FakeController()),
            state_updater=LLMStateUpdater(FakeController()),
            unified_controller=ReadyController(),
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
        ]
        initial = session.build_prompt(messages)
        self.assertNotIn("cross_loop_memory", initial[0]["content"])

        messages.extend([
            {"role": "assistant", "content": "Alpha is supported by 【1†L1-L3】."},
            {"role": "tool", "content": "[1] Alpha evidence\nL1: Alpha"},
        ])
        final_prompt = session.build_prompt(messages)
        self.assertEqual(session.traces[-1].task_status, "READY_TO_ANSWER")
        self.assertIn("Do not call tools", final_prompt[0]["content"])
        self.assertIn("Alpha is supported", final_prompt[-2]["content"])
        self.assertIn("loop_working_state", final_prompt[0]["content"])
        messages.append({"role": "assistant", "content": "Exact Answer: Alpha\nConfidence: 99%"})
        session.finalize(messages)
        self.assertEqual(session.state.state_version, 1)
        self.assertEqual(session.completed_loops[-1].subgoal, "verify Alpha")
        self.assertIn("Exact Answer: Alpha", session.completed_loops[-1].conclusion)

    def test_unified_boundary_archives_the_turn_that_completed_the_work_unit(self):
        class WorkUnitController:
            def __init__(self):
                self.calls = 0

            def decide(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return UnifiedControlDecision(
                        task_status="CONTINUE",
                        research_phase="CANDIDATE_VERIFICATION",
                        switch_loop=False,
                        current_loop_subgoal="diagnose network latency",
                        loop_outcome="IN_PROGRESS",
                        boundary_basis="NONE",
                        state_delta={"mode": "NOOP", "summary": "", "operations": []},
                    )
                return UnifiedControlDecision(
                    task_status="SWITCH_LOOP",
                    research_phase="CANDIDATE_VERIFICATION",
                    switch_loop=True,
                    reason="network hypothesis resolved; inspect a separate dependency",
                    confidence=0.9,
                    current_loop_subgoal="diagnose network latency",
                    next_loop_subgoal="diagnose database latency",
                    loop_outcome="RESOLVED",
                    boundary_basis="SUBGOAL_COMPLETED",
                    loop_progress={
                        "completion_test": "rule out network latency",
                        "progress_summary": "network is not the bottleneck",
                        "resolved_aspects": ["network latency"],
                        "open_aspects": [],
                        "key_evidence": [],
                        "candidate_answer": "",
                        "answer_stable": False,
                        "evidence_sufficient": False,
                        "confidence": 0.8,
                        "expected_information_gain": "HIGH",
                        "tried_strategies": [{
                            "strategy": "network measurements",
                            "outcome": "normal",
                            "evidence_gain": "HIGH",
                        }],
                        "rejected_hypotheses": ["network bottleneck"],
                        "promising_leads": [{
                            "kind": "ENTITY",
                            "entity": "database latency",
                            "source": "dependency trace",
                            "reason": "the remaining slow component",
                            "status": "ACTIVE",
                            "confidence": 0.9,
                        }],
                        "prioritized_open_aspects": [{
                            "aspect": "database latency",
                            "priority": "ANSWER_CRITICAL",
                            "status": "open",
                            "best_next_action": "inspect database trace",
                        }],
                        "research_direction": {
                            "objective": "diagnose database latency",
                            "must_investigate": ["database dependency trace"],
                            "rationale": "network has been ruled out",
                            "stop_condition": "identify or rule out the database bottleneck",
                        },
                        "avoid": ["repeat network measurements"],
                    },
                    state_delta={
                        "mode": "APPLY",
                        "summary": "network is not the bottleneck",
                        "operations": [{
                            "operation": "ADD",
                            "target": "resolved_findings",
                            "value": "Network latency is within the expected range.",
                            "reason": "the network checks passed",
                            "evidence_ids": [],
                            "target_item_ids": [],
                        }],
                    },
                )

        session = OnlineMemorySession(
            qid=45,
            question="Why is the service slow?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(FakeController()),
            state_updater=LLMStateUpdater(FakeController()),
            unified_controller=WorkUnitController(),
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "Why is the service slow?"},
            {"role": "assistant", "content": "Measure network latency."},
            {"role": "tool", "content": "Network latency is normal."},
        ]
        session.build_prompt(messages)
        messages.extend([
            {"role": "assistant", "content": "The network hypothesis is resolved."},
            {"role": "tool", "content": "Packet loss is zero."},
        ])
        compact = session.build_prompt(messages)

        self.assertEqual(len(session.completed_loops), 1)
        archived_text = str(session.completed_loops[0].to_dict())
        self.assertIn("Packet loss is zero", archived_text)
        self.assertEqual(session.current_events, [])
        self.assertEqual(len(compact), 2)
        self.assertEqual(session.traces[-1].current_loop_subgoal, "diagnose database latency")
        self.assertEqual(session.traces[-1].research_phase, "CANDIDATE_VERIFICATION")
        instruction = session._research_instruction()
        self.assertIn("Must investigate before broad exploration: database dependency trace", instruction)
        self.assertIn("database latency", instruction)
        self.assertIn("Do not repeat: repeat network measurements", instruction)
        self.assertIn("choose the browser tool", instruction)

    def test_invalid_controller_delta_archives_with_noop_instead_of_polluting_state(self):
        class SwitchingController:
            def __init__(self):
                self.calls = 0

            def decide(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return UnifiedControlDecision(
                        task_status="CONTINUE",
                        research_phase="DISCOVERY",
                        switch_loop=False,
                        state_delta={"mode": "NOOP", "summary": "", "operations": []},
                    )
                return UnifiedControlDecision(
                    task_status="SWITCH_LOOP",
                    research_phase="CANDIDATE_VERIFICATION",
                    switch_loop=True,
                    reason="genuinely new candidate",
                    confidence=0.9,
                    current_loop_subgoal="find candidate",
                    next_loop_subgoal="verify candidate",
                    state_delta={
                        "mode": "APPLY",
                        "summary": "invalid nested item",
                        "operations": [{
                            "operation": "ADD",
                            "target": "working_hypotheses",
                            "value": "Alpha",
                            "reason": "candidate found",
                            "item": {"status": "not-a-status"},
                        }],
                    },
                )

        session = OnlineMemorySession(
            qid=44,
            question="What is the answer?",
            system_prompt="research",
            boundary_judge=LLMLoopBoundaryJudge(FakeController()),
            state_updater=LLMStateUpdater(FakeController()),
            unified_controller=SwitchingController(),
        )
        messages = [
            {"role": "system", "content": "research"},
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "Search Alpha."},
            {"role": "tool", "content": "Alpha result."},
        ]
        session.build_prompt(messages)
        messages.extend([
            {"role": "assistant", "content": "Now verify Beta."},
            {"role": "tool", "content": "Beta result."},
        ])
        session.build_prompt(messages)

        self.assertEqual(len(session.completed_loops), 1)
        self.assertEqual(session.state.state_version, 1)
        self.assertEqual(session.state.working_hypotheses, [])
        self.assertIn("not-a-status", session.traces[-1].controller_error)
        self.assertEqual(session.traces[-1].research_phase, "CANDIDATE_VERIFICATION")
        self.assertEqual(session.traces[-1].retrieved_memory_ids[0], session.completed_loops[0].loop_id)


if __name__ == "__main__":
    unittest.main()
