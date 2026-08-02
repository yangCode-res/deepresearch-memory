from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = spec_from_file_location(
    "build_researcher_memory_conditioned",
    SCRIPTS / "build_researcher_memory_conditioned.py",
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResearcherMemoryConditionedTest(unittest.TestCase):
    def test_retrieval_uses_only_relevant_available_memory(self):
        memories = [
            {
                "memory_id": "memory_loop_001",
                "summary": "Tim Tebow hosts Million Dollar Mile.",
                "durable_findings": ["Tim Tebow is the host."],
                "unresolved_questions": ["Where did Tim Tebow graduate?"],
            },
            {
                "memory_id": "memory_loop_999",
                "summary": "An unrelated oil-market statistic.",
                "durable_findings": [],
                "unresolved_questions": [],
            },
        ]
        selected, audit = MODULE.select_causal_memories(
            memories,
            {
                "current_subgoal": "Determine where Tim Tebow graduated from",
                "completion_test": "Evidence identifies his university",
                "open_aspects": [],
                "evidence_gaps": [],
            },
        )
        self.assertEqual([item["memory_id"] for item in selected], ["memory_loop_001"])
        self.assertEqual(len(audit), 2)

    def test_action_audit_rewrites_exact_duplicate_tool_call(self):
        message = {
            "message_id": "msg_0002",
            "role": "assistant",
            "recipient": "browser.search",
            "text": '{"query": "Tim Tebow university"}',
        }
        call, error = MODULE.parse_tool_call([message])
        result = MODULE.audit_tool_action(
            assistant_messages=[message],
            tool_call=call,
            parse_error=error,
            loop_history=[{**message, "message_id": "msg_0001"}],
            selected_memories=[],
            working_state={"current_subgoal": "Find Tim Tebow's university"},
        )
        self.assertEqual(result["decision"], "REWRITE")
        self.assertTrue(result["exact_duplicate_tool_call_in_loop"])

    def test_action_audit_keeps_evidence_search_that_overlaps_memory(self):
        message = {
            "message_id": "msg_0010",
            "role": "assistant",
            "recipient": "browser.search",
            "text": '{"query": "Million Dollar Mile Tim Tebow host"}',
        }
        call, error = MODULE.parse_tool_call([message])
        result = MODULE.audit_tool_action(
            assistant_messages=[message],
            tool_call=call,
            parse_error=error,
            loop_history=[],
            selected_memories=[
                {
                    "memory_id": "memory_loop_001",
                    "summary": "Tim Tebow is the host of Million Dollar Mile.",
                    "durable_findings": [],
                    "unresolved_questions": ["Where did he graduate?"],
                }
            ],
            working_state={
                "current_subgoal": "Determine where Tim Tebow graduated from",
                "completion_test": "Evidence identifies his university",
            },
        )
        self.assertEqual(result["decision"], "KEEP")
        self.assertTrue(result["repeats_selected_memory_fact"])
        self.assertTrue(result["review_warnings"])

    def test_tool_call_requires_valid_json_arguments(self):
        call, error = MODULE.parse_tool_call(
            [
                {
                    "message_id": "msg_0001",
                    "role": "assistant",
                    "recipient": "browser.open",
                    "text": "not-json",
                }
            ]
        )
        self.assertIsNone(call)
        self.assertIn("valid JSON", error)

    def test_finalize_keeps_candidate_and_training_ids_aligned(self):
        candidates = [
            {
                "sample_id": "research_000001",
                "quality_gate": {"decision": "KEEP"},
            },
            {
                "sample_id": "research_000002",
                "quality_gate": {"decision": "REWRITE"},
            },
        ]
        alignment = [
            {"researcher_sample_id": "research_000001"},
            {"researcher_sample_id": "research_000002"},
        ]
        training = MODULE.finalize_training_rows(candidates, alignment)
        self.assertEqual(candidates[0]["sample_id"], "research_000001")
        self.assertEqual(training[0]["sample_id"], "research_train_000001")
        self.assertEqual(
            training[0]["lineage"]["candidate_sample_id"], "research_000001"
        )
        self.assertEqual(alignment[0]["training_sample_id"], "research_train_000001")
        self.assertIsNone(alignment[1]["training_sample_id"])

    def test_raw_match_disambiguates_same_qid_trajectory(self):
        replay = [
            {
                "input": {
                    "observed_messages": [
                        {
                            "message_id": "msg_0002",
                            "index": 2,
                            "role": "assistant",
                            "name": None,
                            "recipient": "browser.search",
                            "channel": "analysis",
                            "text": '{"query":"right trajectory"}',
                            "truncated": False,
                        }
                    ]
                }
            }
        ]
        matching = {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "recipient": "browser.search",
                    "channel": "analysis",
                    "content": [{"text": '{"query":"right trajectory"}'}],
                },
            ]
        }
        wrong = {
            "messages": [
                matching["messages"][0],
                {**matching["messages"][1], "content": [{"text": "different"}]},
            ]
        }
        self.assertTrue(MODULE.raw_matches_replay(matching, replay))
        self.assertFalse(MODULE.raw_matches_replay(wrong, replay))

    def test_final_answer_must_follow_ready_boundary(self):
        raw = {
            "messages": [
                {"role": "assistant", "channel": "final", "content": "too early"},
                {"role": "tool", "content": "later observation"},
                {"role": "assistant", "channel": "final", "content": "real answer"},
            ]
        }
        final = MODULE.raw_final_message(raw, after_message_index=2)
        self.assertEqual(final["text"], "real answer")
        self.assertIsNone(MODULE.raw_final_message(raw, after_message_index=3))

    def test_raw_match_reproduces_retrospective_truncation(self):
        replay = [
            {
                "input": {
                    "observed_messages": [
                        {
                            "message_id": "msg_0001",
                            "index": 1,
                            "role": "tool",
                            "name": None,
                            "recipient": None,
                            "channel": "analysis",
                            "text": "abcd",
                            "truncated": True,
                        }
                    ]
                }
            }
        ]
        raw = {
            "messages": [
                {"role": "tool", "channel": "analysis", "content": "abcdefgh"}
            ]
        }
        self.assertTrue(MODULE.raw_matches_replay(raw, replay))


if __name__ == "__main__":
    unittest.main()
