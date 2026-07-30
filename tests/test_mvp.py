import json
from pathlib import Path
import unittest

from cl_gism import CLGISMEngine, HeuristicStateUpdater, RuleBasedLoopBuilder, parse_openresearcher_row


class MvpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "data/openresearcher-dataset/example_qid_39_short_raw.json"
        cls.row = json.loads(path.read_text())

    def test_parser_preserves_all_messages(self) -> None:
        parsed = parse_openresearcher_row(self.row)
        self.assertEqual(len(parsed.events), len(self.row["messages"]))
        self.assertEqual(parsed.anchor.original_goal, self.row["question"])
        self.assertTrue(parsed.events[0].raw_memory.raw_id.startswith("raw_"))

    def test_engine_builds_loops_state_and_context(self) -> None:
        engine = CLGISMEngine(loop_builder=RuleBasedLoopBuilder(), state_updater=HeuristicStateUpdater())
        task = engine.ingest_openresearcher_row(self.row)
        self.assertGreaterEqual(len(task.loops), 2)
        self.assertEqual(len(task.raw_memories), len(self.row["messages"]))
        self.assertGreater(task.state.state_version, 1)
        context = engine.build_llm_context(task.task_id, "release date", top_k=3)
        self.assertIn("task_anchor", context)
        self.assertIn("global_state", context)
        self.assertLessEqual(len(context["retrieved_memories"]), 3)


if __name__ == "__main__":
    unittest.main()
