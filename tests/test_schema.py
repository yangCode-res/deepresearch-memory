import json
import unittest

from cl_gism.schema import (
    DeltaOperation,
    GlobalIntentState,
    LoopMemory,
    MemoryStatus,
    RawMemory,
    SchemaValidationError,
    SourceType,
    StateDelta,
    StateDeltaOperation,
    StateItem,
    StateItemKind,
    TaskAnchor,
)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_id = "task_research_001"
        self.raw = RawMemory(
            task_id=self.task_id,
            source_type=SourceType.USER,
            content="设计一个跨 Loop 的 Memory 系统",
        )
        self.anchor = TaskAnchor(
            task_id=self.task_id,
            original_goal="设计一个跨 Loop 的全局意图状态 Memory 系统",
            success_criteria=["能够追踪状态变化", "能够召回原始证据"],
            immutable_constraints=["保留原始证据"],
            evidence_ids=[self.raw.raw_id],
        )

    def test_four_layers_are_json_serializable(self) -> None:
        goal = StateItem(
            kind=StateItemKind.CURRENT_GOAL,
            value="实现第一版 Schema",
            status=MemoryStatus.CONFIRMED,
            confidence=0.95,
            source_type=SourceType.USER,
            evidence_ids=[self.raw.raw_id],
            user_confirmed=True,
        )
        state = GlobalIntentState(
            task_id=self.task_id,
            current_goal=goal,
        )
        loop = LoopMemory(
            task_id=self.task_id,
            subgoal="定义四层 Memory 数据结构",
            conclusion="Schema 已完成",
            status=MemoryStatus.RESOLVED,
            ended_at="2026-07-24T00:00:00Z",
            evidence_ids=[self.raw.raw_id],
        )
        for record in (self.raw, self.anchor, state, loop):
            payload = record.to_dict()
            json.dumps(payload, ensure_ascii=False)
            self.assertIsInstance(payload, dict)

    def test_state_item_requires_user_source_when_confirmed_by_user(self) -> None:
        item = StateItem(
            kind=StateItemKind.SOFT_PREFERENCE,
            value="优先可追溯",
            user_confirmed=True,
            source_type=SourceType.AGENT,
        )
        with self.assertRaises(SchemaValidationError):
            item.validate()

    def test_state_groups_have_consistent_kinds(self) -> None:
        wrong = StateItem(kind=StateItemKind.OPEN_QUESTION, value="x")
        state = GlobalIntentState(task_id=self.task_id, current_goal=wrong)
        with self.assertRaises(SchemaValidationError):
            state.validate()

    def test_delta_versions_are_monotonic_and_auditable(self) -> None:
        loop = LoopMemory(task_id=self.task_id, subgoal="定义 Schema")
        operation = StateDeltaOperation(
            operation=DeltaOperation.ADD,
            target="open_questions",
            value="如何实现 Loop 边界检测？",
            reason="第一步暂不实现边界模型",
            evidence_ids=[self.raw.raw_id, loop.loop_id],
            loop_id=loop.loop_id,
        )
        delta = StateDelta(
            task_id=self.task_id,
            from_state_version=1,
            to_state_version=2,
            operations=[operation],
            generated_from_loop_id=loop.loop_id,
        )
        payload = delta.to_dict()
        self.assertEqual(payload["operations"][0]["operation"], "ADD")

        invalid = StateDelta(task_id=self.task_id, from_state_version=2, to_state_version=2)
        with self.assertRaises(SchemaValidationError):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
