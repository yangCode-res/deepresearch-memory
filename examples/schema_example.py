"""Create a minimal four-layer CL-GISM record set and print JSON."""

from cl_gism import (
    GlobalIntentState,
    MemoryStatus,
    RawMemory,
    SourceType,
    StateItem,
    StateItemKind,
    TaskAnchor,
)


def main() -> None:
    raw = RawMemory(
        task_id="task_demo_001",
        source_type=SourceType.USER,
        content="我想设计一个可以跨 Loop 维护全局意图的 Memory 系统。",
    )
    anchor = TaskAnchor(
        task_id=raw.task_id,
        original_goal="设计跨 Loop 全局意图状态 Memory 系统",
        success_criteria=["状态可持续更新", "结论可追溯到原始证据"],
        immutable_constraints=["原始证据不可覆盖"],
        evidence_ids=[raw.raw_id],
    )
    goal = StateItem(
        kind=StateItemKind.CURRENT_GOAL,
        value="完成第一版四层 Memory Schema",
        status=MemoryStatus.ACTIVE,
        confidence=0.9,
        source_type=SourceType.AGENT,
        evidence_ids=[raw.raw_id],
    )
    state = GlobalIntentState(task_id=raw.task_id, current_goal=goal)
    for record in (raw, anchor, state):
        print(record.to_json())


if __name__ == "__main__":
    main()
