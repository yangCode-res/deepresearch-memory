# CL-GISM

Cross-Loop Global Intent-State Memory 的第一步实现：定义四层 Memory Schema，并为后续状态更新保留版本化的 State Delta 接口。

## 当前内容

- `TaskAnchor`：稳定的原始任务目标、成功标准和不可变约束。
- `GlobalIntentState`：当前任务状态及其带类型、置信度、来源和证据指针的 `StateItem`。
- `LoopMemory`：围绕一个局部目标形成的阶段性过程记录。
- `RawMemory`：不可覆盖的原始对话、文档、工具结果或实验记录。
- `StateDelta`：可审计的状态变更批次，为下一步 Loop/State 更新流程提供接口。

所有记录都提供 `validate()`、`to_dict()` 和 `to_json()`，ID 自动带有层级前缀，例如 `raw_*`、`loop_*`、`state_*`。

## 运行测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 运行示例

```bash
PYTHONPATH=src python examples/schema_example.py
```
