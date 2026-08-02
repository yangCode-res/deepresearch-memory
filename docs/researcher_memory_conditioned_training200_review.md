# Memory-conditioned Researcher dataset（training200）

## 结果

基于 `working_state_retrospective_mimo25pro_training200.jsonl` 中的 200 条 Controller 训练样本，已回放其所属的 50 条完整 OpenResearcher 轨迹，并构造对应的 Researcher 数据。

- Controller 训练点：200
- 完整轨迹：50
- Researcher 候选：472
- Researcher 成品：470
- 工具动作：420
- 最终回答：50
- 含检索 Memory 的成品：314
- 因果与结构校验违规：0
- 未找到的原始轨迹：0
- 被质量门过滤的动作：2（同一 Loop 内参数完全相同的重复工具调用）

这里的 200 条 Controller 数据是 50 条轨迹中的 200 个状态更新点，不是 200 条独立轨迹。Researcher 训练需要学习状态更新点之间的实际搜索动作，因此成品数量会大于 200。

## Controller 与 Researcher 如何对应

对工具动作 `action_i`：

1. Researcher 读取 `decision_(i-1):after` 的 Global State、Working State 和当时可用的跨 Loop Memory。
2. Researcher 生成原轨迹中的 assistant 推理和一个原生 browser 工具调用。
3. 工具返回 observation。
4. 对应的 Controller `decision_i` 消费该 observation 并产生 StateDelta、LoopDecision、WorkingStateAfter，以及必要时的跨 Loop Memory。

每条 Researcher 样本都包含以下显式关联字段：

- `preceding_controller_decision_id`
- `preceding_controller_training_sample_id`
- `observation_controller_decision_id`
- `observation_controller_training_sample_id`
- `state_ref`
- `memory_snapshot_id`
- `target_assistant_message_ids`

所以后续可以直接用训练样本 ID 进行 join，不需要仅靠 QID 或消息顺序推断。

## Researcher 输入和输出

输入 `input`：

- `question`：用户原始问题
- `global_intent_state`：当前动作之前的全局目标、约束、事实和开放问题
- `working_state`：当前 Loop 的子目标、完成条件、进展、证据缺口和下一方向
- `current_loop_messages`：仅保留当前 Loop 中、当前动作之前的消息
- `retrieval_query`：由当前子目标、完成条件、开放面和证据缺口组成
- `retrieved_memories`：只能从当前动作之前已经形成的跨 Loop Memory 中选择
- `available_memory_ids`：当时可检索的完整 Memory 集合，用于审计
- `tools`：该动作允许使用的原生浏览工具

输出 `target`：

- `TOOL_CALL`：原轨迹的 assistant 推理消息和原生 browser 调用
- `FINAL_ANSWER`：READY 边界之后，同一条原始轨迹的最终回答

## 防止数据泄漏

构造器会检查：

- 当前动作的目标消息没有出现在输入历史中；
- 输入历史的消息索引没有越过动作发生前的因果边界；
- 所有检索到的 Memory 在该动作发生之前已经存在；
- TOOL_CALL 必须有合法的工具名和 JSON 参数；
- FINAL_ANSWER 必须出现在 READY_TO_ANSWER 边界之后；
- 同一 QID 的不同 seed 轨迹通过已观察消息指纹精确匹配，最终答案不会串到另一条轨迹；
- 候选 ID、成品 ID 和对齐表 ID 必须一一对应。

## 质量门

候选集中的两个同 Loop 完全重复工具调用标记为 `REWRITE`，未进入成品。34 个搜索动作与 Memory 中已有事实存在词面重叠，但它们多数是在寻找精确原文、补充引用或验证当前证据缺口，因此保留为 `KEEP`，同时写入 `review_warnings` 供后续人工或 LLM 二审。

Researcher 标签直接来自 OpenResearcher 的原始 assistant 轨迹，没有额外调用 Mimo 改写，因此本轮没有新增 API token 消耗。Mimo-v2.5-pro 生成的 Controller 标签仍决定输入给 Researcher 的状态和 Memory。

## 文件

- 成品：`data/researcher-training/researcher_memory_conditioned.jsonl`
- 全部候选：`data/researcher-training/researcher_memory_conditioned.candidates.jsonl`
- 对齐表：`data/researcher-training/researcher_memory_conditioned.alignment.jsonl`
- QA：`data/researcher-training/researcher_memory_conditioned.qa.json`
- 样本预览：`data/researcher-training/researcher_memory_conditioned.preview.md`

服务器目录与本地相同，位于项目根目录的 `data/researcher-training/`。

## 重建

本地只构造工具动作：

```bash
scripts/build_researcher_training200.sh
```

服务器补齐同轨迹最终答案：

```bash
PYTHON=vendor/openresearcher/.venv/bin/python \
OPENRESEARCHER_RAW_GLOB='data/openresearcher-dataset/seed_*/*.parquet' \
bash scripts/build_researcher_training200.sh
```

当前 JSONL 是保留完整状态、Memory、对齐和审计字段的 canonical 数据。真正启动 SFT 前，再单独冻结 Researcher system prompt 和 chat template，将 canonical 数据序列化成模型所需的 `messages` 格式；这样改 prompt 时不需要重新生成标签。
