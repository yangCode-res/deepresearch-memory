# mimo-v2.5-pro 回溯式 Working State 200 条数据评审

## 结论

本轮已将 pilot 从 40 条扩充为 200 条，可用于小规模格式学习、Loop 决策和 Memory 检索联合训练实验。

- 200 个样本，来自 50 条完整 OpenResearcher 轨迹，每条轨迹精选 4 个决策点。
- 10 条轨迹沿用已评审的 40 条 pilot，新增 40 条轨迹。
- 教师模型全部为 `mimo-v2.5-pro`。
- 动作分布为 100 `CONTINUE_CURRENT_LOOP`、50 `SWITCH_LOOP`、50 `READY_TO_ANSWER`。
- 通用校验和专项审计均未发现结构、状态链、因果消息 ID、方向字段或 StateDelta 违规。

## 最终文件

- `data/working-state-labels/working_state_retrospective_mimo25pro_training200.jsonl`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training200.preview.md`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training200.segments.jsonl`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training200.qa.json`
- `configs/working_state_training200_contract_overrides.json`

## 选样策略

每条完整轨迹固定选择四个训练点：两个 `CONTINUE_CURRENT_LOOP`、第一个 `SWITCH_LOOP` 边界和最终 `READY_TO_ANSWER` 边界。因此训练集保持 2:1:1 的动作平衡，同时覆盖切换前、切换后、切换瞬间和研究结束四种状态。

新增候选采用完整轨迹回溯分段。模型可以看到整条轨迹来判断真实子目标变化，但第二阶段的 State、StateDelta、Memory 和 Retrieval 标注只接收截至当前决策点的可见消息，不能看到未来消息或下一 Loop 契约。

人工筛选继续使用以下标准：

1. 换查询、换页面、换来源或失败重试不构成新 Loop。
2. 定位指定 artifact 与提取其目标信息可以是两个独立 Loop。
3. 只有出现不同的事实依赖、实体关系、计算依赖或用户要求的独立输出时才切换 Loop。
4. 得到答案后再核验同一结论、补引用或汇总证据，不单独构成新 Loop。

本轮完整生成的主要新增候选池共有 70 条轨迹，最终采用 40 条。典型剔除包括：

- `7747`、`7814`：提取答案后又切换到“确认原文/上下文”。
- `7590`：识别总统后又单独拆出“确认签署”和“汇总引用”。
- `967`：同一 Tasmania 结论被连续拆成多次来源核验。
- `2904`：WHO 已经给出传播方式后，再用 CDC 核验同一结论。
- `2969`：效应量已提取后，又把补齐论文引用信息作为新 Loop。

## 显式契约修复

8 条入选轨迹的 Loop 边界和事实标签正确，但教师生成的方向字段出现模板残片或拼接乱码。使用版本化的 `contract_overrides` 只修复 `subgoal` 和 `completion_test` 文本，不修改 Loop 范围、动作、证据 ID、StateDelta 或 Memory：

- `5009`、`5120`、`767`、`3066`
- `3936`、`738`、`2311`、`7547`

其中 `7547` 的真实结构是“识别 Walter V 的配偶 → 确定 Joanna of Châtillon 的父亲”。修复前第二个契约包含 `the requested values` 拼接残片；事实、Memory 和证据链本身无污染。

## 自动审计结果

| 检查项 | 结果 |
| --- | ---: |
| 样本数 | 200 |
| 唯一问题数 | 50 |
| 唯一决策点数 | 200 |
| `CONTINUE` 下 subgoal 漂移 | 0 |
| schema / 状态链违规 | 0 |
| 因果消息 ID 或未来结构泄漏 | 0 |
| 泛化契约或教师模板乱码 | 0 |
| 方向字段中的工具/来源操作 | 0 |
| `completed_subgoal` 不一致 | 0 |
| Global State 操作性 confirmed fact | 0 |
| 非 `mimo-v2.5-pro` 教师样本 | 0 |

## Memory 与 Retrieval 分布

- 输入中没有跨 Loop Memory 候选的样本为 93 个，有 1 条候选的样本为 98 个，有 2 条候选的样本为 9 个。
- 113 个样本不选择 Memory，82 个选择 1 条，5 个选择 2 条；共有 87 个正检索样本、92 个被选 Memory ID。
- 50 个精选的跨 Loop `CONTINUE` 角色中，42 个选择了历史 Memory，8 个保留为负例。

负例不必然代表标注错误：如果当前 Loop 新证据已经覆盖旧 Memory，Retrieval 应避免重复注入。不过，后续训练评估仍应分别报告“有候选但不选”和“根本没有候选”两类负例。

## 生成成本

本轮 5 个主要新增候选池的已完成请求共消耗 8,182,332 tokens；最后一个 3 轨迹兜底池消耗 332,537 tokens，但没有样本入选。已完成任务合计 8,514,869 tokens。此前被提前取消的低通过率试跑没有完整 usage 报告，因此不计入该数值。

成本主要来自整轨迹回溯分段、失败后的修复请求，以及对完整轨迹每个决策点执行因果 State 标注。下一轮扩大规模前，应该先增加便宜的边界 critic，在进入逐决策 State 标注前过滤重复核验型轨迹。

## 仍然存在的限制

1. 数据回溯标注 OpenResearcher 的实际行为，并不等同于最优研究策略；同一 Loop 内的冗余搜索仍可能保留。
2. artifact 定位阶段的工具结果有时会顺带暴露下一 Loop 的候选答案。Memory 可以帮助后续模型复用该信息，但训练数据本身不会自动把原轨迹改写成更早的 `READY_TO_ANSWER`。
3. 50 个问题仍不足以覆盖正式训练所需的题型和研究长度分布。
4. State、Memory 和 Retrieval 主要来自同一个教师模型。扩大到正式规模前，应加入独立 critic 或双模型一致性检查。

## 建议用法

先用这 200 条做 overfit/小规模 SFT，检查模型能否稳定输出合法 JSON、保持 State 连续、区分三类动作，并学习在跨 Loop 后选择或拒绝历史 Memory。通过后再扩到 1,000 条；扩充前优先加入轨迹级边界 critic，避免再次为明显的纯验证型分段支付逐决策标注成本。
