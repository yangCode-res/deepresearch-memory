# mimo-v2.5-pro 回溯式 Working State 40 条试验评审

## 结论

最终数据集可作为训练管线和标签契约的 pilot 数据，但规模仍不足以用于正式训练。

- 40 个样本，来自 10 条完整 OpenResearcher 轨迹，每条轨迹精选 4 个决策点。
- 教师模型全部为 `mimo-v2.5-pro`。
- 动作分布为 20 `CONTINUE_CURRENT_LOOP`、10 `SWITCH_LOOP`、10 `READY_TO_ANSWER`。
- 自动校验未发现状态链断裂、未来消息 ID 泄漏、StateDelta 不一致、方向字段中的工具操作或 Global State 操作性事实污染。
- 人工复核后，剔除了把“换来源验证同一结论”错误切成新 Loop 的轨迹。

## 最终文件

- `data/working-state-labels/working_state_retrospective_mimo25pro_training40.jsonl`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training40.preview.md`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training40.segments.jsonl`
- `data/working-state-labels/working_state_retrospective_mimo25pro_training40.qa.json`

## 最终样本构成

| QID | Loop 划分摘要 | 评价 |
| --- | --- | --- |
| 4626 | 定位指定文章 → 提取开放获取资助模式 | 清晰的“定位 → 提取” |
| 5282 | 确定任职者 → 建立其与英格兰教会领导角色的关系 | 两个不同事实依赖 |
| 5005 | 识别 Scratch 的身份 → 排除同名歧义 | 身份识别与指代消歧分离 |
| 5352 | 识别修道院 → 提取创立年份和最初奉献对象 | 实体解析与属性提取分离 |
| 5095 | 定位指定文章 → 提取 MEDLINE 订阅状态 | 清晰的“定位 → 提取” |
| 5485 | 定位指定榜单 → 匹配描述对应的网站 | 清晰的“定位 → 提取” |
| 5183 | 确定年份 → 处理 major/large 歧义 → 用明确措辞收敛 | 包含真正的语义歧义处理 |
| 6419 | 确定铜的电子转移数 → 建立法拉第定律关系 | 两个不同计算依赖 |
| 6369 | 定位指定文章 → 提取 DOAJ 论文数量 | 清晰的“定位 → 提取” |
| 5668 | 定位文章 → 定位 Semantic Scholar 条目 → 提取数量和方法 | 严格校验后重新生成 |

完整轨迹共有 22 个 Loop：8 条轨迹为 2 个 Loop，2 条轨迹为 3 个 Loop，平均 2.2 个 Loop。精选集只保留每条轨迹的第一个 Switch 边界，以维持 2:1:1 的动作平衡。

## 自动审计结果

| 检查项 | 结果 |
| --- | ---: |
| 样本数 | 40 |
| 唯一问题数 | 10 |
| 唯一决策点数 | 40 |
| Working State 在 CONTINUE 下发生 subgoal 漂移 | 0 |
| 弱契约占位语命中 | 0 |
| 方向字段工具/来源操作命中 | 0 |
| completed_subgoal 与 Loop 决策不一致 | 0 |
| Global State 操作性 confirmed fact | 0 |
| 因果消息 ID / 未来结构泄漏 | 0 |

Memory 标签共有 19 个正检索样本和 21 个负检索样本。输入中无候选 Memory 的样本为 18 个，有 1 个候选的样本为 20 个，有 2 个候选的样本为 2 个。在 12 个被选中的跨 Loop CONTINUE 样本中，9 个选择了历史 Memory，3 个未选择。未选择不必然是错误：判断发生在当前工具结果到达之后，若当前证据已经覆盖旧 Memory，保留负例可以训练模型避免重复注入。

## 人工剔除的典型问题

第一批 Pro 候选中，下列轨迹没有进入最终 40 条：

- `5173`、`5082`、`5202`：已经得到答案后，仅因换来源或继续佐证同一结论而切换 Loop。
- `5190`：出现 `Establish information dependency 2` 一类泛化子目标。
- `5054`：同一个电影问题被过度拆成 5 个 Loop，并出现“没有其他同名电影”的过强 Memory 结论。

补充池中，`5668` 的初版、`5679`、`6254`、`6217` 也因纯验证型边界没有采用。之后给分段校验器增加了两类硬约束：拒绝泛化契约；拒绝把基于其他来源、引用、一致性或准确性的同一结论验证单独切成新 Loop。严格版 `5668` 因而变成了“文章定位 → 条目定位 → 信息提取”。

## 与 mimo-v2.5 小样本池的成本比较

此前 `mimo-v2.5` 候选池为 8 条轨迹、54 个状态样本、288,756 tokens，约 5,347 tokens/状态样本。第一批 `mimo-v2.5-pro` 候选池为 12 条轨迹、106 个状态样本、638,779 tokens，约 6,026 tokens/状态样本，单状态样本 token 增加约 12.7%。

为补足人工审计后的 3 条替换，又生成了 6 条补充候选（286,240 tokens）和 1 条严格候选（27,141 tokens）。候选与替换生成合计 952,160 tokens；契约修复期间失败任务没有在退出前写完整 usage 报告，因此不把修复请求计入该合计。

Pro 的优势不是“直接输出即可训练”，而是更愿意拒绝一部分明显的伪边界，并能在严格错误反馈后写出更具体的契约。它仍会把纯验证误判为新 Loop，所以必须保留校验器和人工/critic 复核。

## 仍然存在的限制

1. 这是对 OpenResearcher 已记录行为的回溯标注，不是最优研究策略。轨迹中本来存在的冗余探索会留在同一 Loop 内，不会自动变成更早的 READY。
2. 只有 10 个问题，题型、Loop 长度和 Memory 候选数量都不够丰富。
3. 部分修复后的完成条件采用统一的 `Evidence is sufficient to ...` 形式，结构稳定但语言多样性较低。
4. State、Memory 和 Retrieval 标签主要来自同一个教师模型，后续仍需独立 critic 或双模型一致性检查，降低同源偏差。

## 下一步建议

先用这 40 条做一次极小规模 overfit/格式学习测试，验证模型能否稳定输出合法 JSON、保持 State 连续并学会三类动作。若这一步通过，再按严格分段规则扩到约 200 条，并在入库前增加独立边界 critic、检索相关性 critic 和 10% 人工抽检。不要直接从 40 条跳到大规模训练。
