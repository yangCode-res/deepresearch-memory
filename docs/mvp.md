# CL-GISM MVP

当前版本实现了第一版架构的离线闭环，不需要模型或外部服务即可运行：

```text
OpenResearcher row
→ Task Anchor + Raw Memory
→ LLM Loop Builder（无 key 时回退规则）
→ Heuristic State Updater + State Delta
→ BM25-style lexical retrieval
→ compact LLM context
```

## 运行示例

```bash
cd /Users/countsheep/Documents/memory
PYTHONPATH=src python examples/run_mvp.py
```

示例使用 `data/openresearcher-dataset/example_qid_39_short_raw.json`，会输出：

- 原始消息数量；
- 识别出的 Loop 数量；
- Global State 版本号；
- 状态条件化的 Retrieval Query；
- 进入 LLM 上下文的候选 Memory。

## 当前实现边界

- Loop 切分现在优先由 LLM 判断，没 key 时回退到规则基线。
- State Updater 是启发式基线，显式答案标记会升级为 confirmed，其余结论保留为 tentative。
- 检索是无依赖的 BM25-style 词法检索，后续可以替换为 Dense + BM25 融合。
- `pack_context` 输出的是结构化上下文对象，后续可接到真正的 LLM chat template。
- 完整 Raw Memory 始终保留；检索只决定哪些历史片段进入模型上下文。
- 如果设置了 `OPENAI_API_KEY`，`LLMLoopBuilder` 会先判断每个候选事件是否应该切换 loop；没有 key 时回退到规则版。
- 可以用 `examples/run_llm_loop.py` 单独查看 loop 切分结果。
- 如果设置了 `OPENAI_API_KEY`，可以用 `examples/run_llm_rerank.py` 让模型先对候选记忆重排；默认模型是 `gpt-5.6-terra`，可用 `CL_GISM_OPENAI_MODEL` 覆盖。
- 如果设置了 `OPENAI_API_KEY`，可以用 `examples/run_llm_state_update.py` 让模型决定 `StateDelta`；默认同样是 `gpt-5.6-terra`。
