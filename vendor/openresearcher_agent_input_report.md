# OpenResearcher 每轮 Agent 输入分析

本报告只分析隔离目录 `vendor/openresearcher` 中下载的代码，不修改当前 CL-GISM 实现。

## 结论

OpenResearcher 的每一轮模型输入不是单独的当前 Query，而是：

```text
到当前为止的完整 messages 历史
+ browser.search / browser.open / browser.find 工具定义
+ tokenizer 的 chat template
```

每一轮生成完成后，代码把新的 `assistant` 输出追加到 `messages`；如果有工具调用，再把工具结果追加为 `tool` 消息。下一轮重新把完整 `messages` 送进 chat template。

## 第一轮输入

在 `deploy_agent.py::run_one` 中，初始消息由两部分组成：

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
]
```

其中 `system_prompt` 是：

```text
DEVELOPER_CONTENT
+
Today's date: YYYY-MM-DD
```

工具定义单独从 `TOOL_CONTENT` 解析成 `tools` 列表，包含：

- `browser.search`
- `browser.open`
- `browser.find`

真正送入模型前，代码执行：

```python
prompt = generator.tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)
tokens = generator.tokenizer.encode(prompt, add_special_tokens=False)
```

因此模型实际看到的是 tokenizer 根据模型模板渲染后的文本/Token，而不是 Python 字典本身。

## 有工具调用的一轮

假设第一轮模型输出了搜索请求，代码先把它记录为：

```python
{
    "role": "assistant",
    "content": "",
    "reasoning_content": reasoning_content,
    "tool_calls": [
        {
            "id": "1",
            "type": "function",
            "function": {
                "name": "browser.search",
                "arguments": {"query": "..."}
            }
        }
    ]
}
```

然后执行工具，并追加：

```python
{
    "role": "tool",
    "tool_call_id": "1",
    "content": "搜索结果全文..."
}
```

下一轮输入就是：

```text
system
user
assistant(tool call)
tool(search result)
```

再加上同一组浏览器工具定义，并重新套 chat template。

## 没有工具调用的一轮

如果模型没有产生工具调用，代码会追加：

```python
{
    "role": "assistant",
    "content": "模型的回答文本",
    "reasoning_content": reasoning_content,
    "tool_calls": None
}
```

当内容出现 `<answer>...</answer>`、`Exact Answer:` + `Confidence:` 或 `Final Answer:` 等终止标记时，循环结束。

## 与数据集保存格式的差异

仓库当前 `run_one` 代码的初始运行时消息是：

```text
system + user
```

而已下载数据集中常见的保存记录是：

```text
system + developer + user + assistant + tool + ...
```

这说明数据集记录和当前仓库代码可能不是完全同一版的序列化格式，或者中间经过了 Harmony 消息转换。共同的语义是一致的：

```text
系统/开发者规则
+ 用户问题
+ 历史 assistant 输出
+ 历史 tool 返回
```

但训练我们自己的 Memory 系统时，不能只按角色名称判断输入结构，应该同时保留：

- role
- content
- reasoning_content
- tool_calls
- tool_call_id
- 消息顺序

## 与 CL-GISM 的映射

```text
每条原始消息                         → RawMemory
一次 assistant tool call + tool result → 一个候选 Loop
完整 messages 历史                    → Active Loop Workspace / 上下文
状态抽取结果                         → Global Intent State
下一轮需要的历史子集                 → Retrieval Plan + Selected Memory
```

最关键的观察是：OpenResearcher 当前实现本身没有独立的 Global State。它的“记忆”主要是把完整 `messages` 历史不断追加并重新送入模型。我们要做的 CL-GISM，可以在这个基础上增加显式的：

```text
Task Anchor
+ Global Intent State
+ Loop Memory
+ 精选历史 Memory
```

从而避免每轮都把全部历史原样塞回模型。
