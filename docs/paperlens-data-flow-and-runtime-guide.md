# PaperLens 数据流转与运行机制

本文以当前代码为准，说明 PaperLens 从论文上传、解析、索引，到多轮问答、多智能体协作、长期记忆和 LLMOps 观测的完整运行过程。它既是项目理解文档，也是定位问题时的代码导航。

## 1. 系统解决什么问题

PaperLens 面向科研论文阅读。它不是把整篇 PDF 直接交给大模型，而是先把论文拆成可定位、可检索、可验证的 Text、Figure 和 Table Evidence，再根据问题复杂度选择 Standard RAG 或 LangGraph 多智能体 RAG。

系统遵循四条核心约束：

1. 回答只能引用当前论文中真实存在的 Evidence ID。
2. 会话历史和长期记忆用于理解用户意图，不能替代论文证据。
3. Figure、Table 问题必须执行对应模态的读取流程，不能仅凭附近正文作答。
4. 执行成功、证据充分和答案正确分别统计，不能互相替代。

## 2. 组件与数据存储

| 组件 | 职责 | 主要位置 |
|---|---|---|
| FastAPI | 上传、论文查询、会话、SSE、记忆、指标 API | `app/main.py` |
| Docling | 提取章节、图片、表格和结构信息 | `app/services/parser.py` |
| PyMuPDF | 提取页码、文本块、句子和 PDF 坐标 | `app/services/parser.py` |
| SQLite | 保存论文、Evidence、会话、Run、Trace 和长期记忆 | `data/paperlens.sqlite3` |
| Qdrant | 保存 BGE-M3 稠密向量索引 | `data/qdrant` 或远程 Qdrant |
| BM25 | 根据当前论文 Chunk 临时计算词法排名 | `app/services/retrieval.py` |
| BGE-M3 | 生成查询和 Evidence 的稠密向量 | `app/services/retrieval.py` |
| CrossEncoder | 对融合后的候选进行精排 | `app/services/retrieval.py` |
| DeepSeek | Query Resolver、Function Calling、证据判断和答案生成 | `app/services/reading.py` |
| Qwen-VL | Figure 原图分析、必要时的 Table 图片降级读取 | `app/multimodal` |
| LangGraph | Supervisor、Specialist、Critic、Repair、Answer 编排 | `app/multi_agent/graph.py` |
| Prometheus | 保存指标 | `observability/prometheus.yml` |
| OpenTelemetry Collector | 接收并转发 Trace | `observability/otel-collector.yml` |
| Tempo | 保存分布式 Trace | `observability/tempo.yml` |
| Grafana | 展示指标和 Trace | `observability/grafana` |

SQLite 中的主要业务表包括：`papers`、`chunks`、`figures`、`sentences`、`document_registry`、`qa_history`、`agent_traces`、`figure_query_cache`、`conversations`、`conversation_messages`、`agent_runs`、`paper_learning_memories` 和 `paper_memory_items`。

## 3. 论文上传与离线解析

### 3.1 上传入口

浏览器调用 `POST /api/papers`。服务端按流式方式保存文件，同时执行：

1. 检查扩展名、文件头和最大大小。
2. 计算 SHA-256 内容哈希。
3. 使用 `document_registry` 做内容级去重。
4. 为新论文创建 `paper_id`，状态先设为 `processing`。
5. 把原 PDF 保存到 `data/uploads/<paper_id>/source.pdf`。

同一内容已经完成解析时会复用已有论文；正在解析时返回 processing；失败时释放占位记录并清理未完成文件和向量索引。

### 3.2 Text 解析

默认优先使用 Docling 获取章节层级和文档结构，同时使用 PyMuPDF 提取每页的文字、单词坐标和句子坐标。

两者的分工是：

- Docling 提供更好的逻辑结构和章节标签。
- PyMuPDF 提供稳定的页码与 BBox，保证前端能跳转 PDF 并高亮原句。
- Docling 失败时，系统自动降级为纯 PyMuPDF 解析，保留基础文本和图片能力。

最终文本对象分为：

- `Chunk`：用于 BM25、Dense 和 Reranker 检索。
- `Sentence`：用于最终引用、Claim 校验和 PDF 高亮。

### 3.3 Figure 解析

Docling 优先导出 Figure 原图，同时保存：

- `figure_id`
- 页码和 BBox
- Caption
- Section
- Nearby Text
- Related Sentence IDs
- Image Path

`FigureUnderstandingService.enrich_paper()` 在建库阶段调用 Qwen-VL，为可读取图片生成结构化描述，例如 Summary、Entities、Components、Relations、Visual Evidence 和 Retrieval Keywords。

原图和结构化描述分工不同：结构化描述用于检索召回；真正的纯视觉问题仍需对命中的原图执行 `analyze_figure_for_query`。

### 3.4 Table 解析

Docling 将表格转换为 Markdown/结构化文本，同时保留表格截图、Caption、页码、Section 和 BBox。结构化内容作为 Table Evidence 参与 BM25 和向量检索。

正常情况下 Table Agent 使用 `read_table` 读取结构化行列。只有结构化解析不可用且显式开启 `TABLE_VLM_FALLBACK_ENABLED` 时，才调用 Qwen-VL 读取表格图片。

### 3.5 建立检索索引

系统把 Text Chunk、Figure 描述和 Table 内容转换为统一的可检索 Chunk：

```text
Text Chunk ───────────────┐
Figure Caption + 描述 ────┼─> BGE-M3 ─> Qdrant
Table Caption + Markdown ─┘
```

Dense 索引持久化到 Qdrant。BM25 不单独持久化倒排库，而是在当前论文的候选 Chunk 上计算词法得分。

### 3.6 阅读卡片

论文解析与索引完成后，DeepSeek 根据论文内容生成阅读卡片。卡片结论绑定候选 Sentence ID，避免生成无法定位到原文的摘要。

完成后论文状态改为 `completed`，允许在线问答。

## 4. 会话、消息和 Run

每个 Conversation 永久绑定一篇论文。用户提交消息时，系统在一个 SQLite 事务中创建：

- User Message
- Agent Run
- `client_message_id` 幂等标识

接口立即返回 `202` 和 `run_id`，后台异步执行本轮问答。数据库唯一约束保证同一 Conversation 同时最多只有一个 queued/running Run。

前端随后订阅：

```text
GET /api/papers/{paper_id}/conversations/{conversation_id}/runs/{run_id}/events
```

通过 SSE 接收 Router、LangGraph Node、Specialist、Function Call、Tool、Citation、Answer Delta 和最终状态事件。

## 5. 上下文与查询消解

每轮问答开始时加载：

- 当前 Conversation 最近若干轮消息
- Conversation Summary
- 历史引用 Evidence ID
- 当前论文的跨会话学习记忆摘要

`QueryResolver` 先识别三类请求：

1. `conversation_history`：例如“我之前问过哪些问题”，只读取当前会话历史。
2. `paper_learning_memory`：例如“总结我之前研究过的内容”，读取论文级跨会话记忆。
3. `paper_question`：需要重新进入论文 RAG。

对“它、上一张图、刚才的表格”等依赖历史的追问，Resolver 会改写为可独立检索的问题。历史答案只用于消解指代，不能作为 Citation。

## 6. Router 如何分流

`QueryRouter` 使用可解释规则识别问题复杂度与所需模态：

- 简单、单一文本事实：`standard_rag`
- 复杂文本推理：`agentic_rag`
- Figure、Table 或跨模态问题：`agentic_rag`

显式 `Figure N`、`Table N` 会触发对应模态。方法主张与实验结果是否一致这类问题，即使没有直接写 Table，也会要求 Table Evidence。

纯 Figure 问题只走 Figure 路径；纯 Table 问题只走 Table 路径，避免无意义 Text 检索。

## 7. Standard RAG 数据流

Standard RAG 用于简单文本事实：

```text
Question
  -> BM25 + BGE-M3/Qdrant
  -> RRF 融合
  -> CrossEncoder 精排
  -> 候选 Chunk
  -> 句子级 Evidence
  -> Grounded Answer
  -> Claim Verification
  -> Citation + Trace
```

支持 `bm25`、`dense`、`hybrid` 和 `hybrid-rerank`。默认最终配置使用 Hybrid + RRF + CrossEncoder，并记录 BM25、Dense、Rerank 各阶段耗时和实际降级情况。

## 8. LangGraph 多智能体数据流

### 8.1 图结构

```text
START
  -> Supervisor Plan
  -> Dispatch Specialists
  -> Evidence Critic
       -> approved/partial -> Answer Agent -> END
       -> retry            -> Targeted Repair -> Evidence Critic
       -> refuse           -> Structured Refusal -> END
```

Planner 是 Supervisor 内部的规则规划组件，不是独立 Agent。它把问题拆成 typed sub-tasks，并给每个任务分配 Agent、查询、预算和执行模式。

### 8.2 Specialist 分工

| Specialist | 白名单工具 | 成功条件 |
|---|---|---|
| Text Agent | `search_text`、`read_sentence` | 获得可引用文本 Evidence |
| Figure Agent | `search_figures`、`read_figure`、`analyze_figure_for_query` | 命中 Figure 且完成问题相关原图分析 |
| Table Agent | `search_tables`、`read_table`、可选 Table VLM fallback | 读取到结构化 Table Evidence |

三个 Specialist 共享论文数据源，但不共享各自未提交的候选结果。它们通过 LangGraph State 中的统一 Evidence Reducer 合并最终 Evidence，并按 `evidence_id` 去重。

### 8.3 Evidence Critic

Critic 检查：

- Required Modalities 是否覆盖
- Evidence 是否为空
- 数值与表格结果是否冲突
- Figure 是否真的完成视觉分析
- Specialist 是否出现可恢复错误

Critic 最多触发一次定向返工，只重新执行缺失模态对应的 Specialist，而不是重跑整个图。无法补齐时进入结构化拒答。

### 8.4 Answer Agent

Answer Agent 只接收 Critic 允许的 Evidence。生成后继续执行 Claim-level Verification：

- 引用 ID 必须在候选 Evidence 集合内。
- Unsupported Claim 会被删除或使状态降为 partial。
- Evidence 不足时返回结构化拒答，而不是使用模型常识补全。

## 9. Function Calling 与固定流程回退

Specialist 默认使用 OpenAI-compatible Function Calling 协议。每个 Agent 只能看到自己的工具 Schema，参数还会在本地进行二次校验：

- 禁止额外字段
- 检查必填字段和类型
- 限制 `top_k`
- 拒绝跨 Agent 工具
- `paper_id`、文件路径和 URL 不交给模型控制

一次 Specialist 最多执行有限模型轮次和工具步骤，并受全局模型调用、Qwen-VL 调用和总超时预算约束。

若发生非法参数、空 Evidence、API 异常或协议失败，系统记录 `function_calling_fallback`，然后回退到固定 Specialist 流程。连续失败达到阈值才打开熔断器，冷却后自动试探恢复。

## 10. Evidence Contract

Text、Figure、Table 都转换为统一结构：

```json
{
  "evidence_id": "...",
  "type": "text | figure | table",
  "page": 1,
  "bbox": [0, 0, 100, 100],
  "content": "...",
  "section": "Method",
  "metadata": {}
}
```

统一 Contract 让检索、Agent、Critic、Answer、Evaluation 和前端引用使用同一种对象。Text 最终引用通常落到 Sentence；Figure/Table 引用落到对应对象本身。

## 11. 短期与长期记忆

### 11.1 会话短期记忆

短期记忆仅属于当前 Conversation，包括最近消息、摘要和引用 ID。它负责追问理解和刷新恢复，不跨 Conversation 直接展示。

### 11.2 论文级长期记忆

长期记忆按 `user_scope + paper_id` 隔离。当前部署只有 `local` user scope，尚未接入登录体系，因此不能宣称完整多租户隔离。

每个完成 Run 会形成一个原子记忆候选：

- `completed + answerable=true + 至少一个 Citation` -> `active`
- partial、拒答或无证据结果 -> `unresolved`
- 论文重新解析导致 Evidence 版本变化 -> `stale`
- 到期或容量淘汰 -> `archived`
- 用户主动遗忘 -> `forgotten`

记忆保存来源 Run、Conversation、解析版本、Evidence ID、模态、章节、重要性、置信度、过期时间、置顶状态和用户备注。聚合记忆只用于导航，不会成为论文事实证据。

## 12. SSE 实时输出

后台运行期间，`RunManager` 在内存中保存有界事件缓冲区。浏览器通过 EventSource 接收事件，并使用 `Last-Event-ID` 在进程存活期间补放遗漏事件。

SSE 能实时展示执行阶段和答案增量，但模型提供商当前返回完整响应，因此 Token 不是上游逐 Token 到达；系统是在答案生成完成后按固定字符数拆分为 `answer.delta`。

页面刷新后，最终消息、Citation 和 Trace 从 SQLite 恢复。进程重启后旧的内存事件缓冲区不会恢复，但持久化 Run 结果仍可查询。

## 13. LLMOps 数据流

```text
PaperLens
  ├─ /metrics --------------------> Prometheus ----> Grafana Dashboard
  ├─ OpenTelemetry OTLP/HTTP ----> OTel Collector -> Tempo -> Grafana Explore
  └─ JSON Structured Logs -------> Console
```

### 13.1 Metrics

Prometheus 指标覆盖：

- HTTP 请求和失败
- Run 数量、状态、活跃数和 P95
- LangGraph Node P95
- Tool/Function Calling 成功、回退、拒绝和熔断状态
- DeepSeek/Qwen-VL 请求、延迟和 Token
- 长期记忆写入与生命周期

Grafana 中 `/5m` 面板使用 `increase(...[5m])` 表示窗口内真实次数；进程生命周期累计 Counter 使用 `sum(counter) or vector(0)`，避免首次序列和无数据造成假零或 `No data`。

### 13.2 Trace

每轮会话根 Span 为 `paperlens.conversation_run`，保存：

- `run_id`
- `conversation_id`
- `paper_id`
- `question_preview`（最多240字符）
- `question_length`

可在 Grafana Explore 的 Tempo 数据源使用：

```traceql
{ resource.service.name = "paperlens" && span.run_id = "<run_id>" }
```

也可以按模型搜索：

```traceql
{ resource.service.name = "paperlens" && name = "provider.qwen_vl" }
```

Trace 瀑布图展示模型调用开始时间和持续时间；LangGraph、Tool 和 Function Calling 细节作为 Span Events 记录。成功的自定义 Span 显式标记为 `OK`，异常 Span 标记为 `ERROR` 并记录错误类型。

### 13.3 日志安全

JSON 日志携带 request/run/conversation/paper 关联字段，但 Prometheus Label 不使用这些高基数字段。API Key、Authorization、Prompt 和论文正文会被隐藏或截断。

## 14. 异常恢复

| 异常 | 处理方式 |
|---|---|
| Docling 失败 | 降级到 PyMuPDF |
| Dense/Reranker 未启用 | 非 strict 模式记录 warning 并使用可用后端 |
| 外部模型临时失败 | 有界指数退避重试 |
| Function Calling 协议错误 | 记录回退并执行固定 Specialist 流程 |
| 连续 Function Calling 失败 | 熔断，冷却后恢复试探 |
| 缺失模态 Evidence | Critic 定向返工一次 |
| Evidence 仍不足 | 结构化拒答或 partial |
| Run 重复提交 | `client_message_id` 幂等返回原 Run |
| 同会话并发提交 | SQLite 唯一约束拒绝第二个活跃 Run |
| 页面刷新 | 从 SQLite 恢复消息和最终状态 |
| 论文重新解析 | 旧解析版本记忆转为 stale |

## 15. Evaluation 数据流

冻结 v2 Benchmark 包含5篇论文、20条人工标注问题，按论文划分为 Dev 12条和 Test 8条。题型覆盖文本、表格、视觉、跨模态和拒答。

评测分别计算：

- Retrieval Recall@K、MRR
- Evidence Precision、Recall、F1
- Modality Routing Accuracy
- Execution Success
- Answer Correct（只有 Judge 可用时确定）
- Task Success（执行和答案同时正确）
- Answerability/Refusal Accuracy
- 延迟、Token、Agent、Tool 和 Qwen-VL 调用

视觉题使用独立 Qwen-VL Judge 重新查看 Gold Figure，避免回答链路使用自己的视觉分析自证。Test 配置在 Dev 调参后冻结，只运行一次，不继续根据 Test 调参。

## 16. 三个端到端示例

### 16.1 简单文本事实

```text
用户问题 -> Query Resolver -> Router -> Standard RAG
-> Hybrid Retrieval -> Sentence Evidence -> Answer -> Claim Verification
-> Citation -> SSE -> SQLite -> active memory
```

### 16.2 Figure 视觉问题

```text
用户问题 -> Router(figure) -> Supervisor -> Figure Agent
-> search_figures -> read_figure -> analyze_figure_for_query(Qwen-VL)
-> Evidence Critic -> Answer Agent -> Figure Citation
```

### 16.3 Figure + Method + Table 综合问题

```text
用户问题 -> Router(text, figure, table) -> Supervisor
-> Text/Figure/Table Specialists
-> Evidence Reducer
-> Evidence Critic
-> Answer Agent + Claim Verification
-> partial/completed -> memory.record -> Prometheus + Tempo
```

## 17. 关键代码导航

| 想理解的功能 | 入口文件 |
|---|---|
| API 与一次 Run 生命周期 | `app/main.py` |
| PDF 解析 | `app/services/parser.py` |
| 混合检索 | `app/services/retrieval.py` |
| 问题路由 | `app/agent/router.py` |
| LangGraph 图 | `app/multi_agent/graph.py` |
| Supervisor | `app/multi_agent/supervisor.py` |
| Specialist | `app/multi_agent/agents` |
| Function Calling | `app/multi_agent/function_calling.py` |
| 工具注册与 Schema | `app/multi_agent/tool_registry.py` |
| Evidence 工具 | `app/tools/evidence_tools.py` |
| Query Resolver | `app/services/query_resolver.py` |
| 会话服务 | `app/services/conversation.py` |
| 长期记忆 | `app/services/long_term_memory.py` |
| SSE | `app/services/run_manager.py` |
| 指标、日志、Trace | `app/observability.py` |
| 评测 | `app/evaluation`、`scripts/evaluate.py` |

## 18. 当前边界

- 当前主要面向本地单用户部署，没有身份认证和真正的多租户权限系统。
- SSE 是应用层答案分片，不是提供商原生 Token Streaming。
- Qwen-VL 是跨模态问题的主要延迟来源，复杂问题可能触发两次视觉调用。
- Function Calling 可能因空 Evidence 或协议错误回退，但固定流程能够恢复。
- 冻结 Benchmark 只有20条，适合验证工程闭环，不能代表大规模统计结论。
- 本地 SQLite、Qdrant 和进程内事件缓冲不适合直接水平扩展，多实例部署需要任务队列、共享状态和集中日志。
