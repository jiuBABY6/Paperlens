# PaperLens 6.0

Evidence-Grounded Multimodal Agentic RAG for Scientific Paper Reading。

系统保留文本 RAG 与 `sentence_id → page + bbox(es) → PDF.js` 高亮链路，并以 LangGraph 编排 Supervisor、Text/Figure/Table Specialists、Evidence Critic 和 Answer Agent。PaperLens 6.0 支持持久化多轮会话、论文级跨会话学习记忆、实时 LangGraph/Tool 事件，以及带严格 Schema、结果信封、熔断恢复和固定流程回退的生产化 Function Calling。

## 最终交付文档

- [PaperLens 数据流转与运行机制](docs/paperlens-data-flow-and-runtime-guide.md)：从上传、三模态解析、索引、路由、多智能体执行、证据校验、SSE、记忆到 LLMOps 的完整数据流。
- [PaperLens AI 应用开发技术面试题库](docs/paperlens-ai-application-interview-qa.md)：60 道基于当前代码与真实边界编写的问答。
- [LLMOps 与长期记忆工程指南](docs/llmops-and-long-term-memory.md)：观测组件、指标、Trace、记忆生命周期与验收方式。
- [对话式多智能体升级进度](docs/conversational-upgrade-progress.md)：升级阶段、自动化测试和评测结果。

当前版本全量自动化回归结果为 **159 passed**，覆盖解析与检索契约、多智能体编排、Function Calling 与熔断、对话持久化、SSE、长期记忆、评测指标、Trace 关联字段和 Grafana 查询口径。该结果表示代码回归通过，不等同于真实业务问题准确率；模型质量指标见后文冻结 Dev/Test 报告。

## 系统架构与 Agentic RAG 流程

系统分为两个阶段：论文上传时完成解析、离线视觉理解与向量建库；用户提问时由 Query Router 在 Standard RAG 和 LangGraph 多智能体链路之间分流。

### 论文上传、离线解析与索引

```mermaid
flowchart TD
    UPLOAD[上传 PDF] --> VALIDATE[校验扩展名、文件头和大小]
    VALIDATE --> STAGE[流式保存临时文件<br/>计算 SHA-256]
    STAGE --> DEDUP{内容哈希去重}

    DEDUP -->|已有 completed 记录| REUSE[复用已有论文和分析结果]
    DEDUP -->|已有 processing 记录| PROCESSING[返回 202 processing]
    DEDUP -->|新论文| RESERVE[创建 paper_id<br/>状态设为 processing]

    RESERVE --> SAVE[保存 source.pdf]
    SAVE --> PARSER{PaperParser}
    PARSER --> DOCLING[Docling 结构解析<br/>章节、Figure、Table、OCR]
    PARSER --> PYMUPDF[PyMuPDF 文本、页码<br/>Sentence 与 BBox]
    DOCLING --> ALIGN[章节结构与文本坐标对齐]
    PYMUPDF --> ALIGN
    DOCLING -. 失败时 .-> FALLBACK[PyMuPDF 降级解析]

    ALIGN --> TEXT[Text Chunk / Sentence<br/>Page / Section / BBox]
    ALIGN --> FIG[Figure Crop / Caption<br/>Nearby Text / Related Sentences]
    ALIGN --> TABLE[Table Image / Markdown<br/>Rows / Columns / BBox]
    FALLBACK --> TEXT
    FALLBACK --> FIG

    FIG --> FIG_CHECK{可读取的 Picture?}
    FIG_CHECK -->|是| OFFLINE[Qwen-VL 离线 Figure 理解]
    FIG_CHECK -->|否或调用不可用| FIG_META[保留原图与元数据<br/>记录 unavailable / error]
    OFFLINE --> FIG_DESC[Summary / Entities / Relations<br/>Keywords / Uncertainty]

    TEXT --> REPRESENT[统一检索表示]
    TABLE --> REPRESENT
    FIG_DESC --> REPRESENT
    FIG_META --> REPRESENT
    REPRESENT --> EMBED[BGE-M3 批量生成向量]
    EMBED --> QDRANT[(Qdrant<br/>论文向量 Collection)]

    REPRESENT --> CARD[DeepSeek 生成阅读卡片]
    CARD --> CARD_VERIFY[结论绑定候选 Sentence ID]
    QDRANT --> INDEX_STATUS[记录 vector_indexed<br/>index_version / index_error]
    CARD_VERIFY --> SQLITE[(SQLite<br/>Paper / Evidence / Card / Status)]
    INDEX_STATUS --> SQLITE
    SQLITE --> COMPLETE[状态 completed<br/>允许在线提问]

    PARSER -. 无法恢复的异常 .-> CLEANUP[清理未完成文件和索引<br/>释放上传记录]
```

离线阶段持久化的是 BGE-M3/Qdrant 稠密向量索引；BM25 在在线检索时根据当前论文的 Chunk 计算词法排名。Docling 失败时系统降级到 PyMuPDF，保留基本文本、页码、Sentence BBox 与图片能力。

### 对话入口、Memory 与流式运行

```mermaid
flowchart TD
    UI[三栏工作台] --> MSG[提交 User Message]
    MSG --> TX[原子创建 Message + Run]
    TX --> SSE[SSE 订阅运行事件]
    TX --> CONTEXT[加载当前 Conversation<br/>摘要 + 最近 6 轮 + 历史引用 ID]
    TX --> PAPER_MEMORY[(论文级跨会话学习记忆)]
    PAPER_MEMORY --> RESOLVE
    CONTEXT --> RESOLVE{是否依赖历史?}
    RESOLVE -->|否| ORIGINAL[沿用原问题]
    RESOLVE -->|是| REWRITE[Query Resolver<br/>改写为独立检索问题]
    REWRITE -->|无法唯一消歧| CLARIFY[请求澄清]
    ORIGINAL --> RAG[单轮 RAG 内核]
    REWRITE --> RAG
    RAG --> EVENTS[节点 / Tool / Citation / Answer Delta]
    EVENTS --> SSE
    RAG --> PERSIST[一次性保存 Assistant Message<br/>Run / Citation / Trace]
    PERSIST --> MEMORY[更新当前会话短期记忆]
    PERSIST --> LONG_MEMORY[仅将已验证问答写入<br/>论文级学习记忆]
    MEMORY --> RECOVER[刷新页面恢复完整时间线]
    LONG_MEMORY --> RECOVER
```

Conversation Memory 只理解当前会话中的“它、上一张图、刚才的表格”等上下文；Paper Learning Memory 按论文记录跨会话的已验证问答、章节/图表进度和未解决问题。两者都不能作为论文事实 Citation，每轮仍会重新检索并构建 Evidence Memory。LangGraph `thread_id` 使用 `conversation_id:turn_index`，避免上一轮 Plan、预算或错误污染下一轮。

### 单轮 RAG 内核与 LangGraph 多智能体编排

```mermaid
flowchart TD
    QUESTION[用户问题] --> ROUTER{查询路由器<br/>复杂度 + 所需模态}

    DATA[(共享论文数据层<br/>PDF / SQLite / Qdrant<br/>Chunk / Sentence / Figure / Table)]

    ROUTER -->|简单文本事实| STANDARD[Standard RAG]
    STANDARD --> QUERY_PLAN[语义查询 + 关键词查询]
    QUERY_PLAN --> TEXT_SEARCH[BM25 + BGE-M3/Qdrant]
    TEXT_SEARCH --> FUSION[RRF 融合 + CrossEncoder 精排]
    DATA -. 读取同一数据源 .-> TEXT_SEARCH
    FUSION --> STANDARD_EVIDENCE[本次请求的文本证据]
    STANDARD_EVIDENCE --> STANDARD_ANSWER[基于文本证据生成回答]
    STANDARD_ANSWER --> STANDARD_VERIFY[标准链路结论校验]
    STANDARD_VERIFY --> STANDARD_RESULT[标准 RAG 结果<br/>文本引用 + 标准链路轨迹]

    ROUTER -->|复杂文本、纯 Figure、纯 Table 或跨模态| GRAPH[LangGraph]
    GRAPH --> SUPERVISOR[主管 Agent]
    SUPERVISOR --> PLANNER[规则任务规划器<br/>逻辑任务 + 查询改写]
    PLANNER --> PLAN[结构化子任务<br/>Agent 分配 + 预算 + 执行模式]
    PLAN --> DISPATCH{有边界的专业 Agent 调度器}

    DISPATCH --> TEXT_AGENT[文本研究 Agent]
    DISPATCH --> FIGURE_AGENT[图片分析 Agent]
    DISPATCH --> TABLE_AGENT[表格分析 Agent]

    TEXT_AGENT --> TEXT_TOOLS[检索 / 读取文本证据]
    FIGURE_AGENT --> FIGURE_TOOLS[检索图片 + Qwen-VL 原图分析]
    TABLE_AGENT --> TABLE_TOOLS[检索 / 读取结构化表格<br/>可选视觉模型降级方案]
    DATA -. 读取同一数据源 .-> TEXT_TOOLS
    DATA -. 读取同一数据源 .-> FIGURE_TOOLS
    DATA -. 读取同一数据源 .-> TABLE_TOOLS

    TEXT_TOOLS --> MEMORY[(本次 LangGraph 运行的<br/>共享证据池)]
    FIGURE_TOOLS --> MEMORY
    TABLE_TOOLS --> MEMORY
    MEMORY --> CRITIC{证据审核 Agent}

    CRITIC -->|通过 / 部分通过| ANSWER[答案整合 Agent]
    CRITIC -->|定向返工一次| REPAIR[定向返工调度<br/>只重跑指定专业 Agent]
    REPAIR --> CRITIC
    CRITIC -->|拒答| REFUSAL[结构化拒答]

    ANSWER --> VERIFY[结论级证据校验]
    VERIFY --> AGENT_RESULT[多智能体结果<br/>文本 / 图片 / 表格引用 + Agent 运行轨迹]
    REFUSAL --> AGENT_RESULT

    STANDARD_RESULT --> API[统一 API 响应格式]
    AGENT_RESULT --> API
    API --> UI[PDF.js 页码跳转 + BBox 高亮]
    API -. 评测脚本离线回放 .-> EVAL[确定性评测<br/>检索 / 证据 / 路由 / 执行]
    EVAL --> JUDGE_GATE{是否启用 --judge?}
    JUDGE_GATE -->|否| NO_JUDGE[answer_correct / task_success = null]
    JUDGE_GATE -->|是：文本、表格及非 visual-only| TEXT_JUDGE[DeepSeek 文本裁判]
    JUDGE_GATE -->|是：visual-only| VISUAL_JUDGE[独立 Qwen-VL 视觉裁判<br/>重新读取标准答案对应图片]

    GRAPH -. 每个节点持久化运行状态 .-> CHECKPOINT[(SQLite 状态检查点)]
```

Router 只负责入口分流、复杂度和模态识别；Planner 是 Supervisor 内部的规则规划组件，并非独立 LangGraph Agent。纯 Figure 和纯 Table 问题同样进入 LangGraph，由 Supervisor 分派给对应 Specialist。Specialist 使用模型原生 Function Calling，但只能看到各自 strict 白名单工具；非法参数、重复调用、预算耗尽或协议失败会记录 Trace，并回退到固定流程。连续协议失败达到阈值才熔断，冷却后自动恢复试探。Figure 必须完成查询相关原图分析，Table 必须完成结构化读取，否则不会仅因“搜索到了对象”而宣告成功。Standard RAG 与多智能体 RAG 读取同一份持久化论文数据和索引，但每次请求的 Query Plan、候选结果、Evidence、State、Citation 与 Trace 相互隔离。

| 层次 | 主要职责 | 关键实现 |
|---|---|---|
| 解析与索引 | 保留论文结构、原图、表格和句子坐标 | Docling、PyMuPDF、SQLite、Qdrant |
| 检索 | 词法、语义、融合与精排 | BM25、BGE-M3、RRF、BGE CrossEncoder |
| Agent | 职责隔离、多智能体调度与有界返工 | LangGraph、Supervisor、Specialists、Critic、Shared Evidence Memory |
| 可信回答 | 限制证据 ID、验证原子 Claim、结构化拒答 | DeepSeek、统一 Evidence Contract |
| 视觉理解 | 对命中原图执行问题相关分析和独立评测 | Qwen-VL、版本化缓存、低分辨率放大 |
| 可观测性 | 记录路由、工具、候选、耗时和模型调用 | Agent Trace、离线 Evaluation |

## 核心能力

- Text：原有 Chunk/Sentence 解析、BM25、Dense、RRF、reranker、DeepSeek 生成与句子级验证完整保留。
- Figure：Docling 优先裁出具体 Figure；保存 caption、section、nearby_text、related_sentence_ids 和原图。建库时调用一次 Qwen-VL 生成结构化检索描述；查询时只对命中的 Figure 查看原图。
- Figure 编号：问题显式指定 `Figure N`/`图 N` 且图注中存在该编号时，检索会先执行精确约束，只分析被点名的原图；图注编号缺失时自动退回语义检索。
- 视觉读取：低分辨率 Figure 会在内存中按比例放大后再发送给 Qwen-VL，原始文件不变；查询缓存带显式版本，提示词或预处理升级后不会复用旧版结果。
- Table：Docling Markdown 同时保留为原文、列名和行对象，经 BM25/Dense 检索；结构化表不可读时可通过 `TABLE_VLM_FALLBACK_ENABLED` 显式启用 Qwen‑VL fallback。
- Agent：Supervisor 生成 typed sub-tasks，Text/Figure/Table Agents 只调用白名单工具；Critic 检查模态覆盖和数值冲突，并最多定向返工一次。
- Evidence：`text | figure | table` 统一为 `evidence_id/type/page/bbox/content/section/metadata`；旧 sentence_id 不变。
- Verification：所有模型返回 ID 都先与候选集合比对；unsupported claim 会被删除。
- Recovery：外部 API 有限指数退避、节点超时、工具/模型/Qwen‑VL 预算、SQLite Checkpoint 和结构化错误；网络重试与 Critic 返工分开统计。
- Trace：记录 router、plan、每个 Agent/Node/Tool、result IDs、缓存命中、延迟、token usage、Qwen-VL 调用、Critic 决策与恢复结果。
- Evaluation：比较 `standard_rag`、`text_agentic_rag`、`multimodal_agentic_rag`；各模态独立计算 Top-K，并分开报告执行成功与答案正确。
- Conversation：一个会话永久绑定一篇论文，支持创建、切换、重命名、归档、幂等消息提交、刷新恢复和单会话活动 Run 互斥。
- Memory：当前会话使用有界短期记忆；同一论文使用跨会话结构化学习记忆。只有已验证问答进入学习记录，partial/refusal 进入未解决列表；历史回答不能充当 Citation。
- Function Calling：Text/Figure/Table Specialist 使用独立 strict Tool Schema、本地二次校验、统一结果信封、输出上限和失败熔断；`paper_id`、路径、URL 等上下文参数不交给模型控制，固定流程作为显式回退。
- SSE：Message API 立即返回 `202`，后台执行 Run；浏览器实时接收 Router、LangGraph Node、Specialist、Function Call、Tool、Repair、核验、答案增量和终态事件，支持 `Last-Event-ID` 进程内重放与协作式取消。

## 启动

```powershell
conda activate paperlens
cd D:\Desktop\job\demo
pip install -r requirements.txt
Copy-Item .env.example .env
# 将两个 YOUR_... 占位符替换为实际 Key；不用 Figure 时 Qwen-VL Key 可留空。
# 默认启用 LangGraph；Function Calling 失败时自动回退固定 Specialist 流程。
# AGENT_ORCHESTRATOR=langgraph
# SPECIALIST_EXECUTION_MODE=function_calling_with_fallback
# FUNCTION_CALL_STRICT=true
# 可选启用无依赖 Specialist 并行：MULTI_AGENT_PARALLEL_ENABLED=true
uvicorn app.main:app --host 127.0.0.1 --port 8010
```

健康检查：`GET /api/health`。未设置 `QDRANT_URL` 时使用 `data/qdrant` 的单进程本地存储。

## LLMOps、Grafana 与长期记忆

本项目已加入一套本地可复现的 LLMOps 链路，而不只是在页面展示 Agent Trace：

- `GET /metrics` 暴露 Prometheus 指标，覆盖 HTTP、RAG 运行结果、LangGraph 节点、Tool/Function Calling、模型延迟与 Token、熔断降级、答案状态和记忆生命周期。
- OpenTelemetry 为 HTTP、单轮 Run、DeepSeek、Qwen-VL 和记忆写入创建关联 Span；对话根 Span 显式记录 `run_id/conversation_id/paper_id/question_preview`，成功 Span 标记为 `OK`，异常标记为 `ERROR`。这些高基数字段用于日志与 Trace 精确关联，但不会作为 Prometheus 标签。
- JSON 结构化日志自动隐藏 API Key、Authorization、Prompt 和论文正文，只记录状态、长度、计数和错误类型。
- `GET /api/health` 用于存活检查，`GET /api/ready` 检查 SQLite、数据目录和 LangGraph Checkpoint 目录，不调用外部模型，也不消耗 Token。
- Grafana 自动预置 `System Overview`、`Agents & Tools`、`Models & Memory` 三张 Dashboard；Prometheus 保存指标，Tempo 保存 Trace，OpenTelemetry Collector 负责接收与转发。

启动应用后，可在另一个终端启动观测组件：

```powershell
docker compose up -d prometheus tempo otel-collector grafana
```

访问地址：PaperLens 指标 `http://127.0.0.1:8010/metrics`，Prometheus `http://127.0.0.1:9090`，Grafana `http://127.0.0.1:3000`。Grafana 默认账号为 `admin`，密码读取 `.env` 的 `GRAFANA_ADMIN_PASSWORD`。生产环境必须替换默认密码，并为 `/metrics` 增加内网或网关访问控制。

新建一轮对话后，可用页面返回的 `run_id` 在 Grafana Explore → Tempo 中精确定位：

```traceql
{ resource.service.name = "paperlens" && span.run_id = "<run_id>" }
```

展开 `paperlens.conversation_run` 可看到安全截断的 `question_preview`，并沿子 Span 查看 Router、LangGraph、Tool、DeepSeek、Qwen-VL 和记忆写入。字段升级前生成的历史 Trace 不会被回填，验收时应重启应用并发起新请求。

Grafana 中标题带 `/ 5m` 的请求、Run、Tool、模型和记忆面板统一使用 PromQL `increase(...[5m])`，表示最近五分钟的实际新增次数，不是每秒速率。Function Calling fallback、拒绝调用和熔断状态属于低频故障信号，Stat 面板展示当前 PaperLens 进程生命周期累计值，并用 `or vector(0)` 将“尚未发生”显示为 `0`，避免误显示 `No data`。

论文级长期记忆采用“源 Run → 原子记忆项 → 聚合学习记忆”三层结构：

- 只有 `completed + answerable=true + 至少一个 Citation` 的回答进入 `active`；拒答和 partial 进入 `unresolved`，不会被当成可靠事实。
- 每条记忆保存来源会话、来源 Run、解析版本、Evidence ID/模态、章节、置信度、重要性、到期时间和用户备注，可完整追溯。
- 论文重新解析后，旧解析版本的记忆（包括置顶项）转为 `stale`，避免引用失效证据；普通记忆到期后转为 `archived`，置顶只跳过过期，不跳过版本失效。
- 支持历史 Run 幂等回填、容量上限、自动遗忘、置顶、备注和主动忘记。旧版聚合记忆在完成回填前保留，不会因升级被直接清空。
- 当前记忆按 `paper_id + local user_scope` 隔离；尚未接入登录体系，因此不能将它描述为完整多租户隔离。

长期记忆接口：

- `GET /api/papers/{paper_id}/learning-memory`：读取聚合学习进度并执行回填、版本失效与过期检查。
- `GET /api/papers/{paper_id}/learning-memory/items?include_inactive=true`：读取原子记忆及生命周期状态。
- `PATCH /api/papers/{paper_id}/learning-memory/items/{memory_id}`：置顶、修改重要性、添加备注或设为 `forgotten`。
- `POST /api/papers/{paper_id}/learning-memory/backfill`：从历史完成/部分完成 Run 幂等回填。
- `DELETE /api/papers/{paper_id}/learning-memory`：只清除派生记忆，不删除论文、会话和原始回答。

记忆质量可使用独立标注集评测，示例格式见 `evals/memory.cases.example.jsonl`：

```powershell
python scripts\evaluate_memory.py --dataset evals\memory.cases.jsonl --output evals\report-memory.json
```

报告区分状态准确率、Active Precision/Recall、错误激活率和活跃记忆来源可追溯率。完整实现与故障边界见 [LLMOps 与长期记忆工程指南](docs/llmops-and-long-term-memory.md)。

## API

- `POST /api/papers`：上传、解析、Figure 离线理解、索引、生成阅读卡片。
- `POST /api/papers/{paper_id}/reanalyze`：使用当前 parser/index/多模态版本重新分析。
- `POST /api/papers/{paper_id}/ask`：自动选择 Standard 或 Agentic RAG。
- `POST /api/papers/{paper_id}/conversations`：创建论文级会话。
- `GET /api/papers/{paper_id}/conversations`：列出当前论文的未归档会话。
- `GET/PATCH/DELETE /api/papers/{paper_id}/conversations/{conversation_id}`：读取、重命名与软归档。
- `POST /api/papers/{paper_id}/conversations/{conversation_id}/messages`：幂等提交消息并异步创建 Run。
- `GET /api/papers/{paper_id}/conversations/{conversation_id}/runs/{run_id}`：查询持久化终态。
- `GET .../runs/{run_id}/events`：订阅 SSE；支持浏览器自动携带 `Last-Event-ID` 重放。
- `POST .../runs/{run_id}/cancel`：请求协作式取消。
- `GET /api/papers/{paper_id}/evidence/{evidence_id}`：统一读取 text/figure/table Evidence。
- `GET /api/papers/{paper_id}/traces/{run_id}`：读取持久化 Trace。
- `GET /api/papers/{paper_id}/figures/{figure_id}`：读取 Figure/Table 原始图像。
- `POST /api/evaluation`：提交离线记录并计算三模式指标。

Agent Tool 层提供：`search_text`、`search_figures`、`search_tables`、`read_text`、`read_sentence`、`read_section`、`read_figure`、`analyze_figure_for_query`、`read_table`、`get_evidence`。

详细节点、边与恢复路径见 [LangGraph 多智能体架构](docs/langgraph-multi-agent-architecture.md)，实施决策见 [升级技术方案](docs/langgraph-multi-agent-upgrade-plan.md)。

## 索引与评测

旧论文需要执行重新分析，才能得到稳定 Figure/Table ID、邻近正文及完整 Qwen-VL 描述。只补描述和 v3 多模态向量索引可运行：

```powershell
python scripts\reindex_papers.py
```

原有文本检索/RAG 评测仍使用 `scripts/evaluate.py`。三模式统一指标接受 JSON 或 JSONL：

```powershell
python scripts\evaluate_agentic.py evals\agentic-results.jsonl --top-k 5
```

每条记录可提供 `mode`、expected/returned evidence IDs、各模态排名 IDs、claims、routing、steps、tool_calls、latency_ms、token_usage、qwen_vl_calls。输出包括按模态计算的 Recall@K/MRR、Evidence P/R/F1、Faithfulness、Claim Support Rate、执行成功率、答案正确率、Task Success、平均步骤/调用、重复率、路由准确率、延迟、Token 与 Qwen-VL 成本计数。路由准确率比较 Gold 模态与 Router 的实际 `routed_modalities`，不再用可能受拒答影响的最终引用模态。文本检索以 Chunk ID 计算 Retrieval，最终引用以 Sentence ID 计算 Evidence；二者不会再混用。未启用 Judge 时，`answer_correct` 与 `task_success` 为 `null`，不会把“执行完成”误当成“答案正确”。

显式 `Figure N` / `Table N` 查询直接使用元数据精确定位，不启动 Dense/Reranker。普通文本 `hybrid-rerank` 默认精排融合排名前 8 个候选，并将单对最大长度限制为 256 tokens；可通过 `RERANK_CANDIDATE_LIMIT` 和 `RERANK_MAX_LENGTH` 调整。检索 Trace 会分别记录 `bm25_latency_ms`、`dense_latency_ms` 与 `rerank_latency_ms`，用于定位性能瓶颈。

`--mode rag --judge` 对带 `visual-only` 标签的题使用独立 Qwen-VL 原图裁判，并单独记录 `visual_judge_qwen_vl_calls`。在线问答的二次视觉核验默认关闭；如需以额外一次 Qwen-VL 调用换取更保守的视觉答案检查，可设置 `ENABLE_ONLINE_VISUAL_JUDGE=true` 后重启服务。

视觉 Judge 会先生成 image-only observation，再分别检查 Gold 与候选答案。若 Gold 本身与原图冲突，报告使用 `dataset_issue=true` 并从答案正确率分母中排除该样本，避免坏标注制造假阴性。

纯 Figure 问题只执行 Figure Retrieval/Reading，不再附带 Text Retrieval。在线视觉二次核验与首次分析冲突时，回答状态改为 `partial`，Figure-backed claim 标记为 `visually_contested`，网页优先显示“视觉核验未通过”。

## 评测方法与最终结果

冻结主评测集 `questions.v2.jsonl` 由 5 篇论文的 20 条人工标注问题组成，并按论文划分为 `dev` 12 条和 `test` 8 条，同一篇论文不会同时进入两个 split。新增 `questions.v3.jsonl` 扩展为 30 条，经统一 Sentence/Figure/Table Evidence 校验通过，暂不作为最终对外指标。题目覆盖简单文本事实、复杂文本推理、结构化表格、必须观察原图的视觉问题，以及论文未报告目标信息时的拒答问题。

- Retrieval 使用人工标注的 Chunk/Figure/Table ID 计算 Recall@5 与 MRR；最终 Citation 使用 Sentence/Figure/Table Evidence ID 计算 Precision、Recall 和完整证据组命中率。
- 同一问题存在多处等价原文时，可使用 `expected_evidence_groups` 表示可替代 Gold 证据组。
- `execution_success` 只衡量链路是否正常完成；`answer_correct` 只在 Judge 可用时确定；`task_success` 要求执行与答案均正确。
- 文本答案由 DeepSeek 根据标准答案、Gold 原文和实际引用证据进行 0–2 分裁判；`visual-only` 问题由独立 Qwen-VL 重新查看 Gold Figure，避免使用回答链路的视觉分析自证。
- 所有参数先在 `dev` 上确定，随后冻结配置并只运行一次 `test`；没有根据最终 test 报告继续调参。

### PaperLens 6.0 对话式多智能体结果

本轮没有扩充正式 Benchmark；仍使用 5 篇论文、20 条冻结 v2 单轮题（Dev 12 / Test 8）。新增多轮 Smoke 只验证指代、会话隔离、取消和流式业务闭环，不包装成正式指标。完整报告见 [Conversation Dev](evals/report-dev-conversation-upgrade-final.json) 与 [Conversation Test](evals/report-test-conversation-upgrade-final.json)。

| 指标 | Dev（12 条） | 冻结 Test（8 条） |
|---|---:|---:|
| Execution Success | 100% | 100% |
| Task Success / Answer Correct | 100%（12/12） | 100%（8/8） |
| Answerability / Refusal Accuracy | 100% / 100% | 100% / 100% |
| Retrieval Recall@5 | 1.000 | 0.833 |
| Evidence Precision / Recall / F1 | 0.944 / 1.000 / 0.963 | 0.833 / 0.833 / 0.833 |
| 平均 / P95 延迟 | 11.28 s / 33.43 s | 15.06 s / 42.11 s |
| 平均 Token Usage | 9895.17 | 8556.63 |
| Qwen-VL 总调用（含 Judge） | 5 | 4 |
| 原生 Function Calls / Fixed Fallbacks | 19 / 1 | 14 / 2 |

Function Calling 增加了工具选择模型轮次，因此延迟和 Token 不应与旧 fixed 报告直接当作纯性能优化对比；它的收益主要是可审计的模型工具协议、参数约束和动态工具选择。冻结 Test 的严格 Gold ID Recall 为 0.833，但 8 条答案均通过独立 Judge；不能把它表述成“检索 100%”。

### Legacy 单 Agent 基线

Legacy 最终配置使用 `hybrid-rerank`、Rerank Top-8、最大输入长度 256、版本 2 Figure 查询缓存，并关闭在线二次视觉检查。旧版完整原始报告见 [Legacy Dev](evals/report-dev-rag-optimized-cold.json) 和 [Legacy Test](evals/report-test-final.json)。

| 指标 | Dev（12 条） | Test（8 条） |
|---|---:|---:|
| 执行成功率 | 100% | 100% |
| Task Success | 100%（12/12） | 100%（8/8） |
| Answerability Accuracy | 100% | 100% |
| Refusal Accuracy | 100% | 100% |
| Modality Routing Accuracy | 100% | 100% |
| Retrieval Recall@5 / MRR | 1.000 / 1.000 | 0.833 / 0.833 |
| Exact Gold Citation Hit | 100% | 83.33%（5/6 个可回答问题） |
| Evidence Precision / Recall | 0.944 / 1.000 | 0.833 / 0.833 |
| Answerable-case Judge 平均分 | 2.0 / 2.0 | 2.0 / 2.0 |
| 平均端到端延迟 | 8.22 s | 8.50 s |
| P95 端到端延迟 | 23.39 s | 28.85 s |
| Dataset Issue | 0 | 0 |

### LangGraph 多智能体 Dev/Test

LangGraph 使用相同的冻结 v2 数据集、检索配置与 Judge 口径。所有参数先在 Dev 上确定；随后保持配置不变，只运行一次 8 题 Test，未根据 Test 结果继续调参。完整报告见 [并行 Dev](evals/report-dev-langgraph-final.json)、[串行 Dev](evals/report-dev-langgraph-serial.json)、[最终 Test](evals/report-test-langgraph-final.json) 和 [消融摘要](evals/report-langgraph-ablation.json)。

| 指标 | Dev Serial（12 条） | Dev Parallel（12 条） | Final Test（8 条） |
|---|---:|---:|---:|
| Task Success / Answer Correct | 100% / 100% | 100% / 100% | 100% / 100% |
| Retrieval Recall@5 / MRR | 1.000 / 1.000 | 1.000 / 1.000 | 0.833 / 0.833 |
| Evidence Precision / Recall / F1 | 0.944 / 1.000 / 0.963 | 0.944 / 1.000 / 0.963 | 0.833 / 0.833 / 0.833 |
| Modality Routing Accuracy | 100% | 100% | 100% |
| Refusal Accuracy | 100% | 100% | 100% |
| Node Failure Rate | 0% | 0% | 0% |
| 平均 Agent 数 | 2.33 | 2.33 | 3.00 |
| 平均 Token Usage | 5180.92 | 5128.08 | 2852.88 |
| Qwen-VL 总调用（含独立视觉 Judge） | 2 | 2 | 2 |
| 平均端到端延迟 | 6.79 s | 6.54 s | 7.66 s |
| P95 端到端延迟 | 28.27 s | 24.27 s | 31.89 s |

并行相对串行保持质量指标完全一致，平均延迟下降 3.7%，P95 下降 14.1%。相对历史 Legacy Dev，LangGraph 并行平均延迟低 20.4%，但两份报告生成时间与缓存状态不同，该差值只作为方向性工程观察，不作为严格因果结论。最终 Test 的 8/8 条执行成功且答案判定正确，两个不可回答问题均正确拒答；测试期间没有节点失败、恢复或 Critic 返工。

Test 中唯一未命中精确 Gold 的 `minda-zero-shot-text-01` 检索到了其他直接支持答案的论文句子，Claim verifier 与文本 Judge 均判定答案正确。因此 83.33% 是 6 个可回答问题上的严格 ID 匹配结果，不能表述为答案错误；冻结后的 Test 报告保持不变，未来评测集版本可将这些位置标注为可替代 Evidence Group。

### 性能优化结果

在相同 12 条 dev RAG 评测上，显式 Figure/Table 编号改为元数据精确定位，纯 Table 路由不再附带 Text Retrieval，CrossEncoder 精排候选由 16 个降至 8 个，并限制为 256 tokens。优化后平均端到端延迟从 21.10 s 降至 8.22 s，下降 61.1%；P95 从 45.97 s 降至 23.39 s，下降 49.1%，同时保持 dev Task Success、路由准确率和 Retrieval Recall@5 均为 100%。Trace 进一步表明，热态文本检索的主要耗时仍来自 CrossEncoder，首次访问新论文时还包含本地 Dense/Qdrant 冷启动成本。

### 复现最终评测

使用本地 Qdrant 时先停止 Uvicorn，避免两个进程同时占用本地存储。命令会将相关论文证据发送到已配置的 DeepSeek/Qwen-VL API：

```powershell
python scripts\validate_dataset.py --dataset evals\questions.v2.jsonl
python scripts\validate_dataset.py --dataset evals\questions.v3.jsonl
python scripts\evaluate.py --dataset evals\questions.v2.jsonl --split dev --mode rag --strategy hybrid-rerank --orchestrator langgraph --parallel --judge --output evals\report-dev-langgraph-final.json
# 冻结 Test 已按此命令运行一次；不要用 Test 继续调参：
python scripts\evaluate.py --dataset evals\questions.v2.jsonl --split test --mode rag --strategy hybrid-rerank --orchestrator langgraph --parallel --judge --output evals\report-test-langgraph-final.json
```

说明：20 条题目适合展示端到端工程闭环和回归能力，但不足以代表大规模统计结论。简历和项目介绍应同时写明论文数量、样本量以及 Judge 类型。

## 已知边界

- 未配置真实 DeepSeek/Qwen-VL Key 时，解析与检索可运行，但生成或视觉阅读会明确返回 unavailable；不会伪造模型结果。
- PyMuPDF 降级路径只能导出 PDF 内嵌且足够大的位图；复杂矢量 Figure 仍依赖 Docling 的区域渲染质量。
- Table VLM fallback 默认关闭；启用后会增加 Qwen‑VL 成本，结构化与视觉结果均不可用时会返回 partial/refusal。
- Router/Planner 第一版采用可测试规则，不是训练过的意图分类器。
- Critic 的跨模态冲突检查目前以结构化 Agent Result、必要模态覆盖和同名数值冲突为主，不等同于领域专家审稿。
- LangGraph Checkpoint 支持按 `run_id` 读取状态；当前 UI 尚未提供人工审批后从中断节点继续的操作界面。
- 论文级长期记忆当前使用本机 `local` 用户范围，尚未接入登录、RBAC、多租户隔离或向量化自由文本记忆。
- SSE 事件缓冲位于单进程内存，多 Worker 部署前需要迁移到 Redis Streams、NATS 等共享事件总线。
- 答案 Delta 是 Claim 核验后的安全文本切片，不是 Provider 原生 token 直出；LangGraph 与 Tool 阶段事件已经是真实执行时发布。
- Web UI 仍是轻量单页 PDF.js 阅读器，支持跳页和 bbox 高亮，但没有缩略图、连续滚动或缩放控件。
