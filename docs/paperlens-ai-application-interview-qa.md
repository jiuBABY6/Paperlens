# PaperLens AI 应用开发技术面试题库

本文包含 60 道与 PaperLens 实际实现一致的技术面试题。回答可作为复习提纲，但建议先理解数据流、关键取舍和边界，再用自己的语言表达。项目中的 MCP 仅存在于独立的 `mcp_demo` 学习项目，PaperLens 主系统没有接入 MCP，面试时不要混为一谈。

## 一、项目定位与整体架构

### Q1：请用一分钟介绍 PaperLens。

PaperLens 是一个面向科研论文阅读的多模态 Agentic RAG 系统。用户上传 PDF 后，系统解析正文、图片和表格，建立带页码、章节和坐标的证据索引；提问时由路由器判断问题类型，简单文本事实走 Standard RAG，图片、表格或跨模态问题进入 LangGraph 编排的多智能体工作流。最终答案必须携带可回到原 PDF 的引用，并经过证据校验；系统还提供多轮会话、论文级跨会话记忆、Function Calling、SSE 事件和基于 Prometheus、Tempo、Grafana 的 LLMOps 可观测能力。

### Q2：这个项目解决了什么实际问题？

普通 PDF 问答往往只返回一段无法核验的文本，而且难以理解图表。PaperLens 解决三类问题：一是把文本答案定位到原论文页码和 bbox，便于核查；二是让 Figure、Table 也成为可以搜索和读取的证据；三是对复杂问题按模态拆解并校验，证据不足时拒答，而不是强行生成。它更接近“可追溯的论文阅读助手”，而不是通用聊天机器人。

### Q3：为什么它属于 RAG？

回答前会先从论文知识库检索证据，再把命中的文本、图片分析或表格内容交给模型生成答案，答案引用检索结果，因此符合 Retrieval-Augmented Generation。系统不是把整篇论文直接塞给模型，而是通过 BM25、Dense、RRF 和 CrossEncoder 找到相关证据，以降低上下文长度并提升可追溯性。

### Q4：为什么又称为 Agentic RAG？

传统 RAG 的“检索一次—生成一次”流程基本固定。PaperLens 会根据问题动态选择文本、图片、表格工具，复杂问题还会规划子任务、派发 Specialist、检查证据是否齐全，并在缺失时定向修复或拒答。模型与编排层不仅生成答案，还参与工具选择和任务推进，所以属于 Agentic RAG。

### Q5：什么是多智能体 RAG 工作流？

这里的“多智能体”不是多个独立聊天机器人自由讨论，而是多个职责明确的执行角色共享同一个有类型状态：Supervisor 规划任务，Text/Figure/Table Specialist 分别获取对应模态证据，Evidence Critic 检查证据覆盖，Answer Agent 合成答案。LangGraph 用显式节点、边和状态把协作过程固定下来，使每一步都能测试和追踪。

### Q6：为什么同时保留 Standard RAG 和多智能体链路？

简单文本事实题通常只需一次文本检索和生成，如果也走完整多智能体图，会增加模型调用、Token 和延迟。Router 因此把它们送到 Standard RAG；只有纯图片、纯表格或跨模态复杂问题才进入 Agentic RAG。双路径是质量、成本和响应时间之间的工程折中，而不是所有问题都堆叠最多组件。

## 二、论文解析、索引与存储

### Q7：论文上传后经历哪些步骤？

系统先校验扩展名、PDF 文件头和大小，流式写入临时文件并计算 SHA-256。根据内容哈希判断复用、处理中或新论文；新论文保存为 `source.pdf` 后执行结构解析、Figure 离线理解、混合检索索引、阅读卡片生成，最后把论文元数据和结构化结果写入 SQLite。任何阶段失败都会更新处理状态，避免把半成品标成完成。

### Q8：为什么需要文件哈希和文档注册表？

文件名不可靠，同一篇论文可能被改名重复上传。SHA-256 可以按内容去重，文档注册表记录 `processing/completed/failed` 状态，使并发上传能够复用已完成结果或返回正在处理，而不是重复解析、重复调用视觉模型和重复建库。

### Q9：正文是怎样解析的？

主路径用 Docling 获取章节、段落、Figure 和 Table 等结构；同时用 PyMuPDF 提取页码、文本块、句子及坐标。系统将结构信息与 PDF 坐标对齐，形成带 `page、section、bbox` 的 Sentence 和 Chunk。Docling 失败时可降级到 PyMuPDF，但复杂版面和矢量图质量会受限。

### Q10：为什么同时使用 Docling 和 PyMuPDF？

Docling 擅长文档结构理解，能区分章节和图表；PyMuPDF 擅长稳定读取 PDF 页、文本坐标和渲染区域。仅用 Docling 不容易保证前端精确高亮，仅用 PyMuPDF 又缺少可靠的语义结构。二者组合分别承担“理解结构”和“定位原文”。

### Q11：Figure 是怎样处理并进入知识库的？

解析阶段保存 Figure 裁剪图、标题、附近正文、页码、章节和关联 Sentence。可读取的图片由 Qwen-VL 离线生成摘要、实体、关系、关键词和不确定性，这些文本化描述与元数据一起进入检索表示；原始图片仍保留，遇到必须观察颜色、位置或连线的问题时，再通过 `read_figure` 和 `analyze_figure_for_query` 做查询相关的视觉分析。

### Q12：为什么 Figure 既要离线理解，又可能在线分析？

离线理解只做一次，适合粗粒度召回，能避免每个问题都调用视觉模型；但通用摘要不一定包含颜色、相对位置等细节。在线分析根据具体问题查看目标 Figure，准确性更高但成本和延迟更大。两阶段方案用离线描述解决检索，用在线视觉解决细粒度问答。

### Q13：Table 是怎样处理的？

系统保留 Table 的标题、页码、章节、bbox、结构化 Markdown 或单元格文本以及表格图片。检索时使用标题、表头、单元格和附近文本；命中后由 `read_table` 返回结构化内容。只有结构化表格不可用且功能开关允许时，才使用视觉模型分析表格图片，避免不必要的 Qwen-VL 调用。

### Q14：文本、图片和表格是否分别建立三套完全独立的向量库？

它们在逻辑上保留不同的模态字段、工具和 Top-K 口径，但共享论文级存储与统一证据契约。文本向量主要来自原文，Figure 和 Table 则将标题、附近正文、离线视觉描述或结构化内容文本化后参与检索。这样既能统一召回接口，又不会丢失原始模态和定位信息。

### Q15：项目使用了哪些数据库，各自负责什么？

SQLite 保存论文元数据、句子、图表、会话、消息、Run、Trace、缓存和长期记忆等事务数据；Qdrant 保存 Dense 向量并执行相似度检索；文件系统保存原 PDF、Figure/Table 图片和模型文件。Prometheus 保存时间序列指标，Tempo 保存分布式 Trace，它们属于可观测系统而不是业务知识库。

## 三、检索、路由与证据

### Q16：什么是 lexical query？

Lexical query 是面向词面匹配的检索查询，重点保留原问题中的论文术语、模型名、Figure/Table 编号和关键实体，用 BM25 查找包含相同或相近词项的内容。它与 Dense 的语义查询互补，尤其适合专有名词、编号和精确数值。

### Q17：Dense 检索是什么？

Dense 检索把查询和证据编码为稠密向量，再按向量距离寻找语义相似内容。PaperLens 使用 BGE-M3 生成向量、Qdrant 存储和搜索，因此即使问题与论文用词不完全一致，也可能找到语义相关段落。它的缺点是冷启动和计算成本更高，并可能弱化精确编号匹配。

### Q18：项目中的混合检索如何工作？

系统并行获得 BM25 词法结果与 Dense 语义结果，用 Reciprocal Rank Fusion 按名次融合，随后用 CrossEncoder 对候选进行更精细的查询—证据相关性评分。最终保留 Top-K。BM25 保证专有词和编号，Dense 提供语义召回，Reranker 提升排序质量。

### Q19：为什么不只使用向量检索？

论文问题常包含 `Figure 2`、`Table 1`、模型名和具体缩写，这些词面信号非常强。只用向量检索可能召回语义类似但编号错误的对象。PaperLens 对显式 Figure/Table 编号先做元数据精确过滤，再结合混合检索，减少无关图片分析和视觉模型调用。

### Q20：跨模态 Top-K 为什么要分别计算？

文本、Figure 和 Table 的候选规模、分数分布和 Gold 证据不同。如果把所有模态混在一个总 Top-K 中，文本候选可能挤掉正确图表，造成 Recall 假阴性。项目按模态计算候选与 Recall，再在答案阶段合并证据，评测口径与实际路由保持一致。

### Q21：Router 如何判断问题走哪条链路？

Router 使用可测试的规则识别显式 Figure/Table 引用、视觉属性词、表格比较词、问题复杂度和所需模态。简单且仅需文本证据的问题走 `standard_rag`；纯 Figure、纯 Table 或文本与图表联合问题走 `agentic_rag`，并附带所需模态。它目前不是训练出的意图分类器，优势是稳定、可解释、可回归。

### Q22：Router、Supervisor 和 Planner 如何协作？

Router 做粗粒度入口决策，确定 Standard 还是 Agentic 以及大致模态。进入 LangGraph 后，Supervisor 中的确定性 Planner 将原问题转成带模态、Agent、预算和执行方式的子任务。Specialist 执行后，Supervisor 通过 Critic 结果决定继续回答、定向补证或拒答。Router 不负责执行细节，Planner 不重新决定整个系统入口。

### Q23：一个跨模态问题具体如何处理？

例如“结合 Figure 2 和 Table 1 解释设计与结果关系”，Router 标记需要 figure、table，通常还需要 text。Planner 生成多个有类型子任务，Dispatcher 可并行调用 Figure、Table、Text Specialist；Evidence reducer 合并并去重证据，Critic 检查三种模态是否覆盖和是否存在明显冲突，最后 Answer Agent 基于通过的证据合成带引用回答。

### Q24：什么是统一证据契约？

所有 Specialist 不直接把任意字符串塞给答案模型，而是返回包含 `evidence_id、modality、paper_id、page、section、content、bbox/asset` 等字段的标准 Evidence。统一契约使去重、引用、前端跳转、Critic 校验、离线评测和 Trace 记录都能复用同一套结构，减少不同模态之间的胶水代码。

## 四、LangGraph 与多智能体编排

### Q25：当前多智能体一共有多少种角色？

核心职责包括 Supervisor/Planner、Text Specialist、Figure Specialist、Table Specialist、Evidence Critic 和 Answer Agent。严格说并非每个角色都是独立大模型实例：Supervisor 规划、Critic 的部分检查和图路由包含确定性逻辑；“Agent”强调职责和输入输出边界，而不是为了数量而拆分模型。

### Q26：项目是 Plan-Execute 还是 Supervisor–Worker？

更准确地说是带确定性 Planner 的 Supervisor–Worker 工作流。Supervisor 先产生结构化子任务，多个 Specialist Worker 执行，Critic 将结果反馈给 Supervisor 决定是否修复，最后 Answer Agent 输出。它借用了 Plan-Execute 的“先规划再执行”，但主控制模式是 Supervisor 管理专门 Worker，而非一个开放式长期计划循环。

### Q27：LangGraph 图中有哪些主要节点？

主要节点是 `plan`、`dispatch_specialists`、`evidence_critic`、`repair_dispatch`、`answer` 和 `structured_refusal`。正常路径为规划、派发、校验、回答；证据缺失进入定向修复，仍无法满足条件则结构化拒答。节点和条件边让控制流显式化，也便于逐节点记录延迟与状态。

### Q28：为什么用 LangGraph，而不是普通 if/else？

小流程当然可以写成 if/else，但当系统需要共享状态、并行 Specialist、条件分支、修复循环、预算、Checkpoint 和事件流时，普通流程会迅速变得难以测试。LangGraph 将节点、边和 State 明确建模，便于增加失败恢复和可观测性。选择它的价值在于状态化编排，而不是只为了在简历中出现框架名称。

### Q29：LangGraph State 中保存什么？

State 包含原问题、解析后的所需模态、计划与子任务、各 Specialist 结果、合并后的 Evidence、预算与重试次数、Critic 结论、答案状态、引用、Trace 和错误信息。节点通过受控字段读写状态，避免依赖隐式全局变量。

### Q30：Reducer 有什么作用？

多个 Specialist 并行返回证据时，普通赋值会互相覆盖。Evidence reducer 按 `evidence_id` 合并列表并去重，使 Text、Figure、Table 的结果安全汇聚到同一 State。它解决的是并发分支状态合并问题，不负责判断证据是否正确；正确性由 Critic 和 Judge 处理。

### Q31：Specialist 为什么按文本、图片、表格划分？

三种模态的检索信号和读取方式不同：文本需要句子与 bbox，Figure 需要图片和视觉问题，Table 优先读取结构化单元格。分开后，每个 Specialist 只有一组最小工具和独立预算，能够减少误调用，也便于分别统计命中率、延迟和失败原因。

### Q32：多 Agent 会不会拆得太多？

如果每个小步骤都做成自由决策 Agent，确实会增加成本和不确定性。PaperLens 只把模态能力与答案职责拆开，路由、预算、Reducer 和大部分 Critic 条件保持确定性。后续若维护成本高，可以把 Specialist 收敛为一个 Agent 加多组 Skill/Tool，但目前的划分对展示跨模态并行、独立预算和可观测性有实际价值。

### Q33：Specialist 可以并行执行吗？

可以。对彼此没有依赖的 text、figure、table 子任务，开启 `MULTI_AGENT_PARALLEL_ENABLED` 后可并发运行，再由 Reducer 合并结果。冻结 Dev 对比中并行与串行质量相同，平均延迟方向性下降；但报告缓存状态和生成时间不同，因此不能把差值解释为严格因果结论。

### Q34：Evidence Critic 做什么，和答案 Judge 有什么区别？

Evidence Critic 在生成前检查任务所需模态是否齐全、结果状态是否成功、是否有明显数值冲突，并决定回答、补证或拒答。Judge 用于离线评测或独立视觉核验，判断最终答案是否语义正确。前者是线上工作流控制，后者是质量评估，不能用“流程执行成功”代替“答案正确”。

## 五、Tool Use、Function Calling 与异常恢复

### Q35：项目中的 Tool Use 是怎样实现的？

系统把检索和读取能力封装为可审计工具，例如 `search_text/read_sentence`、`search_figures/read_figure/analyze_figure_for_query`、`search_tables/read_table`。每个工具有明确参数和返回 Evidence，Specialist 只能访问自己的工具集合。调用过程写入 Agent Trace 和 OpenTelemetry Event，前端也能看到执行顺序。

### Q36：Function Calling 与原来的固定工具流程有什么区别？

固定流程由代码预先规定“先 search 再 read”，稳定但不能根据中间结果动态调整。Function Calling 将允许的工具 Schema 提供给 DeepSeek，由模型返回工具名和结构化参数，执行结果再回传给模型决定下一轮。PaperLens 同时保留固定流程作为回退，因此获得动态选择能力，但不会把可靠性完全交给模型。

### Q37：这是企业常见的 Function Calling 形式吗？

核心形式是一致的：工具 JSON Schema、模型产生 tool call、服务端校验参数、执行本地函数、返回 tool result，再继续模型循环。项目还增加白名单、结果信封、轮次与调用预算、熔断和指标。与大型生产系统的差距主要在分布式限流、权限审计、持久化队列和多租户治理。

### Q38：如何避免模型调用不存在的工具或传入危险参数？

每个 Specialist 只注册白名单工具；参数必须通过严格 Schema 和本地语义校验，论文 ID 等上下文由服务端约束，模型不能任意访问文件或执行代码。无效工具名、字段缺失、越权或重复失败会被拒绝并计数，不会直接执行模型提供的任意字符串。

### Q39：Function Calling 失败时如何恢复？

协议解析失败、空 Evidence、超出预算或提供商错误时，系统记录失败原因并进入固定流程 fallback。连续失败达到阈值后熔断，暂时跳过模型工具选择，直接走可预测的固定流程；恢复窗口后再试探。这样将动态能力作为增强，而不是单点故障。

### Q40：为什么 Grafana 中会看到 `function_calling_fallback`？

这表示某个 Specialist 的原生 Function Calling 没有产出可用证据，系统改用固定工具顺序完成任务。Fallback 不等于整次问答失败，需要结合最终 run outcome 和 Evidence 判断。面板使用进程生命周期累计值，避免低频事件因时间窗口首样本问题显示成假零。

### Q41：项目做了哪些异常恢复？

包括解析主路径失败后的 PyMuPDF 降级、Function Calling 固定流程回退与熔断、模型不可用时的结构化 partial/refusal、缺证据时 Critic 定向补检索、并发节点结果隔离、重复上传幂等复用、Run 取消和失败状态持久化。恢复的原则是显式降级并保留原因，而不是悄悄生成无证据答案。

### Q42：PaperLens 使用 MCP 了吗？

PaperLens 主项目没有接入 MCP，它使用应用内 Tool Registry 和模型 Function Calling。另有一个独立 `mcp_demo` 用于理解 MCP Server、资源或工具协议，但不能把它描述成 PaperLens 的生产链路。若未来需要让外部客户端复用论文检索工具，可以在现有 Tool 层之上增加 MCP 适配器。

## 六、对话、SSE 与长期记忆

### Q43：系统如何支持多轮对话？

每篇论文可以创建 Conversation，用户消息、助手消息和每轮 Agent Run 都持久化到 SQLite。新问题会带上 `conversation_id`，Query Resolver 读取当前会话近期消息，解析“它、这个方法、上一张表”等指代，再把解析后的独立查询送入 Router。不同会话不会共享短期历史。

### Q44：短期上下文和长期记忆有什么区别？

短期上下文是当前 Conversation 内最近若干轮消息，用于指代消解和连续追问；长期记忆按 `paper_id + user_scope` 跨会话保存已经可靠回答过的主题、章节与证据来源，用于总结学习进度。短期上下文保留原对话语义，长期记忆是经过状态筛选和聚合的派生数据。

### Q45：什么内容能进入长期记忆？

只有状态为 `completed`、`answerable=true` 且至少有一个 Citation 的回答才能成为 `active` 记忆。partial、拒答或无引用结果进入 `unresolved`，不会被当成可靠事实。记忆保存来源 Run、Evidence ID、章节、解析版本、置信度、重要性和到期时间，能够回溯来源。

### Q46：长期记忆如何处理过期、遗忘和论文重解析？

普通记忆到期后转为 `archived`；用户可以主动将条目设为 `forgotten`，也能置顶、改重要性和添加备注。论文重新解析后，旧解析版本的记忆转为 `stale`，即使被置顶也不能继续作为有效学习事实，因为原 Evidence ID 或页码可能已经变化。

### Q47：长期记忆是否已经实现多用户隔离？

数据结构按 `paper_id + user_scope` 隔离，但当前没有登录和鉴权，默认 scope 是本机 `local`。所以它实现了用户范围字段和未来扩展接口，但不能声称完成真实多租户隔离。生产化还需要身份系统、RBAC、租户级加密和访问审计。

### Q48：为什么有时 `read_paper_learning_memory → 无结果`？

Agent Trace 中的“无结果”表示这次工具读取没有找到满足条件的 active 记忆，不代表调用报错。可能原因包括新论文尚无可靠完成回答、已有条目是 unresolved/stale/archived，或问题实际需要当前会话历史而不是论文级记忆。工具结果状态应区分 empty、success 和 error，不能仅凭中文“无结果”判断故障。

### Q49：SSE 是如何实现的？

后端为每个 Run 建立事件流，实时发布 run、LangGraph node、tool 和状态事件；前端通过 SSE 持续消费并更新当前回答。最终答案 Delta 是在 Claim 校验完成后进行的安全文本分片，而不是直接转发模型原生 token，因此用户能看到工作流进展，但正文可能在后段较快出现。

### Q50：为什么不直接使用模型原生 Token Streaming？

未经校验的 token 一旦发送给前端就难以撤回，可能先展示错误或无证据内容。项目优先保证 grounded answer，等证据与 Claim 校验通过后再分片输出。未来可以实现两阶段 UI：先实时显示节点和工具事件，再把受控草稿与已验证答案明确区分。

## 七、评测、可靠性与安全

### Q51：项目如何评测 RAG，而不是只看回答“像不像对”？

评测拆成执行、路由、检索、证据、答案和拒答多个层次：Execution Success 判断流程是否完成；Modality Routing 判断模态选择；Recall@K/MRR 判断检索；Evidence Precision/Recall 判断引用；独立 Judge 判断语义正确性；Answerability/Refusal 判断证据不足时是否正确拒答。分层指标能定位问题发生在哪一段。

### Q52：Dev 12/12、Test 8/8 是否等于系统准确率 100%？

不等于。它只表示在 5 篇论文、20 条人工题目的冻结数据集上，独立模型 Judge 判定 Dev 12 条和 Test 8 条任务成功。Test 的严格 Retrieval Recall@5 是 0.833，而且样本很小。正确表述是“在该冻结 Benchmark 上取得 12/12 和 8/8 Task Success”，不能外推为生产场景 100% 准确率。

### Q53：为什么“执行成功”和“答案正确”必须分开？

接口返回 200、工具完成和图走到 answer 节点，只能说明系统执行成功；模型仍可能理解错图、选错表格或生成错误结论。答案正确性需要 Gold、规则或独立 Judge 判断。混在一起会把稳定性指标误当质量指标，掩盖幻觉。

### Q54：视觉答案为什么需要独立 Judge？

如果同一个 Qwen-VL 既分析图片又自行验证 Claim，错误分析可能自洽地通过。项目的离线视觉 Judge独立比较问题、Gold 和输出；在线二次视觉检查可作为可选增强，冲突时标记 partial/视觉核验未通过。这样降低单模型自证的风险，但仍不能替代人工抽检。

### Q55：怎样避免数据泄漏和 Test 调参？

检索参数、Router 规则和提示词先在 Dev 上确定；冻结 Test 后只按固定配置运行一次，不根据 Test 结果继续调参。报告同时保留 Test 的不足，例如严格证据 Recall 0.833。若修改数据标注或系统关键逻辑，应建立新版本数据集，而不是覆盖旧报告制造不可比结果。

### Q56：系统有哪些安全与隐私措施？

上传校验文件类型、文件头和大小；工具只能访问当前论文上下文；日志与 Trace 不记录 API Key、Authorization、完整 Prompt 和论文正文，只保留长度、状态、ID 和安全预览。`.env`、业务数据、Qdrant 和本地模型目录被 Git 忽略。生产环境仍需增加鉴权、上传恶意文件扫描、网络隔离、密钥管理和数据保留策略。

## 八、LLMOps、工程化与项目边界

### Q57：PaperLens 的 LLMOps 可观测体系包含什么？

Prometheus 采集 HTTP、Run 结果、节点、工具、模型延迟与 Token、Fallback、熔断和记忆生命周期指标；OpenTelemetry 把 HTTP、对话 Run、LangGraph、工具、DeepSeek、Qwen-VL 和记忆写入串成 Trace；Tempo 保存 Trace；Grafana 提供 System、Agents & Tools、Models & Memory 三张 Dashboard。结构化日志使用相同关联 ID 排障。

### Q58：如何把某次页面回答与 Tempo Trace 精确对应？

每轮 Run 都有 `run_id`，根 Span `paperlens.conversation_run` 保存 `run_id、conversation_id、paper_id、question_preview`。在 Tempo 使用 `{ resource.service.name = "paperlens" && span.run_id = "<run_id>" }` 即可定位该轮，再查看其子 Span 和 Event。旧 Trace 不会被回填这些新字段，必须用重启后的新请求验收。

### Q59：Grafana 三张 Dashboard 分别看什么？

System Overview 看吞吐、HTTP 失败、活跃 Run、端到端 P95、结果状态和模型 P95；Agents & Tools 看工具调用、LangGraph 节点 P95、Function Calling fallback、拒绝调用和熔断状态；Models & Memory 看模型请求、Token、记忆生命周期和读写操作。`/5m` 面板使用 `increase` 表示五分钟实际增量，低频累计故障则显示进程生命周期总数。

### Q60：当前项目最值得继续升级的地方是什么？

第一是扩大跨论文、多轮和视觉题 Benchmark，并加入人工抽检与 CI 质量门禁；第二是真正的身份认证、多租户记忆隔离和权限审计；第三是将进程内 SSE 事件迁移到 Redis Streams 或消息队列，支持多 Worker；第四是加入限流、成本预算、集中日志和告警。当前版本已能展示完整 AI 应用工程闭环，但不应声称已经达到大规模生产系统水平。

## 使用建议

面试时不要逐字背诵。建议每题先说结论，再结合一次真实链路解释取舍，最后主动说明边界。例如谈 SSE 时明确“节点与 Tool 事件实时、答案是校验后分片”；谈评测时明确“冻结 20 条 Benchmark 的 Task Success，不代表生产准确率 100%”。这种表达比堆叠框架名称更能体现工程判断。
