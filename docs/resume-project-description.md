# PaperLens：简历项目描述

推荐项目名称：**PaperLens｜基于 LangGraph 的多智能体多模态 RAG**

技术栈：Python、FastAPI、LangGraph、Docling、PyMuPDF、SQLite、Qdrant、BGE-M3、BGE Reranker、DeepSeek、Qwen-VL、OpenTelemetry、Prometheus、Tempo、Grafana、Docker、Pytest。

## 简历正文

- 独立完成面向学术论文深度阅读的多智能体 RAG 系统，基于 LangGraph 编排 Supervisor、Text/Figure/Table Specialists、Evidence Critic 与 Answer Agent；支持跨模态任务拆解、证据合并、定向补证和结构化拒答，并将 Sentence/Figure/Table 引用定位到 PDF 页码与 BBox。

- 建立可追溯的对话式 AI 应用链路：通过受限 Function Calling 动态选择检索与读取工具，增加严格 Schema、本地参数校验、调用预算、熔断和固定流程回退；实现 SSE 工作流事件、持久化多轮会话及带来源、版本、重要性、过期和主动遗忘能力的论文级长期记忆。

- 构建覆盖 5 篇论文、60 条问题的多模态 Benchmark；在已运行的 36 条 v4 Dev 上取得 36/36 Task Success，Answerability、Refusal 和 Modality Routing Accuracy 均为 100%，Gold Citation Hit 为 100%。完成 177 项自动化回归测试，并将平均/P95 延迟相对同集基线降低 18.93%/11.32%。

- 接入 OpenTelemetry、Prometheus、Tempo 与 Grafana，统一观测 HTTP、LangGraph 节点、Tool、DeepSeek/Qwen-VL、Token、Fallback、熔断和记忆生命周期；支持通过 `run_id` 从页面回答定位完整 Trace，形成从功能、评测到故障排查的工程闭环。

## 30 秒项目介绍

PaperLens 是一个面向科研论文阅读的多模态 RAG 系统。它会把论文正文、图片和表格解析成可检索、可定位的统一证据，再由 Router 将简单文本事实送入 Standard RAG，将图表和跨模态问题送入 LangGraph 多智能体工作流。系统不只生成答案，还会校验证据、返回原文引用，在证据不足时拒答；同时具备 Function Calling、SSE 多轮对话、论文级长期记忆和完整 LLMOps 可观测能力。

## 指标表述边界

- “36/36”是 5 篇论文上 v4 Dev 的本次冻结运行结果，不是生产场景准确率，也不是 v4 Test 成绩。
- 答案正确性由 DeepSeek 文本 Judge 和独立 Qwen-VL Visual Judge 判定，不应描述为人工专家评审。
- Complete Gold Citation Hit 为 93.33%；两条跨模态题引用了语义等价正文而非全部固定 Gold ID，因此答案和 Citation Hit 通过，但严格完整 ID 未满分。
- v4 Test 有 24 条且尚未运行。早期 v2 曾按冻结配置完成 8 条 Test，简历通常无需同时堆叠两代数据；面试追问时再解释版本关系。
- 当前是单机作品集部署：SQLite、嵌入式 Qdrant、进程内 SSE 和 `local` user scope 不能包装成高并发、多租户生产集群。
