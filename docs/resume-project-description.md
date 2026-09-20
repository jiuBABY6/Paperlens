# PaperLens 4.0：简历项目描述

项目名称建议写为：**PaperLens 4.0——基于 LangGraph 的多智能体多模态论文 RAG 系统**

技术栈：Python、FastAPI、LangGraph、Docling、PyMuPDF、SQLite、Qdrant、BGE-M3、BGE Reranker、DeepSeek、Qwen-VL、PDF.js、Pytest。

## 可直接使用的 3 条描述

- 将单 Agentic RAG 重构为 LangGraph Supervisor 多智能体系统，编排 Text、Figure、Table Specialist Agents，通过 Shared Evidence Memory、Evidence Critic 与 Answer Agent 完成跨模态证据合并、冲突检查、一次定向返工和结构化拒答；保留 Sentence/Figure/Table 引用及 PDF BBox 高亮。

- 建立 5 篇论文、20 条冻结 v2 Benchmark（论文级 Dev/Test 隔离）并扩展 30 条 v3 集合；区分执行成功、答案正确和严格证据命中，使用 DeepSeek 文本 Judge 与独立 Qwen‑VL Visual Judge。冻结配置下 LangGraph Dev 12/12、Test 8/8 Task Success，Test 路由与拒答准确率 100%。

- 实现 API 有限重试、节点超时、工具/模型/Qwen‑VL 预算、SQLite Checkpoint 和 6 类故障注入测试；对无依赖 Specialist 安全并行，在质量指标不变的前提下，相比 LangGraph 串行将 Dev 平均延迟降低 3.7%、P95 降低 14.1%，全量测试 108 项通过。

## 表述边界

- “答案正确率”来自 DeepSeek 文本 Judge 与独立 Qwen-VL Visual Judge，不应写成人工专家评审结果。
- LangGraph Test 的严格 Gold Citation Hit 为 5/6；未命中的一题由其他直接支持句回答且 Judge 判定正确，因此简历应分别报告 Task Success 与严格证据命中，不能把后者写成答案正确率。
- 冻结主对照仍是 5 篇论文、20 条 v2 问题；v3 的 30 条问题已完成标注校验但尚未作为最终指标运行，不代表大规模学术基准成绩。
