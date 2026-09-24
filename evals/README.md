# 评测集说明

当前数据集版本如下：

- `questions.v2.jsonl`：20 条，已用于历史 Dev/Test 正式报告。
- `questions.v3.jsonl`：30 条，保留 v2 全部题目并扩充题型。
- `questions.v4.jsonl`：60 条，覆盖 5 篇论文，每篇 12 条；Dev 36 条、Test 24 条。其中 50 条可回答、10 条不可回答，覆盖文本、表格、纯视觉和跨模态问题。
- `questions.v4.additions.jsonl`：v4 相对 v3 新增的 30 条，便于单独审阅和追踪。

`questions.v4.jsonl` 已通过本地结构、Evidence ID、页码和原文引用的严格校验。36 条 Dev 已完成完整 LangGraph RAG + Judge 评测，最终报告为 `report-dev-v4-final.json`，修复前对照为 `report-dev-v4-baseline.json`。24 条 v4 Test 尚未运行，不能把 Dev 结果表述成 Test 成绩。完整格式参考 `questions.example.jsonl`。

先按页码、章节或关键词查看可标注证据：

```powershell
python scripts\annotate.py --paper-id <ID> --page 6
python scripts\annotate.py --paper-id <ID> --section Method --keyword attention
python scripts\annotate.py --paper-id <ID> --unit chunk --page 6 --jsonl
```

写入或修改评测集后，先执行严格校验：

```powershell
python scripts\validate_dataset.py --dataset evals\questions.v4.jsonl --json
# 历史数据集需要复现时再单独校验：
python scripts\validate_dataset.py --dataset evals\questions.v2.jsonl --json
```

校验器会检查重复 case、字段类型、论文处理状态、Chunk/Sentence 归属、页码一致性以及 `gold_quotes` 是否确实来自已标注句子。可回答问题必须同时标注 Chunk、Sentence 和原文；无答案问题的 Gold Evidence 必须为空。同一篇论文不能同时出现在 `dev` 和 `test`，避免论文内容泄漏。

默认使用 `expected_chunk_ids` 表示一组完整证据。如果论文中有多个位置都能独立回答同一问题，可增加可替代证据组；`expected_chunk_ids` 必须等于所有组的并集：

```json
{
  "expected_chunk_ids": ["chunk-a", "chunk-b"],
  "expected_chunk_groups": [["chunk-a"], ["chunk-b"]]
}
```

句子级引用同样支持可替代证据组。`expected_sentence_ids` 和 `expected_evidence_ids` 分别必须等于对应分组的并集：

```json
{
  "expected_sentence_ids": ["page4-s1", "page4-s2", "page5-s1", "page5-s2"],
  "expected_sentence_groups": [["page4-s1", "page4-s2"], ["page5-s1", "page5-s2"]],
  "expected_evidence_ids": ["page4-s1", "page4-s2", "page5-s1", "page5-s2", "figure-2"],
  "expected_evidence_groups": [["page4-s1", "page4-s2"], ["page5-s1", "page5-s2"], ["figure-2"]]
}
```

视觉专属题仍用图注 Sentence 标注页面与来源位置，但标准答案必须来自原图，统一 Gold 使用 `expected_figure_ids` 和 Figure `expected_evidence_ids`。

运行检索评测：

```powershell
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy hybrid-rerank --limit 5 --output evals\report-v4-retrieval.json
# 中文问题检索英文论文时，同时评测英文查询改写：
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy hybrid-rerank --limit 5 --rewrite --output evals\report-v4-retrieval-rewrite.json
```

评测默认只读取 `dev`。只有冻结配置后才显式运行 `--split test`；`--split all` 仅用于汇总。脚本会严格检查所选策略依赖的模型与 Qdrant，失败时直接终止，不会生成静默降级的误导报告。

四组消融实验：

```powershell
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy bm25 --limit 5 --output evals\report-v4-dev-bm25.json
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy dense --limit 5 --output evals\report-v4-dev-dense.json
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy hybrid --limit 5 --output evals\report-v4-dev-hybrid.json
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --strategy hybrid-rerank --limit 5 --output evals\report-v4-dev-hybrid-rerank.json
```

脚本会统计 Citation Hit@K、完整证据组命中率、Recall@K、MRR、nDCG@K、分论文指标，以及冷启动、热查询平均、平均/P95/最大延迟。无答案问题不会混入检索排名指标。`expected_answer` 留给人工或另行的 LLM-as-judge 流程评分，避免把自动相似度误当作回答正确率。

使用本地 Qdrant 时，运行评测前必须停止 Uvicorn；需要并发访问时应启动 Qdrant Server 并配置 `QDRANT_URL`。

建议统计：Citation Hit@5、完整证据组命中率、回答正确率、拒答准确率和热查询平均延迟。先在 `dev` 上比较四种检索策略，再冻结配置并运行 `test`。

完整 RAG 评测会执行与网页一致的 Query Router。复杂文本题进入 Text Agentic RAG；Figure/Table 题进入 Multimodal Agentic RAG，并记录 Agent Trace、工具调用、最终 text/figure/table Evidence、Token 用量及 Qwen-VL 调用次数；简单问题继续使用 Standard RAG。所有路径均执行回答、证据审核和结构化拒答：

```powershell
python scripts\evaluate.py --dataset evals\questions.v4.jsonl --split dev --mode rag --strategy hybrid-rerank --orchestrator langgraph --parallel --judge --output evals\report-dev-v4-final.json
```

增加 `--judge` 后，文本题使用标准答案、Gold 原文和回答实际引用的原始证据进行 0–2 分裁判；被实际引用证据直接支持的相关补充细节不会仅因标准答案较短而扣分。带 `visual-only` 标签的题会改由独立 Qwen-VL 重新查看 Gold Figure，不信任回答链路中已有的视觉分析。视觉 Judge 会分别判断 Gold 和候选答案是否受原图支持；Gold 与原图冲突时输出 `dataset_issue=true`，且 `answer_correct/task_success` 为 `null`，该样本不会被错误计入答案失败。该分数带有模型主观性，应与确定性的 answerability、refusal 和 gold citation 指标分开报告。
调试时可增加 `--case-id <CASE_ID>` 只运行一题，避免一次产生大量模型调用。

RAG 报告中的三个成功字段含义不同：`execution_success` 仅表示链路正常完成；`answer_correct` 只在启用 `--judge` 后给出；`task_success` 是两者同时为真。`answer_evaluated_count` 是已得到语义裁判的数量，`answer_correct_count` 才是其中正确的数量。未启用 Judge 时后两项为 `null`。`modality_routing_accuracy` 比较 Gold 模态与 Router 的实际输出 `routed_modalities`，不受后续拒答或最终 Citation 是否为空影响。Text/Figure/Table 的候选排名分别写入 `ranked_evidence_ids_by_modality`，Recall@K 与 MRR 在各自模态的 Top-K 内计算，不再按工具调用顺序拼成一个跨模态列表。文本 Retrieval 使用 `expected_chunk_ids` 对比检索返回的 Chunk ID；最终 Citation/Evidence 指标仍使用 Sentence ID，避免两个层级直接比较产生假阴性。

当前冻结 v4 Dev 摘要：36/36 Task Success，Answerability、Refusal 和 Modality Routing Accuracy 均为 100%，Gold Citation Hit 为 100%，Complete Gold Citation Hit 为 93.33%，平均/P95 延迟为 10.53 s / 36.67 s。完整指标以 JSON 报告为准；该结果来自固定 5 篇论文的 Dev 集，不代表任意论文上的普遍准确率。
