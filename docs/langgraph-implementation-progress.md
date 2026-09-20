# PaperLens LangGraph 实施记录

## Phase 0：冻结基线

- 项目路径：`D:\Desktop\job\demo`
- Benchmark：`questions.v2.jsonl`，20 条，Dev 12 / Test 8，5 篇论文。
- 数据集校验：`rows=20 errors=0 warnings=0`。
- 基线测试：`88 passed`。
- Python：3.11.15。
- LangGraph：1.2.11。
- LangGraph SQLite Checkpoint：3.1.1。
- LangChain Core：1.6.3。
- 保留的旧报告：
  - `report-dev-rag-p1.json`
  - `report-dev-rag-optimized-cold.json`
  - `report-test-final.json`

说明：当前目录没有 Git 仓库；用户已在外部复制项目作为备份。本次实施不覆盖旧 Executor、v2 Benchmark 和上述报告。

## 阶段状态

- [x] Phase 0：冻结基线
- [x] Phase 1：LangGraph 骨架（91 passed）
- [x] Phase 2：Specialist Agents（92 passed）
- [x] Phase 3：Supervisor 与复杂任务（95 passed）
- [x] Phase 4：Critic 与返工（99 passed）
- [x] Phase 5：异常恢复与 Checkpoint（105 passed）
- [x] Phase 6：并行与性能（107 passed）
- [x] Phase 7：评测与交付（108 passed）

## Phase 7 当前结果

- v2/v3 数据校验：`20/30 rows, 0 errors, 0 warnings`。
- LangGraph 真实视觉冒烟：通过，节点 Trace 与独立视觉 Judge 均正常。
- 冻结 v2 Dev：12/12 Task Success，Recall@5 1.000，Evidence F1 0.963。
- 串/并行对照：质量一致；并行平均延迟下降 3.7%，P95 下降 14.1%。
- 恢复测试：6 类故障注入均通过。
- README、架构图、简历描述、消融报告：已完成。
- 冻结 v2 Test：8/8 Execution Success、Answer Correct 与 Task Success；Recall@5/MRR 0.833/0.833，Evidence F1 0.833，路由与拒答准确率 100%。
- Test 仅运行一次，未根据结果调参；唯一严格 Gold 未命中案例由其他直接支持句回答，文本 Judge 判定正确。
- 最终报告：`evals/report-test-langgraph-final.json`。
- 最终全量回归：`108 passed in 5.53s`。
