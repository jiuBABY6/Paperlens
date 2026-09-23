# PaperLens 6.0 对话式多智能体升级实施记录

## 完成状态

升级按技术方案阶段 0–7 实施完成。业务库采用增量迁移，没有删除或重建论文、Chunk、Sentence、Figure/Table 与 Qdrant 数据。升级前备份为 `data/paperlens.pre-conversation-upgrade.20260921.sqlite3`，当前库与备份的 `PRAGMA integrity_check` 均为 `ok`。

## 阶段结果

| 阶段 | 结果 | 验证 |
|---|---|---|
| 0 基线 | 原系统测试与数据基线冻结 | 108 passed；5 篇论文、1271 Chunk、3171 Sentence、53 Figure/Table |
| 1 数据与 API | Conversation、Message、Run 表和归属/幂等/互斥约束 | Repository/API 测试通过；SQLite WAL、foreign keys、busy timeout |
| 2 三栏页面 | 历史对话、阅读卡片、当前聊天；Evidence PDF 抽屉 | 静态契约测试；模型和论文文本统一转义 |
| 3 Context/Memory | 最近 6 轮、摘要、引用 ID、Query Resolver 与澄清降级 | 首轮免模型、追问改写、失败回退、会话隔离测试 |
| 4 SSE/取消 | 后台 Run、事件缓冲、心跳、重放、终态恢复、协作式取消 | 顺序、Last-Event-ID、终态恢复、Paper/Conversation 归属测试 |
| 5 答案增量 | 已验证答案按 Delta 推送，完成后一次性落库 | RunManager 与端到端工作流测试 |
| 6 Function Calling | Specialist 白名单 Tool Schema、参数校验、预算、去重和 fixed 回退 | 越权、附加参数、重复调用、视觉结果回填、完成条件测试 |
| 7 回归与评测 | README、Dev/Test、新文档与最终回归 | 142 项测试；Dev 12/12；冻结 Test 8/8 |
| 8 论文级长期记忆 | 跨会话已验证问答、章节/图表进度、未解决问题、控制意图与清空 API | CRUD、论文隔离、跨会话读取和污染防护测试 |
| 9 实时编排事件 | Router、LangGraph Node、Specialist、Function Call、Tool、Repair 与核验事件 | 执行边界事件和前端状态测试 |
| 10 Function Calling 生产化 | strict Schema、本地二次校验、结果信封、输出上限、阈值熔断与冷却恢复 | 越界拒绝、信封、事件和熔断恢复测试 |

## 最终配置与报告

- `AGENT_ORCHESTRATOR=langgraph`
- `SPECIALIST_EXECUTION_MODE=function_calling_with_fallback`
- `ENABLE_CONVERSATIONS=true`
- `ENABLE_QUERY_RESOLVER=true`
- `ENABLE_SSE=true`
- Dev：`evals/report-dev-conversation-upgrade-final.json`
- Test：`evals/report-test-conversation-upgrade-final.json`
- 多轮功能规格：`evals/conversations.smoke.jsonl`（非正式 Benchmark）

## 已知实现边界

- SSE 是单进程内存缓冲；多 Worker 或多实例部署前需要 Redis/NATS 等事件总线。
- 取消是协作式的，正在执行的同步 HTTP/模型请求不能被 Python 立即强制中断，只能在返回后的安全边界停止。
- 页面展示的是对已完成、已验证答案的应用层增量切片；底层 Claim 生成与验证仍是同步模型调用，因此首个答案 Delta 前可能存在等待。
- Query Resolver 仅覆盖一篇论文内的指代、省略和纠正；不支持跨论文对话、用户画像或历史消息向量检索。
- SQLite 适合当前单机作品集与小规模并发，不宣称支持高并发多租户生产部署。
- Function Calling 依赖模型协议稳定性；连续失败达到阈值后熔断，冷却到期自动试探，并始终保留 fixed fallback。
- 论文级长期记忆当前是本机 `local` 用户范围；多用户部署前需接入认证与租户隔离。
