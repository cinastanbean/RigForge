# Handoff（Agent）


## 当前状态

1. 项目聚焦装机 Agent 主线。
2. 运行数据输入固定为 `data/data_jd.csv` 与 `data/data_newegg.csv`。
3. 服务启动自动重建 `data/agent_parts.db`。

## 今日完成

1. 清理非 Agent 叙述，统一文档口径。
2. 保留并强化模型路由与规则兜底策略。
3. 保留会话持久化、TTL 清理、指标统计链路。

## 明日建议

1. 跑一轮端到端对话回归（模型可用/不可用两种路径）。
2. 检查推荐与兼容校验在典型预算下的稳定性。
3. 基于 `GET /api/metrics` 评估热情档位效果。

## 关键文档

1. `docs/REQUIREMENTS_DESIGN.md`
2. `docs/SYSTEM_ARCHITECTURE.md`
3. `docs/PROJECT_MASTER_RECORD.md`
4. `docs/DISCUSSION_PLANNING_RECORD.md`
5. `docs/TIMELINE_BRIEF.md`
