# RigForge（锐格锻造坊）项目总记录（Agent）


## 1. 项目定位

RigForge（锐格锻造坊）是面向装机咨询的对话式 Agent：

1. 多轮沟通收集需求。
2. 生成配置并校验兼容性。
3. 在模型不可用时保持可用回复。

## 2. 核心流程

1. 破冰开场。
2. 结构化收集需求（预算、用途、分辨率、显示器、存储、静音）。
3. 生成推荐清单与估算。
4. 输出兼容性风险和替代方案。

## 3. 当前技术方案

1. FastAPI 提供服务。
2. LangGraph 编排状态流。
3. LangChain 完成提示词、结构化抽取与工具调用。
4. 规则模式作为兜底。

## 4. 数据输入约定

1. 运行时读取 `data/data_jd.csv` 与 `data/data_newegg.csv`。
2. 启动自动重建 `data/agent_parts.db`。
3. Agent 仅消费运行库，不维护外部数据流程。

## 5. 已落地能力（摘要）

1. 热情度档位：`standard` / `high`。
2. 模型路由与回退原因可见。
3. 会话状态持久化与 TTL 清理。
4. 规则话术模板外置与会话内去重。
5. 服务管理脚本：`scripts/server.sh`。

## 6. 已知风险

1. 外部模型仍可能限流或超时。
2. 推荐质量依赖输入数据完整度。
3. 指标维度仍有细化空间（时间窗口、分群）。

## 7. 下一步优先级

1. 提升规则与模型输出一致性。
2. 增强推荐质量评测与回放工具。
3. 完善兼容性规则覆盖率与可解释性。

## 8. 文档索引

1. `docs/REQUIREMENTS_DESIGN.md`
2. `docs/SYSTEM_ARCHITECTURE.md`
3. `docs/DISCUSSION_PLANNING_RECORD.md`
4. `docs/TIMELINE_BRIEF.md`
