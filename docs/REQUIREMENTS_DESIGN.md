# RigForge（锐格锻造坊）需求设计文档

## 1. 背景与目标

RigForge（锐格锻造坊）是一个面向装机咨询的对话式推荐系统。核心目标：

1. 通过积极、结构化的多轮互动提升用户参与感。
2. 收集预算、用途、分辨率、静音与显示器等关键需求。
3. 基于统一件库自动生成推荐方案，并给出兼容性提示。
4. 使用 LangChain / LangGraph 构建可扩展的对话流程。

## 2. 产品范围

### 2.1 必问信息

1. 预算区间
2. 使用场景（游戏/办公/剪辑/AI）
3. 目标分辨率
4. 是否包含显示器
5. 存储需求
6. 静音偏好

### 2.2 本版本硬件类目边界

1. `CPU`
2. `CPU cooler`
3. `Motherboard`
4. `Memory`
5. `Storage`
6. `Video card (GPU)`
7. `Power supply (PSU)`
8. `Case`
9. `Monitor`
10. `Keyboard/Mouse` 作为后续版本范围。

### 2.3 输出内容

1. 推荐配置（8 件套核心硬件）
2. 预估总价、功耗标签
3. 兼容性校验结果与风险说明
4. 可选替代建议

## 3. 对话体验原则

1. 先破冰，再收集，再推荐。
2. 每轮默认 1 问，最多 2 问。
3. 保持口语化、自然、积极反馈。
4. 推荐阶段先确认目标再给执行建议。
5. `CPU 偏好` 作为主动追问项，默认会问，但不阻塞推荐产出。
6. 聊天过程中实时更新“推荐配置”预览，不等待最终轮次。

## 4. 互动档位

1. `standard`：专业、简洁。
2. `high`：高互动、强反馈。

用途：用于 A/B 对比完成率与转化率。

## 5. 技术设计

### 5.1 LangChain 优先

1. `ChatPromptTemplate`
2. `with_structured_output`
3. Tool calling
4. Runnable 组合
5. 回退兜底机制

### 5.2 LangGraph 状态流

1. `collect_requirements`
2. `generate_follow_up`
3. `recommend_build`
4. `validate_build`
5. `compose_reply`

### 5.3 路由规则

1. 关键信息缺失 -> 追问
2. 关键信息齐全 -> 推荐

## 6. 数据输入约束（当前）

1. 主项目运行时只消费 `data/` 下两份 CSV：
2. `data/data_jd.csv`
3. `data/data_newegg.csv`
4. 服务启动时自动重建运行数据库 `data/agent_parts.db`。

## 7. 接口

### 7.1 聊天接口

`POST /api/chat`

### 7.2 观察接口

`GET /api/metrics`

## 8. 非功能需求

1. 模型异常时可回退规则，服务可用。
2. 核心流程有自动化测试覆盖。
3. 模块化结构，便于迭代策略。

## 9. 启动约定

推荐使用：`scripts/server.sh`

1. `./scripts/server.sh restart bg`
2. `./scripts/server.sh restart fg`
3. `./scripts/server.sh stop`
4. `./scripts/server.sh status`
