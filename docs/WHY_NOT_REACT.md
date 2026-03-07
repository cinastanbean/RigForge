# 为什么装机推荐系统不使用 ReAct 范式

## 概述

本文档解释了 RigForge 装机推荐系统为什么选择 **LangGraph 状态机模式** 而不是 **ReAct (Reasoning + Acting)** 范式。

## 两种范式的核心区别

### ReAct 范式

```
Observation (用户输入)
    ↓
LLM Reasoning (思考下一步)
    ↓
Action/Tool (调用工具)
    ↓
Observation (观察结果)
    ↓
... (循环直到完成)
```

**特点：**
- LLM 动态决定下一步行动
- 每步都需要调用 LLM
- 工具调用顺序不固定
- 适合开放式、探索性任务

### LangGraph 状态机模式

```
需求收集节点 ──► 配置生成节点 ──► 验证节点 ──► 回复节点
     │                │              │            │
     └────────────────┴──────────────┴────────────┘
                    (代码控制流程)
```

**特点：**
- 预定义的工作流节点
- 代码直接控制流程
- 工具调用顺序固定
- 适合结构化、确定性任务

## 为什么装机推荐不适合 ReAct

### 1. 装机是结构化任务

装机推荐有明确的、固定的流程：

1. **收集需求** → 了解预算、用途、偏好
2. **分配预算** → 按固定比例分配各类配件预算
3. **搜索配件** → 在预算范围内搜索最佳配件
4. **验证兼容性** → 检查硬件之间是否兼容
5. **生成回复** → 输出配置单和解释

这个流程是**确定的**，不需要 LLM 动态决定"现在该做什么"。

### 2. 工具调用顺序固定

在装机场景中，工具调用有严格的依赖关系：

```
必须先搜索配件 ──► 才能验证兼容性
     │                    │
     └────────────────────┘
   (不能先验证再搜索)
```

ReAct 的灵活性在这里反而是缺点——LLM 可能做出错误的调用顺序决策。

### 3. 准确性要求极高

装机配置一旦出错，后果严重：
- ❌ CPU 和主板插槽不匹配 → 无法开机
- ❌ 显卡太长机箱装不下 → 物理冲突
- ❌ 电源功率不足 → 系统不稳定

LangGraph 的**确定性**确保了流程不会出错。

### 4. 成本与性能考量

| 指标 | ReAct | LangGraph |
|------|-------|-----------|
| LLM 调用次数 | 5-10 次 | 2-3 次 |
| Token 消耗 | 高 | 低 |
| 响应延迟 | 5-10 秒 | 1-2 秒 |
| 成本 | 高 | 低 |

装机推荐不需要每步都推理，直接执行即可。

### 5. 可维护性

**LangGraph 的优势：**
- 代码即文档，流程清晰可见
- 调试简单，可以单步跟踪
- 单元测试容易编写
- 问题定位快速

**ReAct 的问题：**
- LLM 的决策是黑盒
- 错误难以复现
- 调试困难
- 需要大量日志分析

## 当前架构设计

### 状态图定义

```python
builder = StateGraph(GraphState)

# 定义节点
builder.add_node("collect_requirements", self.collect_requirements)
builder.add_node("generate_follow_up", self.generate_follow_up)
builder.add_node("recommend_build", self.recommend_build)
builder.add_node("validate_build", self.validate_build)
builder.add_node("compose_reply", self.compose_reply)

# 定义边（流程）
builder.set_entry_point("collect_requirements")
builder.add_conditional_edges(
    "collect_requirements",
    self.route_after_collection,
    {"ask_more": "generate_follow_up", "recommend": "recommend_build"},
)
builder.add_edge("generate_follow_up", "compose_reply")
builder.add_edge("recommend_build", "validate_build")
builder.add_edge("validate_build", "compose_reply")
```

### 节点职责清晰

| 节点 | 职责 | 是否使用 LLM |
|------|------|-------------|
| `collect_requirements` | 提取用户需求 | ✅ 是 |
| `generate_follow_up` | 生成追问问题 | ✅ 是 |
| `recommend_build` | 生成配置方案 | ❌ 否（代码逻辑） |
| `validate_build` | 验证兼容性 | ❌ 否（规则检查） |
| `compose_reply` | 生成最终回复 | ✅ 是 |

**关键洞察：** 只有需要"理解"和"生成自然语言"的步骤才使用 LLM，其他步骤用代码逻辑即可。

### 工具调用方式

```python
# 直接调用，不是由 LLM 决定
def recommend_build(self, state: GraphState):
    req = state["requirements"]
    
    # 直接调用工具函数
    _context = self.tool_map["recommendation_context"].invoke(req.model_dump())
    build = pick_build_from_candidates(req, self.tool_map["search_parts"])
    build = ensure_budget_fit(req, build)
    
    return {"build": build}
```

## 适用场景对比

### 适合 LangGraph（状态机）的场景

- ✅ 流程固定的任务
- ✅ 工具调用顺序确定
- ✅ 准确性要求高
- ✅ 需要严格控制输出
- ✅ 成本敏感

**示例：**
- 装机推荐
- 订单处理
- 审批流程
- 数据报表生成

### 适合 ReAct 的场景

- ✅ 开放式问题
- ✅ 需要多步推理
- ✅ 工具调用顺序不确定
- ✅ 探索性任务
- ✅ 可以接受一定错误率

**示例：**
- 复杂问答系统
- 代码调试助手
- 研究助手
- 创意写作

## 结论

RigForge 选择 **LangGraph 状态机模式** 是因为：

1. **装机推荐是结构化任务**，有明确的流程
2. **准确性要求高**，不能容忍流程错误
3. **成本敏感**，需要减少 LLM 调用次数
4. **可维护性重要**，代码需要清晰易懂

ReAct 的灵活性在这种场景下是**过度设计**，反而会增加复杂性和出错概率。

**简单问题用简单方案。** 装机推荐不需要 LLM 每步都"思考"，直接执行预定义流程即可。

## 参考

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - 系统架构详细说明
