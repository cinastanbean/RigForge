# RigForge 多智能体架构设计讨论

## 一、当前架构分析

### 1.1 现有架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph 单体工作流                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  collect_requirements → [ask_more | recommend] → compose_reply  │
│                           │            │                        │
│                      follow_up    validate_build                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 现有设计理念

根据 `README.md` 中的设计哲学：

> **小心地切分你的处理流程，让模型只干一个简单的事情**

当前通过 **节点拆分** 实现职责分离：
- `extract.py` - 需求提取
- `follow_up.py` - 追问生成
- `recommend.py` - 配置推荐
- `validate.py` - 兼容性验证
- `compose.py` - 回复组装

### 1.3 当前架构的问题

#### 问题1：被动交互模式

**现状**：
- 用户说一句话 → 系统提取需求 → 追问下一个缺失字段 → 等待用户回复
- 推荐配置只有在 `should_continue=false` 时才生成
- 缺乏主动信息推送

**影响**：
- 用户需要多轮对话才能看到任何推荐信息
- 库存、性能信息无法实时反馈
- 交互体验不够"智能"

#### 问题2：数据利用不充分

**现状**：
- CSV数据仅包含基础字段：`sku, name, category, brand, price, socket, tdp, score...`
- 缺少详细的性能参数、评测链接、兼容性说明
- 数据库查询只在推荐阶段触发

**影响**：
- 无法提供性能要点介绍
- 无法提供详细链接
- 用户决策信息不足

#### 问题3：单一工作流限制

**现状**：
- 线性状态机：一条路径走到黑
- 无法并行处理多种信息需求
- 无法在对话过程中动态调整策略

---

## 二、多智能体架构提案

### 2.1 核心思路

将当前的 **"节点拆分"** 升级为 **"智能体拆分"**，实现：

1. **并行执行**：多个Agent同时工作，各自负责专业领域
2. **实时响应**：每轮对话后主动推送最新信息
3. **主动服务**：不只是回答问题，而是主动提供增值信息

### 2.2 Agent 角色定义

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Multi-Agent Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ 对话协调器   │◄──►│ 库存查询器   │◄──►│ 性能信息器   │            │
│   │ (Coordinator)│    │ (Inventory) │    │ (PerfInfo)  │            │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│          │                  │                  │                    │
│          │          ┌───────┴───────┐          │                    │
│          │          │               │          │                    │
│          │          ▼               ▼          │                    │
│          │   ┌─────────────┐  ┌─────────────┐  │                    │
│          └──►│ 兼容性校验器 │  │ 推荐生成器   │◄─┘                    │
│              │ (CompatCheck)│  │(Recommender)│                       │
│              └─────────────┘  └─────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Agent 1: 对话协调器 (Coordinator Agent)

**职责**：
- 维持与用户对话的连贯性
- 理解用户意图，分发任务给其他Agent
- 整合各Agent的输出，生成自然语言回复
- 管理对话状态和上下文

**输入**：
- 用户消息
- 其他Agent的查询结果

**输出**：
- 自然语言回复
- 任务分发指令

#### Agent 2: 库存查询器 (Inventory Agent)

**职责**：
- 实时查询配件库存状态
- 监控价格变动
- 提供可购买性信息

**输入**：
- 用户需求中的配件偏好
- 当前推荐配置

**输出**：
- 库存状态（有货/缺货/即将到货）
- 价格信息
- 购买链接

#### Agent 3: 性能信息器 (Performance Agent)

**职责**：
- 查询芯片和组件的详细性能参数
- 提供性能要点摘要
- 生成评测链接和详细说明

**输入**：
- 配件型号/类别
- 用户用途场景

**输出**：
- 性能参数（跑分、功耗、温度等）
- 场景适配性说明
- 详细评测链接

#### Agent 4: 兼容性校验器 (Compatibility Agent)

**职责**：
- 检查配件之间的物理兼容性
- 检查功耗匹配
- 预警潜在风险

**输入**：
- 当前配置方案
- 用户特殊需求

**输出**：
- 兼容性报告
- 风险提示
- 替代建议

#### Agent 5: 推荐生成器 (Recommender Agent)

**职责**：
- 根据需求生成配置方案
- 动态调整推荐策略
- 提供多个备选方案

**输入**：
- 用户需求
- 库存状态
- 兼容性约束

**输出**：
- 推荐配置方案
- 备选方案
- 价格区间分析

### 2.3 交互流程设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                      每轮对话后的并行处理流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   用户输入                                                           │
│      │                                                               │
│      ▼                                                               │
│   ┌──────────────────────────────────────────────┐                  │
│   │           Coordinator Agent                   │                  │
│   │  1. 理解用户意图                              │                  │
│   │  2. 提取需求更新                              │                  │
│   │  3. 决定需要哪些Agent参与                     │                  │
│   └──────────────────┬───────────────────────────┘                  │
│                      │                                               │
│         ┌────────────┼────────────┬────────────┐                    │
│         │            │            │            │                    │
│         ▼            ▼            ▼            ▼                    │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │Inventory │ │  Perf    │ │ Compat   │ │Recommend │              │
│   │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │              │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│        │            │            │            │                      │
│        └────────────┴────────────┴────────────┘                      │
│                           │                                          │
│                           ▼                                          │
│              ┌────────────────────────┐                              │
│              │   信息整合与回复生成    │                              │
│              │   (Coordinator Agent)  │                              │
│              └────────────┬───────────┘                              │
│                           │                                          │
│                           ▼                                          │
│                    用户回复 + 实时信息                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 示例对话场景

```
用户: 我想配一台9000元左右的游戏主机

Coordinator: 
  [分发任务]
  - Inventory Agent: 查询9000元价位主流配件库存
  - Performance Agent: 获取游戏场景性能参考
  - Recommender Agent: 生成初步配置方案

Coordinator 回复:
"好的，9000元预算做游戏主机很合适！我为您生成了一份初步配置：

【实时推荐】
CPU: Intel i5-13600KF (库存充足 | ¥1650)
     性能亮点：14核20线程，单核性能优秀，适合游戏+多任务
     [详细评测链接]

显卡: RTX 4060 Ti (库存紧张，仅剩3件 | ¥2899)
     性能亮点：支持DLSS 3，2K游戏流畅运行
     [详细评测链接]

...
【兼容性】✓ 所有配件已通过兼容性校验
【功耗估算】整机约450W，建议550W电源

您对这套配置有什么想法？或者有特定的品牌偏好吗？"

用户: 显卡能不能换成AMD的？

Coordinator:
  [分发任务]
  - Inventory Agent: 查询AMD显卡库存
  - Performance Agent: 对比NVIDIA vs AMD性能
  - Compat Agent: 校验新配置兼容性

Coordinator 回复:
"没问题！AMD在这个价位有很好的选择：

替代方案：RX 7700 XT (库存充足 | ¥2699)
性能对比：
- 相比4060 Ti，传统光栅化性能提升约8%
- 不支持DLSS，但有FSR 3.0
- 显存更大(12GB vs 8GB)，适合高材质游戏

[RX 7700 XT详细评测] [对比测试数据]

已更新推荐配置，总价降至8799元，您觉得如何？"
```

---

## 三、数据模型扩展

### 3.1 现有数据结构

```python
class Part(BaseModel):
    sku: str
    name: str
    category: PartCategory
    brand: str
    price: int
    socket: str = ""
    tdp: int = 0
    score: int = 0
    # ... 基础字段
```

### 3.2 需要扩展的字段

```python
class PartExtended(BaseModel):
    # 原有字段
    sku: str
    name: str
    category: PartCategory
    brand: str
    price: int
    
    # === 库存信息 ===
    stock_status: Literal["in_stock", "low_stock", "out_of_stock", "preorder"]
    stock_count: int = 0
    last_price_update: str = ""
    price_history: List[Dict] = []  # 价格历史
    purchase_url: str = ""  # 购买链接
    
    # === 性能信息 ===
    performance_score: int = 0
    gaming_score: int = 0  # 游戏场景得分
    productivity_score: int = 0  # 生产力场景得分
    power_consumption: int = 0  # 实际功耗
    temperature_idle: int = 0
    temperature_load: int = 0
    noise_level: int = 0  # 噪音分贝
    
    # === 详细信息 ===
    specs_detail: Dict = {}  # 详细规格参数
    review_url: str = ""  # 评测链接
    review_summary: str = ""  # 评测摘要
    pros: List[str] = []  # 优点
    cons: List[str] = []  # 缺点
    
    # === 兼容性信息 ===
    compatible_with: List[str] = []  # 兼容的配件SKU列表
    incompatible_with: List[str] = []  # 不兼容的配件
    compatibility_notes: List[str] = []  # 兼容性注意事项
```

### 3.3 数据来源建议

| 信息类型 | 来源 | 更新频率 |
|---------|------|---------|
| 基础规格 | 官方数据 | 产品发布时 |
| 库存/价格 | 电商API | 实时/每小时 |
| 性能评分 | 跑分数据库 | 月度 |
| 评测信息 | 内容平台 | 发布时 |

---

## 四、技术实现方案

### 4.1 方案A：LangGraph 多子图架构

```python
from langgraph.graph import StateGraph

# 主图：协调器
main_graph = StateGraph(CoordinatorState)

# 子图：库存查询
inventory_graph = StateGraph(InventoryState)
inventory_graph.add_node("check_stock", check_stock_node)
inventory_graph.add_node("check_price", check_price_node)

# 子图：性能查询
perf_graph = StateGraph(PerfState)
perf_graph.add_node("query_benchmark", query_benchmark_node)
perf_graph.add_node("generate_summary", generate_summary_node)

# 子图：兼容性校验
compat_graph = StateGraph(CompatState)

# 主图注册子图
main_graph.add_node("inventory", inventory_graph.compile())
main_graph.add_node("performance", perf_graph.compile())
main_graph.add_node("compatibility", compat_graph.compile())
```

### 4.2 方案B：Agent 编排框架

使用 LangGraph 的 Agent 编排能力：

```python
from langgraph.prebuilt import create_agent_executor

# 定义各Agent
coordinator_agent = create_agent_executor(llm, coordinator_tools)
inventory_agent = create_agent_executor(llm, inventory_tools)
perf_agent = create_agent_executor(llm, perf_tools)

# 编排Agent交互
def orchestrate(user_input, state):
    # 1. Coordinator 分析意图
    coord_result = coordinator_agent.invoke({
        "input": user_input,
        "state": state
    })
    
    # 2. 并行调用专业Agent
    with ThreadPoolExecutor() as executor:
        inventory_future = executor.submit(
            inventory_agent.invoke, coord_result["inventory_query"]
        )
        perf_future = executor.submit(
            perf_agent.invoke, coord_result["perf_query"]
        )
        
    # 3. 整合结果
    inventory_result = inventory_future.result()
    perf_result = perf_future.result()
    
    # 4. 生成最终回复
    return coordinator_agent.invoke({
        "task": "compose_reply",
        "inventory": inventory_result,
        "performance": perf_result
    })
```

### 4.3 方案C：事件驱动架构

```python
from langgraph.graph import END, StateGraph

class AgentEvent(BaseModel):
    agent_name: str
    event_type: str
    data: Dict

def event_driven_workflow():
    graph = StateGraph(EventState)
    
    # 事件分发器
    graph.add_node("dispatcher", event_dispatcher)
    
    # Agent处理器
    graph.add_node("inventory_handler", inventory_handler)
    graph.add_node("perf_handler", perf_handler)
    graph.add_node("compat_handler", compat_handler)
    
    # 结果聚合器
    graph.add_node("aggregator", result_aggregator)
    
    # 边：基于事件类型路由
    graph.add_conditional_edges(
        "dispatcher",
        route_by_event,
        {
            "inventory": "inventory_handler",
            "performance": "perf_handler",
            "compatibility": "compat_handler"
        }
    )
    
    return graph.compile()
```

---

## 五、关键决策（已确认）

### 5.1 项目目标

**明确目标**：跑通多Agent流程，验证架构可行性，而非实际工程落地。

### 5.2 数据来源策略

| 数据类型 | 实现方式 | 说明 |
|---------|---------|------|
| 库存信息 | **伪造函数** | 默认返回"库存充足" |
| 性能评估 | **伪造函数** | 基于设备接口数据计算评分 |
| 价格信息 | 现有CSV | 保持不变 |

```python
# 伪实现示例
def get_inventory_status(sku: str) -> dict:
    """模拟库存查询，默认返回充足"""
    return {
        "sku": sku,
        "status": "in_stock",
        "quantity": 999,
        "message": "库存充足"
    }

def evaluate_performance(part: Part, use_case: str) -> dict:
    """模拟性能评估，基于设备参数打分"""
    base_score = part.score
    
    # 根据用途场景调整评分
    if use_case == "gaming" and part.category == "gpu":
        adjusted_score = base_score * 1.2
    elif use_case == "video_editing" and part.category == "cpu":
        adjusted_score = base_score * 1.1
    else:
        adjusted_score = base_score
    
    return {
        "sku": part.sku,
        "base_score": base_score,
        "adjusted_score": int(adjusted_score),
        "scenario": use_case,
        "summary": f"在{use_case}场景下表现{'优秀' if adjusted_score > 80 else '良好'}"
    }
```

### 5.3 Agent 粒度定义（已确认）

```
┌─────────────────────────────────────────────────────────────────┐
│                      最终 Agent 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Coordinator Agent（对话协调器）              │   │
│   │  职责：                                                  │   │
│   │  - 理解用户意图                                          │   │
│   │  - 分发任务给其他Agent                                   │   │
│   │  - 整合各Agent输出，生成自然语言回复                      │   │
│   │  - 包含推荐功能（生成配置方案）                           │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│            ┌──────────────┼──────────────┐                      │
│            │              │              │                      │
│            ▼              ▼              ▼                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│   │  Inventory  │ │ Performance │ │ Compatibility│              │
│   │   Agent     │ │   Agent     │ │   Agent      │              │
│   │  库存查询   │ │  性能评估    │ │  兼容性校验   │              │
│   └─────────────┘ └─────────────┘ └─────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**各Agent职责边界**：

| Agent | 输入 | 输出 | 实现方式 |
|-------|-----|------|---------|
| **Coordinator** | 用户消息 + 其他Agent结果 | 自然语言回复 + 推荐配置 | LLM + 规则 |
| **Inventory** | 配件SKU列表 | 库存状态字典 | 伪造函数 |
| **Performance** | 配件 + 用途场景 | 性能评分 + 摘要 | 伪造函数 |
| **Compatibility** | 配置方案 | 兼容性报告 + 风险提示 | 规则引擎（复用现有） |

### 5.4 实时性要求

**结论**：虚拟环境，库存默认充足，无需真实API对接。

### 5.5 迁移策略

**结论**：后续重构为多Agent架构，当前仅做方案设计。

---

## 六、详细设计方案

### 6.1 Agent 接口定义

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel

# === Agent 输出模型 ===

class InventoryResult(BaseModel):
    sku: str
    status: str  # "in_stock" | "low_stock" | "out_of_stock"
    quantity: int
    message: str

class PerformanceResult(BaseModel):
    sku: str
    base_score: int
    adjusted_score: int
    scenario: str
    summary: str
    highlights: List[str] = []  # 性能亮点

class CompatibilityResult(BaseModel):
    is_compatible: bool
    issues: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class CoordinatorOutput(BaseModel):
    reply: str
    build: Optional[BuildPlan] = None
    inventory_info: Dict[str, InventoryResult] = {}
    performance_info: Dict[str, PerformanceResult] = {}
    compatibility_info: Optional[CompatibilityResult] = None
    should_ask_more: bool = False
    next_question: Optional[str] = None

# === Agent 基类 ===

class BaseAgent(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        """执行Agent任务"""
        pass

# === 具体 Agent ===

class InventoryAgent(BaseAgent):
    """库存查询Agent - 伪造实现"""
    
    def invoke(self, skus: List[str]) -> Dict[str, InventoryResult]:
        results = {}
        for sku in skus:
            results[sku] = InventoryResult(
                sku=sku,
                status="in_stock",
                quantity=999,
                message="库存充足"
            )
        return results

class PerformanceAgent(BaseAgent):
    """性能评估Agent - 伪造实现"""
    
    def __init__(self, parts_repo):
        self.repo = parts_repo
    
    def invoke(self, skus: List[str], use_case: str) -> Dict[str, PerformanceResult]:
        results = {}
        for sku in skus:
            part = self.repo.find_by_sku(sku)
            if part:
                # 简单的评分逻辑
                base = part.score
                adjusted = self._adjust_score(base, part.category, use_case)
                results[sku] = PerformanceResult(
                    sku=sku,
                    base_score=base,
                    adjusted_score=adjusted,
                    scenario=use_case,
                    summary=self._generate_summary(part, use_case, adjusted),
                    highlights=self._get_highlights(part)
                )
        return results
    
    def _adjust_score(self, base: int, category: str, use_case: str) -> int:
        # 场景加成逻辑
        modifiers = {
            ("gpu", "gaming"): 1.2,
            ("cpu", "video_editing"): 1.15,
            ("cpu", "ai"): 1.1,
            ("storage", "video_editing"): 1.1,
        }
        modifier = modifiers.get((category, use_case), 1.0)
        return int(base * modifier)
    
    def _generate_summary(self, part, use_case: str, score: int) -> str:
        level = "优秀" if score > 80 else "良好" if score > 60 else "一般"
        return f"{part.name} 在{use_case}场景下表现{level}（评分: {score}）"
    
    def _get_highlights(self, part) -> List[str]:
        highlights = []
        if part.tdp and part.tdp > 100:
            highlights.append(f"高性能设计（TDP {part.tdp}W）")
        if part.vram and part.vram >= 12:
            highlights.append(f"大显存（{part.vram}GB），适合高分辨率")
        return highlights

class CompatibilityAgent(BaseAgent):
    """兼容性校验Agent - 复用现有逻辑"""
    
    def invoke(self, build: BuildPlan) -> CompatibilityResult:
        issues = []
        warnings = []
        suggestions = []
        
        # CPU-主板插槽兼容性
        if build.cpu and build.motherboard:
            if build.cpu.socket != build.motherboard.socket:
                issues.append(
                    f"CPU插槽不兼容：{build.cpu.socket} vs {build.motherboard.socket}"
                )
        
        # 功耗检查
        total_power = self._estimate_power(build)
        if build.psu and build.psu.watt < total_power * 1.2:
            warnings.append(
                f"电源功率可能不足：建议 {int(total_power * 1.3)}W 以上"
            )
        
        # 显卡长度检查
        if build.gpu and build.case:
            if build.gpu.length_mm > build.case.length_mm - 20:
                issues.append("显卡过长，可能无法安装")
        
        return CompatibilityResult(
            is_compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
```

### 6.2 Coordinator Agent 编排逻辑

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class CoordinatorAgent(BaseAgent):
    """对话协调器 - 核心编排Agent"""
    
    def __init__(
        self,
        llm,
        inventory_agent: InventoryAgent,
        performance_agent: PerformanceAgent,
        compatibility_agent: CompatibilityAgent,
        parts_repo
    ):
        self.llm = llm
        self.inventory_agent = inventory_agent
        self.performance_agent = performance_agent
        self.compatibility_agent = compatibility_agent
        self.repo = parts_repo
    
    def invoke(
        self,
        user_input: str,
        current_requirements: UserRequirements,
        current_build: Optional[BuildPlan] = None
    ) -> CoordinatorOutput:
        """
        主编排流程：
        1. 理解用户意图
        2. 决定是否需要生成/更新推荐
        3. 并行调用专业Agent
        4. 整合信息生成回复
        """
        
        # Step 1: 意图理解 + 需求提取
        intent = self._understand_intent(user_input, current_requirements)
        
        # Step 2: 判断是否需要推荐
        should_recommend = self._should_recommend(intent, current_requirements)
        
        # Step 3: 如果需要推荐，生成配置
        if should_recommend:
            build = self._generate_build(current_requirements)
        else:
            build = current_build
        
        # Step 4: 并行调用专业Agent
        agent_results = self._invoke_agents_parallel(build, current_requirements)
        
        # Step 5: 整合信息，生成回复
        reply = self._compose_reply(
            user_input=user_input,
            intent=intent,
            build=build,
            inventory_info=agent_results["inventory"],
            performance_info=agent_results["performance"],
            compatibility_info=agent_results["compatibility"],
            requirements=current_requirements
        )
        
        return CoordinatorOutput(
            reply=reply,
            build=build,
            inventory_info=agent_results["inventory"],
            performance_info=agent_results["performance"],
            compatibility_info=agent_results["compatibility"],
            should_ask_more=not should_recommend,
            next_question=intent.get("next_question")
        )
    
    def _invoke_agents_parallel(
        self,
        build: Optional[BuildPlan],
        requirements: UserRequirements
    ) -> Dict:
        """并行调用专业Agent"""
        
        if not build:
            return {
                "inventory": {},
                "performance": {},
                "compatibility": None
            }
        
        # 收集所有SKU
        skus = [p.sku for p in build.as_dict().values() if p]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交任务
            inventory_future = executor.submit(
                self.inventory_agent.invoke, skus
            )
            performance_future = executor.submit(
                self.performance_agent.invoke, 
                skus, 
                requirements.use_case[0] if requirements.use_case else "gaming"
            )
            compatibility_future = executor.submit(
                self.compatibility_agent.invoke, build
            )
            
            # 收集结果
            return {
                "inventory": inventory_future.result(),
                "performance": performance_future.result(),
                "compatibility": compatibility_future.result()
            }
    
    def _compose_reply(
        self,
        user_input: str,
        intent: Dict,
        build: Optional[BuildPlan],
        inventory_info: Dict,
        performance_info: Dict,
        compatibility_info: Optional[CompatibilityResult],
        requirements: UserRequirements
    ) -> str:
        """整合信息，生成自然语言回复"""
        
        # 构建上下文
        context = {
            "user_input": user_input,
            "intent": intent,
            "build": build.as_dict() if build else None,
            "inventory": {k: v.model_dump() for k, v in inventory_info.items()},
            "performance": {k: v.model_dump() for k, v in performance_info.items()},
            "compatibility": compatibility_info.model_dump() if compatibility_info else None,
            "requirements": requirements.model_dump()
        }
        
        # 调用LLM生成回复
        prompt = self._build_compose_prompt(context)
        response = self.llm.invoke(prompt)
        
        return response.content
```

### 6.3 整体工作流图

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    user_input: str
    requirements: UserRequirements
    build: Optional[BuildPlan]
    inventory_info: Dict
    performance_info: Dict
    compatibility_info: Optional[CompatibilityResult]
    reply: str
    route: Literal["ask_more", "recommend"]

def build_multi_agent_graph(
    coordinator: CoordinatorAgent,
    inventory: InventoryAgent,
    performance: PerformanceAgent,
    compatibility: CompatibilityAgent
):
    """构建多Agent工作流"""
    
    graph = StateGraph(MultiAgentState)
    
    # 节点定义
    graph.add_node("understand_intent", understand_intent_node)
    graph.add_node("check_inventory", check_inventory_node)
    graph.add_node("evaluate_performance", evaluate_performance_node)
    graph.add_node("check_compatibility", check_compatibility_node)
    graph.add_node("compose_reply", compose_reply_node)
    graph.add_node("ask_more", ask_more_node)
    
    # 入口
    graph.set_entry_point("understand_intent")
    
    # 路由
    graph.add_conditional_edges(
        "understand_intent",
        lambda state: state["route"],
        {
            "ask_more": "ask_more",
            "recommend": "check_inventory"
        }
    )
    
    # 并行分支
    graph.add_edge("check_inventory", "evaluate_performance")
    graph.add_edge("evaluate_performance", "check_compatibility")
    graph.add_edge("check_compatibility", "compose_reply")
    
    # 结束
    graph.add_edge("compose_reply", END)
    graph.add_edge("ask_more", END)
    
    return graph.compile()
```

---

## 七、迁移实施路径（方案设计）

### 阶段一：准备工作

1. **定义Agent接口**：`src/rigforge/agents/base.py`
2. **实现伪造函数**：库存、性能评估
3. **扩展数据模型**：添加库存、性能相关字段

### 阶段二：Agent 实现

1. **InventoryAgent**：基于伪造函数实现
2. **PerformanceAgent**：基于现有 `score` 字段实现评分逻辑
3. **CompatibilityAgent**：迁移现有 `compatibility.py` 逻辑
4. **CoordinatorAgent**：实现编排逻辑

### 阶段三：工作流重构

1. 替换现有 `graph.py` 为多Agent工作流
2. 实现并行调用机制
3. 调整回复生成逻辑

### 阶段四：前端适配

1. 扩展 API 返回字段
2. 展示库存状态、性能评分
3. 增强兼容性报告展示

---

## 八、关键决策确认

### 8.1 Coordinator LLM使用策略

**决策：方案B - 意图理解用规则，仅回复生成用LLM**

```python
class CoordinatorAgent:
    def invoke(self, user_input, requirements, build):
        # Step 1: 规则引擎理解意图（不调用LLM）
        intent = self._understand_intent_by_rules(user_input, requirements)
        
        # Step 2: 规则判断是否触发推荐
        should_recommend = self._should_recommend_by_rules(intent, requirements)
        
        # Step 3: 调用其他Agent（并行）
        agent_results = self._invoke_agents_parallel(...)
        
        # Step 4: LLM生成自然语言回复（唯一LLM调用）
        reply = self._compose_reply_with_llm(intent, agent_results, requirements)
        
        return CoordinatorOutput(reply=reply, ...)
```

**意图理解规则示例**：

```python
def _understand_intent_by_rules(self, user_input: str, requirements: UserRequirements) -> dict:
    """规则引擎理解用户意图"""
    
    user_lower = user_input.lower()
    
    # 意图分类
    if any(kw in user_lower for kw in ["推荐", "配置", "方案", "多少钱"]):
        return {"type": "request_recommend", "trigger_recommend": True}
    
    elif any(kw in user_lower for kw in ["换", "改", "不要", "换成"]):
        return {"type": "modify_build", "trigger_recommend": True}
    
    elif any(kw in user_lower for kw in ["库存", "有没有", "能买到"]):
        return {"type": "query_inventory", "trigger_inventory": True}
    
    elif any(kw in user_lower for kw in ["性能", "跑分", "怎么样"]):
        return {"type": "query_performance", "trigger_performance": True}
    
    elif any(kw in user_lower for kw in ["兼容", "能不能装", "会冲突"]):
        return {"type": "query_compatibility", "trigger_compat": True}
    
    else:
        # 默认为需求补充
        return {"type": "provide_info", "trigger_recommend": False}

def _should_recommend_by_rules(self, intent: dict, requirements: UserRequirements) -> bool:
    """规则判断是否触发推荐"""
    
    # 用户主动请求推荐
    if intent.get("trigger_recommend"):
        return True
    
    # 核心需求发生变化
    if requirements.budget_set and requirements.use_case_set:
        # 预算或用途刚设置，触发推荐
        return True
    
    return False
```

**优势**：
- 减少90%的LLM调用（仅回复生成需要LLM）
- 意图判断可控、可预测
- 降低响应延迟

### 8.2 推荐触发时机

**决策：方案C - 用户询问或需求变化时推荐**

```python
class RecommendTrigger:
    """推荐触发规则"""
    
    def should_trigger(self, 
                       user_input: str, 
                       old_requirements: UserRequirements,
                       new_requirements: UserRequirements) -> bool:
        
        # 触发条件1：用户主动请求
        if self._is_recommend_request(user_input):
            return True
        
        # 触发条件2：核心需求变化
        if self._core_requirements_changed(old_requirements, new_requirements):
            return True
        
        # 触发条件3：用户表达不满意，需要新方案
        if self._is_dissatisfied(user_input):
            return True
        
        return False
    
    def _core_requirements_changed(self, old, new) -> bool:
        """核心需求变化检测"""
        return (
            old.budget_min != new.budget_min or
            old.budget_max != new.budget_max or
            old.use_case != new.use_case or
            old.resolution != new.resolution or
            old.cpu_preference != new.cpu_preference or
            old.gpu_preference != new.gpu_preference
        )
    
    def _is_recommend_request(self, user_input: str) -> bool:
        """检测推荐请求"""
        keywords = ["推荐", "配置", "方案", "给我个", "来一套"]
        return any(kw in user_input for kw in keywords)
    
    def _is_dissatisfied(self, user_input: str) -> bool:
        """检测用户不满意"""
        keywords = ["太贵", "太便宜", "不行", "换一个", "再看看"]
        return any(kw in user_input for kw in keywords)
```

### 8.3 性能信息展示粒度

**决策：简略版 - 只显示评分和一句话摘要**

```python
# 输出格式
class PerformanceDisplay(BaseModel):
    score: int                          # 评分 0-100
    summary: str                        # 一句话摘要
    
# 示例输出
{
    "score": 85,
    "summary": "i5-13600KF 在游戏场景下表现优秀"
}

# 前端展示
┌─────────────────────────────────────┐
│ CPU: Intel i5-13600KF               │
│ 性能评分: 85 ★★★★☆                  │
│ 游戏场景表现优秀                     │
└─────────────────────────────────────┘
```

---

## 九、多进程架构设计

### 9.1 为什么选择多进程而非多线程

| 维度 | 多线程 | 多进程 |
|------|-------|-------|
| GIL限制 | 受Python GIL限制，CPU密集型任务无法真正并行 | 突破GIL，真正并行 |
| 进程隔离 | 共享内存，一个崩溃可能影响全部 | 独立进程，互不影响 |
| 通信开销 | 低（共享内存） | 稍高（需要IPC） |
| 适合场景 | I/O密集型 | CPU密集型 + 需要隔离 |
| 故障恢复 | 难以隔离故障 | 单个Agent崩溃不影响其他 |

**结论**：Agent之间存在独立职责，且需要更好的故障隔离，多进程更合适。

### 9.2 进程间通信方案

```python
from multiprocessing import Process, Queue, Manager
from multiprocessing.managers import BaseManager

# 方案选择：Queue + 共享字典

class AgentProcessManager:
    """Agent进程管理器"""
    
    def __init__(self):
        self.manager = Manager()
        self.task_queue = Queue()          # 任务队列
        self.result_dict = self.manager.dict()  # 结果共享字典
        self.status_dict = self.manager.dict()  # 状态共享字典
        self.processes = {}
    
    def start_agents(self):
        """启动所有Agent进程"""
        
        agents = [
            ("inventory", InventoryAgentProcess),
            ("performance", PerformanceAgentProcess),
            ("compatibility", CompatibilityAgentProcess),
        ]
        
        for name, agent_class in agents:
            p = Process(
                target=agent_class.run,
                args=(self.task_queue, self.result_dict, self.status_dict)
            )
            p.start()
            self.processes[name] = p
    
    def submit_task(self, agent_name: str, task: dict):
        """提交任务给指定Agent"""
        self.task_queue.put({
            "agent": agent_name,
            "task": task,
            "task_id": str(uuid.uuid4())
        })
    
    def get_results(self) -> dict:
        """获取所有Agent结果"""
        return dict(self.result_dict)
    
    def get_status(self) -> dict:
        """获取所有Agent状态"""
        return dict(self.status_dict)
```

### 9.3 Agent进程实现

```python
class InventoryAgentProcess:
    """库存查询Agent进程"""
    
    @staticmethod
    def run(task_queue: Queue, result_dict: dict, status_dict: dict):
        agent_name = "inventory"
        status_dict[agent_name] = "idle"
        
        while True:
            try:
                # 阻塞等待任务
                item = task_queue.get()
                
                if item is None:  # 终止信号
                    break
                
                if item["agent"] != agent_name:
                    # 放回队列供其他Agent处理
                    task_queue.put(item)
                    continue
                
                # 更新状态
                status_dict[agent_name] = "processing"
                task_id = item["task_id"]
                task = item["task"]
                
                # 执行任务
                result = InventoryAgentProcess._execute(task)
                
                # 写入结果
                result_dict[f"{agent_name}_{task_id}"] = result
                status_dict[agent_name] = "completed"
                
            except Exception as e:
                status_dict[agent_name] = f"error: {str(e)}"
    
    @staticmethod
    def _execute(task: dict) -> dict:
        """执行库存查询"""
        skus = task.get("skus", [])
        return {
            sku: {
                "status": "in_stock",
                "quantity": 999,
                "message": "库存充足"
            }
            for sku in skus
        }


class PerformanceAgentProcess:
    """性能评估Agent进程"""
    
    @staticmethod
    def run(task_queue: Queue, result_dict: dict, status_dict: dict):
        agent_name = "performance"
        status_dict[agent_name] = "idle"
        
        while True:
            try:
                item = task_queue.get()
                
                if item is None:
                    break
                
                if item["agent"] != agent_name:
                    task_queue.put(item)
                    continue
                
                status_dict[agent_name] = "processing"
                task_id = item["task_id"]
                task = item["task"]
                
                result = PerformanceAgentProcess._execute(task)
                
                result_dict[f"{agent_name}_{task_id}"] = result
                status_dict[agent_name] = "completed"
                
            except Exception as e:
                status_dict[agent_name] = f"error: {str(e)}"
    
    @staticmethod
    def _execute(task: dict) -> dict:
        """执行性能评估（伪造实现）"""
        parts = task.get("parts", [])
        use_case = task.get("use_case", "gaming")
        
        results = {}
        for part in parts:
            base_score = part.get("score", 50)
            adjusted = PerformanceAgentProcess._adjust_score(
                base_score, part.get("category"), use_case
            )
            results[part["sku"]] = {
                "score": adjusted,
                "summary": f"在{use_case}场景下表现{'优秀' if adjusted > 80 else '良好'}"
            }
        return results
    
    @staticmethod
    def _adjust_score(base: int, category: str, use_case: str) -> int:
        modifiers = {
            ("gpu", "gaming"): 1.2,
            ("cpu", "video_editing"): 1.15,
            ("cpu", "ai"): 1.1,
        }
        modifier = modifiers.get((category, use_case), 1.0)
        return int(base * modifier)
```

### 9.4 Coordinator 与 Agent 的交互流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    多进程交互流程                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   用户输入                                                           │
│      │                                                               │
│      ▼                                                               │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    Coordinator 进程                          │  │
│   │                                                              │  │
│   │  1. 规则引擎理解意图                                         │  │
│   │  2. 判断需要哪些Agent参与                                    │  │
│   │  3. 提交任务到Queue                                          │  │
│   │  4. 立即返回"正在查询..."给用户                               │  │
│   │  5. 轮询检查Agent状态                                        │  │
│   │  6. 收集结果后生成最终回复                                   │  │
│   │                                                              │  │
│   └────────────────────────┬─────────────────────────────────────┘  │
│                            │                                         │
│         ┌──────────────────┼──────────────────┐                    │
│         │                  │                  │                    │
│         ▼                  ▼                  ▼                    │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                │
│   │Inventory │      │Performance│     │Compat    │                │
│   │ Process  │      │ Process   │     │ Process  │                │
│   │          │      │           │     │          │                │
│   │ status:  │      │ status:   │     │ status:  │                │
│   │processing│      │ processing│     │processing│                │
│   └────┬─────┘      └─────┬─────┘     └────┬─────┘                │
│        │                  │                 │                       │
│        └──────────────────┴─────────────────┘                       │
│                           │                                          │
│                           ▼                                          │
│                    result_dict (共享)                                │
│                           │                                          │
│                           ▼                                          │
│                   Coordinator 收集结果                               │
│                           │                                          │
│                           ▼                                          │
│                    最终回复给用户                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.5 用户体验设计：即时反馈 + 后台更新

```python
class CoordinatorAgent:
    """Coordinator进程实现"""
    
    def __init__(self, process_manager: AgentProcessManager):
        self.manager = process_manager
    
    def handle_user_input(self, user_input: str, state: dict) -> dict:
        """处理用户输入，返回即时响应 + 后台任务"""
        
        # 1. 规则理解意图（快速，不阻塞）
        intent = self._understand_intent_by_rules(user_input, state["requirements"])
        
        # 2. 判断是否需要后台Agent工作
        need_background = self._need_agent_work(intent)
        
        if need_background:
            # 3. 提交后台任务
            task_id = str(uuid.uuid4())
            self._submit_agent_tasks(intent, state, task_id)
            
            # 4. 立即返回即时响应
            return {
                "reply": self._generate_interim_reply(intent),
                "status": "processing",
                "task_id": task_id,
                "show_loading": True,  # 前端显示加载图标
                "background_agents": ["inventory", "performance", "compatibility"]
            }
        else:
            # 不需要后台处理，直接回复
            return {
                "reply": self._generate_direct_reply(intent, user_input),
                "status": "completed",
                "show_loading": False
            }
    
    def _generate_interim_reply(self, intent: dict) -> str:
        """生成即时回复（让用户知道正在处理）"""
        
        templates = {
            "request_recommend": "好的，我正在为您检索库存、评估性能并检查兼容性，请稍候...",
            "modify_build": "明白，我正在重新查询配件信息，马上为您更新方案...",
            "query_inventory": "正在查询库存状态...",
            "query_performance": "正在评估性能表现...",
            "query_compatibility": "正在检查兼容性...",
        }
        
        return templates.get(intent["type"], "收到，正在处理...")
    
    def check_background_status(self, task_id: str) -> dict:
        """检查后台任务状态（供前端轮询）"""
        
        status = self.manager.get_status()
        results = self.manager.get_results()
        
        # 检查各Agent是否完成
        inventory_done = status.get("inventory") == "completed"
        performance_done = status.get("performance") == "completed"
        compatibility_done = status.get("compatibility") == "completed"
        
        all_done = inventory_done and performance_done and compatibility_done
        
        if all_done:
            # 所有Agent完成，生成最终回复
            final_reply = self._compose_final_reply(results, task_id)
            return {
                "status": "completed",
                "reply": final_reply,
                "results": results,
                "show_loading": False
            }
        else:
            # 仍在处理中
            progress = []
            if not inventory_done:
                progress.append("库存查询中...")
            if not performance_done:
                progress.append("性能评估中...")
            if not compatibility_done:
                progress.append("兼容性检查中...")
            
            return {
                "status": "processing",
                "progress": progress,
                "show_loading": True
            }
```

### 9.6 WebSocket 实时推送设计

**决策：使用 WebSocket 替代轮询，实时推送Agent状态**

```python
# backend/websocket.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import json

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """建立连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        """断开连接"""
        self.active_connections.pop(session_id, None)
    
    async def send_message(self, session_id: str, message: dict):
        """发送消息给指定session"""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接"""
        for connection in self.active_connections.values():
            await connection.send_json(message)


# WebSocket消息类型
class MessageType:
    INTERIM_REPLY = "interim_reply"      # 即时回复
    AGENT_STATUS = "agent_status"        # Agent状态更新
    AGENT_RESULT = "agent_result"        # Agent完成结果
    FINAL_REPLY = "final_reply"          # 最终回复
    ERROR = "error"                      # 错误消息


# FastAPI WebSocket端点
from fastapi import APIRouter

router = APIRouter()
manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                # 处理用户消息
                await handle_chat_message(session_id, data["message"])
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)


async def handle_chat_message(session_id: str, user_message: str):
    """处理聊天消息，推送Agent状态"""
    
    # 1. 发送即时回复
    await manager.send_message(session_id, {
        "type": MessageType.INTERIM_REPLY,
        "content": "好的，我正在为您检索库存、评估性能并检查兼容性..."
    })
    
    # 2. 发送Agent状态（开始处理）
    await manager.send_message(session_id, {
        "type": MessageType.AGENT_STATUS,
        "agents": {
            "inventory": "processing",
            "performance": "pending",
            "compatibility": "pending"
        }
    })
    
    # 3. 启动后台任务（异步）
    task_id = await start_background_agents(session_id, user_message)
    
    # 4. 监控Agent状态变化并推送
    asyncio.create_task(
        monitor_and_push_results(session_id, task_id)
    )


async def monitor_and_push_results(session_id: str, task_id: str):
    """监控Agent执行，实时推送结果"""
    
    from .agents import get_agent_status, get_agent_results
    
    completed_agents = set()
    
    while len(completed_agents) < 3:
        await asyncio.sleep(0.2)  # 200ms检查一次
        
        status = get_agent_status(task_id)
        
        for agent_name in ["inventory", "performance", "compatibility"]:
            if status.get(agent_name) == "completed" and agent_name not in completed_agents:
                
                # 获取单个Agent结果
                result = get_agent_results(task_id, agent_name)
                
                # 推送Agent完成消息
                await manager.send_message(session_id, {
                    "type": MessageType.AGENT_RESULT,
                    "agent": agent_name,
                    "result": result
                })
                
                completed_agents.add(agent_name)
                
                # 更新其他Agent状态
                await manager.send_message(session_id, {
                    "type": MessageType.AGENT_STATUS,
                    "agents": status
                })
    
    # 5. 所有Agent完成，推送最终回复
    final_reply = compose_final_reply(task_id)
    
    await manager.send_message(session_id, {
        "type": MessageType.FINAL_REPLY,
        "content": final_reply["reply"],
        "build": final_reply["build"],
        "inventory_info": final_reply["inventory_info"],
        "performance_info": final_reply["performance_info"],
        "compatibility_info": final_reply["compatibility_info"]
    })
```

**前端WebSocket实现**：

```javascript
// frontend/websocket.js

class AgentWebSocket {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.ws = null;
        this.handlers = {};
    }
    
    connect() {
        const wsUrl = `ws://${location.host}/ws/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // 自动重连
            setTimeout(() => this.connect(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleMessage(data) {
        const { type, ...payload } = data;
        
        switch (type) {
            case 'interim_reply':
                // 显示即时回复
                this.handlers.onInterimReply?.(payload.content);
                break;
                
            case 'agent_status':
                // 更新Agent状态图标
                this.handlers.onAgentStatus?.(payload.agents);
                break;
                
            case 'agent_result':
                // 单个Agent完成，更新对应区域
                this.handlers.onAgentResult?.(payload.agent, payload.result);
                break;
                
            case 'final_reply':
                // 所有完成，显示最终结果
                this.handlers.onFinalReply?.(payload);
                break;
                
            case 'error':
                this.handlers.onError?.(payload.message);
                break;
        }
    }
    
    sendChat(message) {
        this.ws.send(JSON.stringify({
            type: 'chat',
            message: message
        }));
    }
    
    on(event, handler) {
        this.handlers[event] = handler;
    }
}


// 使用示例
const ws = new AgentWebSocket(sessionId);
ws.connect();

ws.on('onInterimReply', (content) => {
    appendMessage('assistant', content);
});

ws.on('onAgentStatus', (agents) => {
    // 更新右侧面板的Agent状态图标
    updateAgentStatus(agents);
    // agents: { inventory: 'processing', performance: 'pending', ... }
});

ws.on('onAgentResult', (agent, result) => {
    // 单个Agent完成，立即显示结果
    if (agent === 'inventory') {
        showInventoryInfo(result);
    } else if (agent === 'performance') {
        showPerformanceInfo(result);
    } else if (agent === 'compatibility') {
        showCompatibilityInfo(result);
    }
    
    // 标记该Agent为完成
    markAgentCompleted(agent);
});

ws.on('onFinalReply', (payload) => {
    // 最终回复
    appendMessage('assistant', payload.content);
    
    // 更新推荐面板
    updateBuildPanel(payload.build);
});

// 发送消息
function sendMessage(message) {
    ws.sendChat(message);
}
```

**UI状态展示**：

```html
<!-- 右侧面板Agent状态指示器 -->
<div class="agent-status-panel">
    <div class="agent-item" id="inventory-status">
        <span class="agent-icon">📦</span>
        <span class="agent-name">库存查询</span>
        <span class="agent-state pending">待处理</span>
    </div>
    <div class="agent-item" id="performance-status">
        <span class="agent-icon">⚡</span>
        <span class="agent-name">性能评估</span>
        <span class="agent-state pending">待处理</span>
    </div>
    <div class="agent-item" id="compatibility-status">
        <span class="agent-icon">🔧</span>
        <span class="agent-name">兼容性检查</span>
        <span class="agent-state pending">待处理</span>
    </div>
</div>
```

```css
/* Agent状态样式 */
.agent-state {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
}

.agent-state.pending {
    background: #e0e0e0;
    color: #666;
}

.agent-state.processing {
    background: #fff3cd;
    color: #856404;
    animation: pulse 1s infinite;
}

.agent-state.completed {
    background: #d4edda;
    color: #155724;
}

.agent-state.error {
    background: #f8d7da;
    color: #721c24;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
```

```javascript
// 状态更新函数
function updateAgentStatus(agents) {
    for (const [agent, status] of Object.entries(agents)) {
        const el = document.querySelector(`#${agent}-status .agent-state`);
        el.className = `agent-state ${status}`;
        el.textContent = {
            'pending': '待处理',
            'processing': '处理中...',
            'completed': '已完成',
            'error': '出错'
        }[status];
    }
}
```

### 9.7 进程间通信方案确认

**决策：使用 Python 原生 Queue，不引入外部消息队列**

```python
from multiprocessing import Process, Queue, Manager

class AgentCommunication:
    """Agent进程间通信"""
    
    def __init__(self):
        self.manager = Manager()
        
        # 任务队列：Coordinator → Agent
        self.task_queue = Queue()
        
        # 结果字典：Agent → Coordinator（共享）
        self.result_dict = self.manager.dict()
        
        # 状态字典：Agent状态实时更新（共享）
        self.status_dict = self.manager.dict()
```

**选择理由**：
- 流程验证阶段，无需引入Redis等外部依赖
- Python原生Queue足够满足进程间通信需求
- 保持项目简洁，降低部署复杂度

### 9.8 错误处理与降级策略

**决策：单个Agent超时/崩溃时使用虚拟兜底数据**

```python
import time
from functools import wraps

# 兜底数据模板
FALLBACK_DATA = {
    "inventory": {
        "status": "in_stock",
        "quantity": 999,
        "message": "库存信息暂时不可用，默认显示充足"
    },
    "performance": {
        "score": 70,
        "summary": "性能信息暂时不可用"
    },
    "compatibility": {
        "is_compatible": True,
        "issues": [],
        "warnings": ["兼容性检查暂时跳过"],
        "suggestions": []
    }
}

def with_fallback(agent_name: str, timeout: float = 5.0):
    """Agent超时/异常时的降级装饰器"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 设置超时
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # 检查是否超时
                if time.time() - start_time > timeout:
                    print(f"[WARN] {agent_name} 超时，使用兜底数据")
                    return FALLBACK_DATA[agent_name]
                
                return result
                
            except Exception as e:
                print(f"[ERROR] {agent_name} 异常: {e}，使用兜底数据")
                return FALLBACK_DATA[agent_name]
        
        return wrapper
    return decorator


# Agent实现示例
class InventoryAgentProcess:
    
    @staticmethod
    @with_fallback("inventory", timeout=3.0)
    def _execute(task: dict) -> dict:
        """库存查询（带降级保护）"""
        skus = task.get("skus", [])
        
        # 模拟可能的延迟或异常
        # time.sleep(5)  # 测试超时
        # raise Exception("模拟异常")  # 测试异常
        
        return {
            sku: {
                "status": "in_stock",
                "quantity": 999,
                "message": "库存充足"
            }
            for sku in skus
        }
```

**降级策略表**：

| Agent | 超时时间 | 兜底数据 | 用户体验 |
|-------|---------|---------|---------|
| Inventory | 3s | 库存充足 | 不阻塞购买流程 |
| Performance | 5s | 评分70，无摘要 | 用户可正常看到配置 |
| Compatibility | 3s | 通过，带警告 | 用户自行确认兼容性 |

### 9.9 进程架构优势

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户体验提升                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统模式：                                                       │
│  用户 → 等待... → 完整响应                                       │
│         └── 用户等待时间长，无反馈 ──┘                            │
│                                                                  │
│  多进程即时反馈模式：                                             │
│  用户 → 即时回复"正在查询..." → 显示加载动画                       │
│         ↓                                                        │
│         后台Agent并行工作（用户可继续对话）                        │
│         ↓                                                        │
│         结果就绪 → 更新右侧面板 + 补充回复                         │
│                                                                  │
│  优势：                                                          │
│  ✓ 用户始终知道系统在做什么                                       │
│  ✓ 不会感觉"卡住"                                                │
│  ✓ 单个Agent崩溃不影响其他                                        │
│  ✓ 可以在等待过程中继续对话                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十、目录结构规划

```
src/rigforge/
├── agents/                    # 新增：Agent模块
│   ├── __init__.py
│   ├── base.py               # Agent基类和接口定义
│   ├── coordinator.py        # 对话协调器
│   ├── inventory.py          # 库存查询Agent
│   ├── performance.py        # 性能评估Agent
│   └── compatibility.py      # 兼容性校验Agent
│
├── graph.py                  # 重构：多Agent工作流编排
├── tools.py                  # 保留：配件搜索等工具
├── schemas.py                # 扩展：新增输出模型
│
├── nodes/                    # 可能废弃或保留部分
│   └── ...
│
└── ...
```

---

## 十一、补充设计考量

### 11.1 会话状态管理

**问题**：多进程环境下如何管理用户会话状态？

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

@dataclass
class AgentSession:
    """Agent会话状态"""
    session_id: str
    requirements: UserRequirements
    current_build: Optional[BuildPlan] = None
    
    # Agent执行状态
    active_task_id: Optional[str] = None
    agent_status: Dict[str, str] = field(default_factory=dict)
    agent_results: Dict[str, dict] = field(default_factory=dict)
    
    # 对话历史
    messages: list = field(default_factory=list)
    
    # 时间戳
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def update_agent_status(self, agent_name: str, status: str):
        """更新Agent状态"""
        self.agent_status[agent_name] = status
        self.updated_at = time.time()
    
    def set_agent_result(self, agent_name: str, result: dict):
        """设置Agent结果"""
        self.agent_results[agent_name] = result
        self.updated_at = time.time()
    
    def is_all_agents_completed(self) -> bool:
        """检查所有Agent是否完成"""
        return all(
            status == "completed" 
            for status in self.agent_status.values()
        )


class SessionManager:
    """会话管理器（多进程安全）"""
    
    def __init__(self, manager):
        self.sessions = manager.dict()  # 共享字典
        self._lock = manager.Lock()
    
    def get_or_create(self, session_id: str) -> AgentSession:
        """获取或创建会话"""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = AgentSession(session_id=session_id)
            return self.sessions[session_id]
    
    def update(self, session_id: str, **kwargs):
        """更新会话"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                for key, value in kwargs.items():
                    setattr(session, key, value)
                session.updated_at = time.time()
    
    def cleanup_expired(self, ttl_seconds: int = 3600):
        """清理过期会话"""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, sess in self.sessions.items()
                if now - sess.updated_at > ttl_seconds
            ]
            for sid in expired:
                del self.sessions[sid]
```

### 11.2 日志与监控

**问题**：如何追踪多进程Agent的执行情况？

```python
import logging
import json
from datetime import datetime
from typing import Optional

class AgentLogger:
    """Agent执行日志记录器"""
    
    def __init__(self, log_dir: str = "logs/agents"):
        self.log_dir = log_dir
        self._setup_logger()
    
    def _setup_logger(self):
        """配置日志"""
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(f"{self.log_dir}/agent.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.logger.addHandler(fh)
    
    def log_agent_start(self, session_id: str, agent_name: str, task: dict):
        """记录Agent开始执行"""
        self.logger.info(json.dumps({
            "event": "agent_start",
            "session_id": session_id,
            "agent": agent_name,
            "task": task,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_agent_complete(self, session_id: str, agent_name: str, 
                           result: dict, duration_ms: float):
        """记录Agent完成"""
        self.logger.info(json.dumps({
            "event": "agent_complete",
            "session_id": session_id,
            "agent": agent_name,
            "duration_ms": duration_ms,
            "result_keys": list(result.keys()),
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_agent_error(self, session_id: str, agent_name: str, 
                        error: str, fallback: bool = False):
        """记录Agent错误"""
        self.logger.error(json.dumps({
            "event": "agent_error",
            "session_id": session_id,
            "agent": agent_name,
            "error": error,
            "fallback": fallback,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_coordinator_decision(self, session_id: str, 
                                  intent: str, agents_triggered: list):
        """记录Coordinator决策"""
        self.logger.info(json.dumps({
            "event": "coordinator_decision",
            "session_id": session_id,
            "intent": intent,
            "agents_triggered": agents_triggered,
            "timestamp": datetime.now().isoformat()
        }))


# 使用示例
logger = AgentLogger()

def agent_execute_with_logging(agent_name: str, task: dict, session_id: str):
    start = time.time()
    logger.log_agent_start(session_id, agent_name, task)
    
    try:
        result = execute_agent(agent_name, task)
        duration = (time.time() - start) * 1000
        logger.log_agent_complete(session_id, agent_name, result, duration)
        return result
    except Exception as e:
        logger.log_agent_error(session_id, agent_name, str(e), fallback=True)
        return FALLBACK_DATA[agent_name]
```

**监控指标**：

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class AgentMetrics:
    """Agent性能指标"""
    
    # 执行统计
    total_calls: Dict[str, int] = {}       # 各Agent调用次数
    success_calls: Dict[str, int] = {}     # 成功次数
    fallback_calls: Dict[str, int] = {}    # 降级次数
    
    # 耗时统计
    avg_duration_ms: Dict[str, float] = {}  # 平均耗时
    max_duration_ms: Dict[str, float] = {}  # 最大耗时
    
    def record(self, agent_name: str, duration_ms: float, 
               success: bool, fallback: bool):
        """记录一次执行"""
        self.total_calls[agent_name] = self.total_calls.get(agent_name, 0) + 1
        
        if success and not fallback:
            self.success_calls[agent_name] = self.success_calls.get(agent_name, 0) + 1
        if fallback:
            self.fallback_calls[agent_name] = self.fallback_calls.get(agent_name, 0) + 1
        
        # 更新耗时统计
        current_avg = self.avg_duration_ms.get(agent_name, 0)
        count = self.total_calls[agent_name]
        self.avg_duration_ms[agent_name] = (
            (current_avg * (count - 1) + duration_ms) / count
        )
        self.max_duration_ms[agent_name] = max(
            self.max_duration_ms.get(agent_name, 0), duration_ms
        )
    
    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            "total_calls": self.total_calls,
            "success_rate": {
                name: self.success_calls.get(name, 0) / count * 100
                for name, count in self.total_calls.items()
            },
            "fallback_rate": {
                name: self.fallback_calls.get(name, 0) / count * 100
                for name, count in self.total_calls.items()
            },
            "avg_duration_ms": self.avg_duration_ms,
            "max_duration_ms": self.max_duration_ms
        }
```

### 11.3 测试策略

**问题**：如何测试多进程Agent系统？

```python
import pytest
from multiprocessing import Queue, Manager
from unittest.mock import patch, MagicMock

# === 单元测试 ===

class TestInventoryAgent:
    """库存Agent单元测试"""
    
    def test_normal_execution(self):
        """正常执行测试"""
        result = InventoryAgentProcess._execute({
            "skus": ["CPU-001", "GPU-001"]
        })
        
        assert result["CPU-001"]["status"] == "in_stock"
        assert result["GPU-001"]["quantity"] == 999
    
    def test_empty_skus(self):
        """空SKU列表测试"""
        result = InventoryAgentProcess._execute({"skus": []})
        assert result == {}


class TestPerformanceAgent:
    """性能Agent单元测试"""
    
    def test_gaming_scenario(self):
        """游戏场景评分测试"""
        result = PerformanceAgentProcess._execute({
            "parts": [{"sku": "GPU-001", "score": 80, "category": "gpu"}],
            "use_case": "gaming"
        })
        
        # 游戏场景GPU加成1.2倍
        assert result["GPU-001"]["score"] == 96  # 80 * 1.2
    
    def test_video_editing_scenario(self):
        """视频剪辑场景测试"""
        result = PerformanceAgentProcess._execute({
            "parts": [{"sku": "CPU-001", "score": 70, "category": "cpu"}],
            "use_case": "video_editing"
        })
        
        # 视频剪辑CPU加成1.15倍
        assert result["CPU-001"]["score"] == 80  # 70 * 1.15


class TestCoordinatorIntent:
    """Coordinator意图理解测试"""
    
    def test_recommend_request(self):
        """推荐请求识别"""
        intent = CoordinatorAgent._understand_intent_by_rules(
            "给我推荐一套配置", {}
        )
        assert intent["type"] == "request_recommend"
        assert intent["trigger_recommend"] == True
    
    def test_inventory_query(self):
        """库存查询识别"""
        intent = CoordinatorAgent._understand_intent_by_rules(
            "这个显卡有货吗", {}
        )
        assert intent["type"] == "query_inventory"


# === 集成测试 ===

class TestMultiAgentIntegration:
    """多Agent集成测试"""
    
    @pytest.fixture
    def setup_processes(self):
        """设置测试环境"""
        manager = Manager()
        task_queue = Queue()
        result_dict = manager.dict()
        status_dict = manager.dict()
        
        yield {
            "task_queue": task_queue,
            "result_dict": result_dict,
            "status_dict": status_dict
        }
        
        # 清理
        task_queue.put(None)  # 发送终止信号
    
    def test_parallel_execution(self, setup_processes):
        """并行执行测试"""
        task_queue = setup_processes["task_queue"]
        result_dict = setup_processes["result_dict"]
        status_dict = setup_processes["status_dict"]
        
        # 启动Agent进程
        from threading import Thread
        from time import sleep
        
        def run_inventory():
            InventoryAgentProcess.run(task_queue, result_dict, status_dict)
        
        thread = Thread(target=run_inventory, daemon=True)
        thread.start()
        
        # 提交任务
        task_queue.put({
            "agent": "inventory",
            "task_id": "test-001",
            "task": {"skus": ["CPU-001"]}
        })
        
        # 等待结果
        sleep(1)
        
        assert "inventory_test-001" in result_dict
        assert status_dict.get("inventory") == "completed"


# === 端到端测试 ===

class TestEndToEnd:
    """端到端测试"""
    
    def test_full_conversation_flow(self, test_client, websocket_client):
        """完整对话流程测试"""
        
        # 1. 建立WebSocket连接
        ws = websocket_client.connect("/ws/test-session")
        
        # 2. 发送用户消息
        ws.send_json({"type": "chat", "message": "我想配一台游戏主机"})
        
        # 3. 验证即时回复
        response = ws.receive_json()
        assert response["type"] == "interim_reply"
        assert "正在" in response["content"]
        
        # 4. 验证Agent状态更新
        status = ws.receive_json()
        assert status["type"] == "agent_status"
        
        # 5. 等待最终回复
        while True:
            msg = ws.receive_json()
            if msg["type"] == "final_reply":
                assert msg["build"] is not None
                break
        
        ws.close()


# === Mock测试（不启动真实进程）===

class TestWithMock:
    """使用Mock的测试"""
    
    @patch('multiprocessing.Process')
    def test_coordinator_orchestration(self, mock_process):
        """测试Coordinator编排逻辑"""
        
        # Mock Agent进程
        mock_process.return_value.start.return_value = None
        
        coordinator = CoordinatorAgent(...)
        
        result = coordinator.handle_user_input(
            "推荐一套配置",
            {"requirements": UserRequirements()}
        )
        
        assert result["status"] == "processing"
        assert result["show_loading"] == True
```

### 11.4 配置管理

**问题**：如何管理Agent的超时、重试等配置？

```python
# config/agent_config.yaml

agents:
  inventory:
    timeout_seconds: 3
    max_retries: 2
    retry_delay_seconds: 0.5
    fallback:
      status: "in_stock"
      quantity: 999
      message: "库存信息暂时不可用"
  
  performance:
    timeout_seconds: 5
    max_retries: 1
    retry_delay_seconds: 1.0
    fallback:
      score: 70
      summary: "性能信息暂时不可用"
    score_modifiers:
      gaming:
        gpu: 1.2
        cpu: 1.0
      video_editing:
        cpu: 1.15
        gpu: 1.0
      ai:
        cpu: 1.1
        gpu: 1.1
  
  compatibility:
    timeout_seconds: 3
    max_retries: 0
    fallback:
      is_compatible: true
      issues: []
      warnings: ["兼容性检查暂时跳过"]

coordinator:
  intent_rules:
    recommend_keywords: ["推荐", "配置", "方案", "多少钱"]
    inventory_keywords: ["库存", "有没有", "能买到"]
    performance_keywords: ["性能", "跑分", "怎么样"]
    compatibility_keywords: ["兼容", "能不能装", "会冲突"]

websocket:
  heartbeat_seconds: 30
  max_connections: 100
  message_queue_size: 100

session:
  ttl_seconds: 3600
  cleanup_interval_seconds: 300
```

```python
# src/rigforge/config.py

import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class AgentConfig:
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float
    fallback: dict

@dataclass
class Config:
    agents: dict[str, AgentConfig]
    coordinator: dict
    websocket: dict
    session: dict

def load_config(config_path: str = "config/agent_config.yaml") -> Config:
    """加载配置文件"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    agents = {
        name: AgentConfig(**cfg)
        for name, cfg in raw["agents"].items()
    }
    
    return Config(
        agents=agents,
        coordinator=raw["coordinator"],
        websocket=raw["websocket"],
        session=raw["session"]
    )

# 全局配置实例
config = load_config()
```

### 11.5 数据一致性

**问题**：多进程并发访问共享数据如何保证一致性？

```python
from multiprocessing import Manager, Lock
from typing import Any
import threading

class ThreadSafeDict:
    """线程/进程安全的字典"""
    
    def __init__(self, manager):
        self._dict = manager.dict()
        self._lock = manager.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key: str, value: Any):
        with self._lock:
            self._dict[key] = value
    
    def update(self, key: str, **kwargs):
        """原子更新嵌套字段"""
        with self._lock:
            current = self._dict.get(key, {})
            current.update(kwargs)
            self._dict[key] = current
    
    def delete(self, key: str):
        with self._lock:
            self._dict.pop(key, None)


class AgentResultAggregator:
    """Agent结果聚合器（保证一致性）"""
    
    def __init__(self, manager):
        self.results = ThreadSafeDict(manager)
        self.status = ThreadSafeDict(manager)
        self._expected_agents = ["inventory", "performance", "compatibility"]
    
    def set_result(self, agent_name: str, result: dict):
        """设置单个Agent结果"""
        self.results.set(agent_name, result)
        self.status.set(agent_name, "completed")
    
    def is_all_complete(self) -> bool:
        """检查是否全部完成"""
        for agent in self._expected_agents:
            if self.status.get(agent) != "completed":
                return False
        return True
    
    def get_all_results(self) -> dict:
        """获取所有结果"""
        return {
            agent: self.results.get(agent, FALLBACK_DATA.get(agent, {}))
            for agent in self._expected_agents
        }
```

### 11.6 进程生命周期管理

**问题**：Agent进程如何启动、监控、停止？

```python
import signal
import sys
from multiprocessing import Process
from typing import Dict

class AgentProcessManager:
    """Agent进程生命周期管理"""
    
    def __init__(self):
        self.processes: Dict[str, Process] = {}
        self.manager = Manager()
        self.task_queue = Queue()
        self.result_dict = self.manager.dict()
        self.status_dict = self.manager.dict()
        self._running = False
    
    def start_all(self):
        """启动所有Agent进程"""
        agent_configs = [
            ("inventory", InventoryAgentProcess.run),
            ("performance", PerformanceAgentProcess.run),
            ("compatibility", CompatibilityAgentProcess.run),
        ]
        
        for name, target in agent_configs:
            p = Process(
                target=target,
                args=(self.task_queue, self.result_dict, self.status_dict),
                name=f"agent-{name}",
                daemon=True
            )
            p.start()
            self.processes[name] = p
            print(f"[START] Agent {name} started, PID={p.pid}")
        
        self._running = True
        
        # 注册信号处理
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def stop_all(self):
        """停止所有Agent进程"""
        print("[STOP] Stopping all agents...")
        
        # 发送终止信号
        for _ in self.processes:
            self.task_queue.put(None)
        
        # 等待进程结束
        for name, p in self.processes.items():
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                print(f"[STOP] Agent {name} terminated")
            else:
                print(f"[STOP] Agent {name} stopped")
        
        self._running = False
        self.processes.clear()
    
    def restart_agent(self, agent_name: str):
        """重启单个Agent"""
        if agent_name in self.processes:
            old_process = self.processes[agent_name]
            old_process.terminate()
            old_process.join(timeout=3)
        
        # 重新启动
        target = {
            "inventory": InventoryAgentProcess.run,
            "performance": PerformanceAgentProcess.run,
            "compatibility": CompatibilityAgentProcess.run,
        }[agent_name]
        
        p = Process(
            target=target,
            args=(self.task_queue, self.result_dict, self.status_dict),
            name=f"agent-{agent_name}",
            daemon=True
        )
        p.start()
        self.processes[agent_name] = p
        print(f"[RESTART] Agent {agent_name} restarted, PID={p.pid}")
    
    def health_check(self) -> dict:
        """健康检查"""
        return {
            name: {
                "alive": p.is_alive(),
                "pid": p.pid,
                "status": self.status_dict.get(name, "unknown")
            }
            for name, p in self.processes.items()
        }
    
    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        print(f"\n[SHUTDOWN] Received signal {signum}")
        self.stop_all()
        sys.exit(0)


# FastAPI启动时集成
from contextlib import asynccontextmanager
from fastapi import FastAPI

agent_manager = AgentProcessManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    agent_manager.start_all()
    yield
    # 关闭时
    agent_manager.stop_all()

app = FastAPI(lifespan=lifespan)
```

### 11.7 API设计

**问题**：前端如何调用多Agent系统？

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

router = APIRouter()

# === HTTP API ===

class ChatRequest(BaseModel):
    session_id: str
    message: str
    requirements: Optional[UserRequirements] = None

class ChatResponse(BaseModel):
    session_id: str
    status: Literal["processing", "completed"]
    interim_reply: Optional[str] = None
    reply: Optional[str] = None
    build: Optional[BuildPlan] = None

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送聊天消息（HTTP方式）
    返回即时响应，后续通过WebSocket推送
    """
    session = session_manager.get_or_create(request.session_id)
    
    # 处理消息
    result = coordinator.handle_user_input(
        request.message,
        {"requirements": session.requirements}
    )
    
    return ChatResponse(
        session_id=request.session_id,
        status=result["status"],
        interim_reply=result.get("reply") if result["status"] == "processing" else None,
        reply=result.get("reply") if result["status"] == "completed" else None
    )


# === WebSocket API ===

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket连接，实时接收Agent状态更新
    """
    await manager.connect(websocket, session_id)
    session = session_manager.get_or_create(session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                await handle_chat_message(session_id, data["message"], session)
            
            elif data["type"] == "status":
                # 返回当前状态
                await websocket.send_json({
                    "type": "status",
                    "session": session.model_dump()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)


# === 监控API ===

@router.get("/api/agents/health")
async def agent_health():
    """Agent健康检查"""
    return agent_manager.health_check()

@router.get("/api/agents/metrics")
async def agent_metrics():
    """Agent性能指标"""
    return metrics.get_summary()

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话状态"""
    session = session_manager.sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.model_dump()
```

### 11.8 迁移路径

**问题**：如何从现有架构平滑迁移？

```
┌─────────────────────────────────────────────────────────────────┐
│                        迁移阶段规划                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  阶段1：并行运行（不改变现有逻辑）                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  现有流程 ─────────────────────► 正常工作               │   │
│  │       │                                                  │   │
│  │       └── 同时启动Agent进程 ──► 仅记录日志，不影响结果   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  阶段2：灰度切换（部分流量使用新架构）                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  if session_id in BETA_USERS:                           │   │
│  │      使用多Agent架构                                     │   │
│  │  else:                                                  │   │
│  │      使用原有架构                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  阶段3：完全切换                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  废弃原有 nodes/ 模块                                    │   │
│  │  全部使用 agents/ 模块                                   │   │
│  │  保留原有代码作为回滚备份                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**代码迁移对照表**：

| 现有模块 | 新模块 | 迁移说明 |
|---------|-------|---------|
| `nodes/extract.py` | `agents/coordinator.py` | 意图理解迁移到规则引擎 |
| `nodes/recommend.py` | `agents/coordinator.py` | 推荐逻辑融入Coordinator |
| `nodes/validate.py` | `agents/compatibility.py` | 直接迁移为独立Agent |
| `nodes/compose.py` | `agents/coordinator.py` | 回复生成保留LLM调用 |
| `graph.py` | `agents/orchestrator.py` | 重构为多进程编排 |

---

## 十二、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 多进程调试困难 | 开发效率降低 | 完善日志系统，本地Mock测试 |
| WebSocket连接不稳定 | 用户体验下降 | 自动重连机制 + HTTP降级 |
| Agent进程崩溃 | 功能缺失 | 虚拟兜底数据 + 自动重启 |
| 内存泄漏 | 服务不稳定 | 定期清理会话 + 进程监控 |
| 并发竞争 | 数据不一致 | 使用Manager.Lock保护共享数据 |

---

## 附录：架构对比

| 维度 | 当前架构 | 多Agent架构 |
|------|---------|------------|
| 交互模式 | 被动问答 | 主动信息推送 |
| 响应延迟 | 单次LLM调用 | 并行多Agent调用 |
| 信息丰富度 | 基础推荐 | 性能+库存+评测 |
| 扩展性 | 线性流程 | 模块化Agent |
| 实现复杂度 | 低 | 中高 |
| LLM成本 | 低 | 较高 |
| 用户价值 | 配置推荐 | 决策支持 |
| 容错能力 | 单点故障 | Agent独立降级 |
| 可观测性 | 基础日志 | 结构化日志+指标 |
