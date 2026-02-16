# RigForge 文档索引

本目录包含 RigForge（锐格锻造坊）的核心设计文档。

## 核心文档

| 文档 | 说明 |
|------|------|
| [REQUIREMENTS_DESIGN.md](REQUIREMENTS_DESIGN.md) | 需求设计说明书 - 产品目标、功能范围、对话体验原则 |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | 系统架构说明书 - 模块职责、技术方案、配置项 |
| [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) | 工作流程详解 - 会话到推荐生成的完整流程框图 |
| [DATA_INPUT_CONTRACT.md](DATA_INPUT_CONTRACT.md) | 数据输入契约 - CSV 字段定义、类目范围、约束条件 |

## 文档关系

```
REQUIREMENTS_DESIGN.md     ← 需求侧（做什么）
        │
        ▼
SYSTEM_ARCHITECTURE.md    ← 架构侧（怎么设计）
        │
        ▼
WORKFLOW_DIAGRAM.md       ← 流程侧（怎么执行）
        │
        ▼
DATA_INPUT_CONTRACT.md    ← 数据侧（数据格式）
```

## 快速导航

### 我想了解...

- **产品定位** → [REQUIREMENTS_DESIGN.md#背景与目标](REQUIREMENTS_DESIGN.md#1-背景与目标)
- **功能范围** → [REQUIREMENTS_DESIGN.md#产品范围](REQUIREMENTS_DESIGN.md#2-产品范围)
- **技术架构** → [SYSTEM_ARCHITECTURE.md#总体架构](SYSTEM_ARCHITECTURE.md#2-总体架构)
- **模块职责** → [SYSTEM_ARCHITECTURE.md#模块职责](SYSTEM_ARCHITECTURE.md#3-模块职责)
- **会话流程** → [WORKFLOW_DIAGRAM.md#状态流转图](WORKFLOW_DIAGRAM.md#二状态流转图)
- **节点详解** → [WORKFLOW_DIAGRAM.md#各节点详细说明](WORKFLOW_DIAGRAM.md#三各节点详细说明)
- **数据字段** → [DATA_INPUT_CONTRACT.md#字段约定](DATA_INPUT_CONTRACT.md#3-字段约定32-字段)

## 文档约定

- 标题简短、层级清晰
- 优先使用列表与表格表达规则
- 路径统一使用相对路径
- 不记录本机隐私信息
