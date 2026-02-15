# RigForge（锐格锻造坊）

RigForge 是一个面向 PC 装机咨询的对话式 Agent，基于 `LangChain + LangGraph + FastAPI`。  
它通过多轮问答收集需求，并在聊天过程中实时更新推荐配置与风险提示。

## Development Note

- 本项目开发过程中采用 `vibe coding` 工作方式。
- 主要协作工具为 `Codex`。
- 本仓库记录的协作模型信息：`gpt-5.3-codex`。

## Design Philosophy

### 单一职责原则（Single Responsibility Principle）

在对话式 AI 系统中，我们遵循以下核心设计原则：

> **小心地切分你的处理流程，让模型只干一个简单的事情，不要让模型处理太复杂的事情，一次只让一个模型处理一个小的问题。这种方式在对话这种强交互的场景是合适的。避免让模型陷入比较深的思考。**

#### 具体实践

1. **任务分解**：将复杂的业务逻辑拆分为多个独立的、简单的任务
   - 需求提取：只负责从用户输入中提取结构化信息
   - 路由判断：只负责决定下一步的处理流程
   - 配置推荐：只负责从数据库中搜索合适的配件
   - 回复生成：只负责生成自然语言回复

2. **避免过度依赖 LLM**：不要让 LLM 承担所有逻辑判断
   - 使用规则引擎进行关键决策（如是否结束对话）
   - 使用数据库查询替代 LLM 推理（如配件选择）
   - 使用模板化回复替代 LLM 生成（如简单确认）

3. **双重验证机制**：LLM 判断 + 规则引擎验证
   - LLM 负责灵活的语义理解
   - 规则引擎负责可靠的逻辑判断
   - 优先使用规则引擎的判断，避免 LLM 误判

4. **超时保护**：为每个 LLM 调用设置合理的超时时间
   - 避免因 LLM 响应慢导致系统卡住
   - 超时后自动 fallback 到备用方案

#### 优势

- **可靠性**：规则引擎提供稳定的逻辑判断
- **灵活性**：LLM 提供自然的语言理解
- **性能**：避免 LLM 处理复杂任务导致的超时
- **可维护性**：每个模块职责清晰，易于调试和优化

## Features

- 多轮需求收集（预算、用途、分辨率、显示器、存储、静音、CPU 偏好）
- 实时推荐预览（不等待会话结束）
- 模型路由与回退（智谱 -> OpenRouter -> 规则）
- 三数据路径模式（`newegg` / `jd` / `jd_newegg`）
- 会话持久化与指标统计（SQLite/Redis）

## UI Layout

- 左栏：需求画像
- 中栏：聊天窗口
- 右栏：推荐配置 + 风险与估算

## Frontend Rendering

- 当前前端采用原生静态页面方案，不使用 React/Vue 等前端框架。
- 后端通过 FastAPI 返回 `frontend/index.html`，并将 `frontend/` 目录挂载为 `/static` 提供静态资源。
- 浏览器端由 `frontend/app.js` 通过 `fetch('/api/chat')` 调用接口，并用 DOM 更新三栏内容（需求画像、聊天、推荐与风险）。

## Project Structure

```text
src/rigforge/          # 后端核心逻辑
frontend/               # 前端页面与交互
data/                   # 运行数据（CSV + runtime DB）
config/                 # 规则模板等配置
scripts/                # 服务启动脚本
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src
uvicorn rigforge.main:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Service Commands

```bash
./scripts/server.sh restart bg   # 后台启动/重启
./scripts/server.sh restart fg   # 前台启动/重启
./scripts/server.sh stop         # 停止
./scripts/server.sh status       # 状态
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ZHIPU_API_KEY` | - | 智谱 API Key |
| `OPENROUTER_API_KEY` | - | OpenRouter API Key |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `LLM_TIMEOUT_SECONDS` | `12` | 模型超时 |
| `LLM_MIN_INTERVAL_SECONDS` | `1.0` | 调用最小间隔 |
| `LLM_TURN_TIMEOUT_SECONDS` | `5` | 单轮超时 |
| `SESSION_STORE` | `sqlite` | 会话存储类型 |
| `SESSION_REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 地址 |
| `SESSION_TTL_SECONDS` | `604800` | 会话 TTL |

完整示例见：`.env.example`

## API

- `POST /api/chat`：聊天主接口
- `GET /api/metrics`：会话统计
- `GET /api/parts`：当前件库查看（调试）

## Data Source Modes

前端可选三种路径（默认 `newegg`）：

- `newegg`
- `jd`
- `jd_newegg`

服务启动时会读取：

- `data/data_jd.csv`
- `data/data_newegg.csv`

并自动重建：

- `data/agent_parts.db`

## Testing

```bash
export PYTHONPATH=src
pytest -q
```

## Docs

- 文档导航：`docs/README.md`
- 快速启动：`FAST_START.md`
