# RigForge（锐格锻造坊）

RigForge 是一个面向 PC 装机咨询的对话式 Agent，基于 `LangChain + LangGraph + FastAPI`。  
它通过多轮问答收集需求，并在聊天过程中实时更新推荐配置与风险提示。

## Development Note

- 本项目开发过程中采用 `vibe coding` 工作方式。
- 主要协作工具为 `Codex`。
- 本仓库记录的协作模型信息：`gpt-5.3-codex`。

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

