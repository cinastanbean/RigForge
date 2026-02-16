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
- 两种交互模式（对话方式 / 组件选择）
- 对话式需求收集（LLM 动态设计问题）
- 性能追踪系统（详细的耗时分析）
- LLM 输入输出日志（便于调试和优化）

## UI Layout

- 左栏：需求画像 + 交互控制
- 中栏：聊天窗口
- 右栏：推荐配置 + 风险与估算

## Interaction Modes

### 对话方式（Chat Mode）

- LLM 动态设计问题，根据已收集信息智能提问
- 每次只问一个最关键的问题
- LLM 负责自然语言理解和回复生成
- 适合不熟悉硬件配置的用户

### 组件选择（Component Mode）

- 用户直接选择或描述需要的组件
- 系统根据用户选择生成配置
- 适合对硬件配置有一定了解的用户

## Conversation Flow

### 需求收集阶段

1. LLM 从用户输入中提取装机需求信息
2. LLM 判断是否需要继续提问收集更多信息
3. 如果需要继续，LLM 生成自然、友好的回复并提出下一个最关键的问题
4. 如果信息足够或用户拒绝继续，LLM 标记 `should_continue=false`，表示需求收集完成

### 停止条件

**用户主动结束**：
- 用户说"不用了"、"够了"、"就这样"、"不用再问"、"可以了"、"没问题"、"行"、"OK"、"ok"等表示结束的词
- 用户说"开始推荐"、"推荐吧"、"给我推荐"、"随便推荐"、"随便给我推荐"、"随便给我推荐个吧"等
- 用户说"随便"、"都可以"、"你看着办"、"你决定"等表示让系统决定

**系统判断结束**：
- 已收集信息中包含预算、用途、分辨率三个核心信息
- 对话轮数超过 10 轮
- 已收集 2 个关键信息且对话轮数超过 5 轮

### 推荐环节

当 `should_continue=false` 时，系统会：
1. 调用 `recommend_build`：根据需求生成配置方案
2. 调用 `validate_build`：验证配置的兼容性
3. 调用 `compose_reply`：生成推荐回复

## Performance Tracking

系统内置性能追踪功能，可以详细记录各个组件的执行时间：

- `collect_requirements`：需求收集时间
- `recommend_build`：配置推荐时间
- `validate_build`：配置验证时间
- `compose_reply`：回复生成时间
- 每个组件内部的详细时间（LLM 调用、数据库查询、数据处理等）

性能追踪输出示例：

```
============================================================
[PERF] collect_requirements completed in 2.345s
  - extract_and_reply: 1.234s (52.6%)
  - merge_requirements: 0.123s (5.2%)
  - route_decision: 0.012s (0.5%)
  - LLM setup: 0.987s (42.1%)
============================================================
```

## LLM Logging

系统记录所有 LLM 调用的输入输出信息，便于调试和优化：

**输入日志**：
- 已收集信息
- 用户输入
- 语气风格
- 其他上下文信息

**输出日志**：
- 需求更新（完整 JSON）
- 回复内容（完整）
- 继续提问（true/false）
- 调用耗时

日志格式示例：

```
============================================================
[LLM INPUT] 对话式需求收集
  已收集信息: 预算: 9000元, 用途: 办公
  用户输入: 我主要是用来办公的
  语气风格: 热情、友好、专业
============================================================

============================================================
[LLM OUTPUT] 对话式需求收集结果
  需求更新: {"use_case": ["office"], "use_case_set": true, "budget_min": 9000, "budget_max": 9000, "budget_set": true}
  回复内容: 明白了，办公用途已记录。请问您对显示器分辨率有什么要求吗？比如1080p、2K还是4K？
  继续提问: true
============================================================
```

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
| `ZHIPU_MODEL` | `GLM-4-Flash` | 智谱模型名称 |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/` | 智谱 API 地址 |
| `OPENROUTER_API_KEY` | - | OpenRouter API Key |
| `OPENROUTER_MODEL` | `openrouter/free` | OpenRouter 模型名称 |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API 地址 |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI 模型名称 |
| `LLM_RATE_LIMIT_ENABLED` | `false` | 是否启用 LLM 调用频率限制 |
| `LLM_TIMEOUT_SECONDS` | `12` | 模型超时 |
| `LLM_MAX_RETRIES` | `0` | 最大重试次数 |
| `LLM_MIN_INTERVAL_SECONDS` | `1.0` | 调用最小间隔 |
| `LLM_TURN_TIMEOUT_SECONDS` | `30` | 单轮超时 |
| `LLM_PROVIDER_PROBE_TIMEOUT_SECONDS` | `10` | 模型健康检查超时 |
| `LLM_PROVIDER_FAIL_COOLDOWN_SECONDS` | `120` | 模型失败冷却时间 |
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

### 测试说明

项目包含以下测试：

- `test_model_provider_routing.py`：测试模型路由和回退机制
- `test_llm_fallback.py`：测试 LLM 不可用时的规则回退
- `test_enthusiasm_level.py`：测试不同热情度档位的回复
- `test_budget_intent.py`：测试预算意图识别
- `test_followup_dialogue.py`：测试追问对话流程
- `test_compatibility.py`：测试兼容性检查
- `test_metrics.py`：测试会话指标统计
- `test_api_live_refresh.py`：测试 API 实时刷新

### 测试配置

测试时会自动启用 LLM 调用频率限制，避免并发测试导致 QPS 过高：

```python
os.environ["LLM_RATE_LIMIT_ENABLED"] = "true"
os.environ["LLM_MIN_INTERVAL_SECONDS"] = "1.25"
```

测试通过 `test_rate_limit` 装饰器确保每次 LLM 调用之间有最小间隔。

## Debugging

### 查看性能追踪日志

系统会自动输出性能追踪日志，格式如下：

```
============================================================
[PERF] collect_requirements completed in 2.345s
  - extract_and_reply: 1.234s (52.6%)
  - merge_requirements: 0.123s (5.2%)
  - route_decision: 0.012s (0.5%)
  - LLM setup: 0.987s (42.1%)
============================================================
```

### 查看 LLM 输入输出日志

系统会自动记录所有 LLM 调用的输入输出信息，格式如下：

```
============================================================
[LLM INPUT] 对话式需求收集
  已收集信息: 预算: 9000元, 用途: 办公
  用户输入: 我主要是用来办公的
  语气风格: 热情、友好、专业
============================================================

============================================================
[LLM OUTPUT] 对话式需求收集结果
  需求更新: {"use_case": ["office"], "use_case_set": true, "budget_min": 9000, "budget_max": 9000, "budget_set": true}
  回复内容: 明白了，办公用途已记录。请问您对显示器分辨率有什么要求吗？比如1080p、2K还是4K？
  继续提问: true
============================================================
```

### 常见问题

**Q: LLM 调用超时怎么办？**

A: 可以通过以下方式解决：
1. 增加 `LLM_TURN_TIMEOUT_SECONDS` 的值
2. 检查网络连接
3. 切换到更快的 LLM 模型

**Q: 大模型不按照指示提问怎么办？**

A: 可以通过以下方式解决：
1. 检查 Prompt 是否明确
2. 增加更多的示例
3. 切换到更听话的 LLM 模型

**Q: 聊天窗口没有显示大模型的回复怎么办？**

A: 可以通过以下方式解决：
1. 检查后台日志中的 `Route decision` 信息
2. 确认 `route` 是否为 `"ask_more"`
3. 检查 `response_text` 是否有值

**Q: 如何启用 LLM 调用频率限制？**

A: 在 `.env` 文件中设置 `LLM_RATE_LIMIT_ENABLED=true`

## Docs

- 文档导航：`docs/README.md`
- 快速启动：`FAST_START.md`
- 系统架构：`docs/SYSTEM_ARCHITECTURE.md`
- 需求设计：`docs/REQUIREMENTS_DESIGN.md`
