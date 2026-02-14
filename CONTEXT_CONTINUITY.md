# 项目上下文连续性文档（重建）

更新时间：2026-02-13
目录：`<project_root>`

## 1. 记录恢复结论

- 未在项目目录中找到“项目开始时的原始聊天记录/会话日志”文件。
- 当前连续上下文由以下信息重建：
  - 代码与文件修改时间线
  - 当前 Git 状态与提交
  - 已执行的联调与测试结果

## 2. 当前项目状态（可直接接力）

- 后端：FastAPI + LangGraph，入口：`src/rigforge/main.py`
- 前端：静态页面，目录：`frontend/`
- 测试：`PYTHONPATH=src pytest -q` 通过（6 passed）
- 当前分支：`main`
- 当前提交：`7c3559c 第一版`

## 3. 本轮已确认/已落地的关键变更

1. 前端请求失败问题处理
- 文件：`frontend/app.js`
- 增加了 API 基地址解析（本地非 8000 端口时自动请求 `http://127.0.0.1:8000`）
- 增加了请求异常捕获与更明确的错误提示

2. 默认模型设置
- 文件：`src/rigforge/graph.py`
- 智谱默认模型设置为：`glm-4.7-flash`

3. 项目环境变量加载
- 文件：`src/rigforge/main.py`
- 启动时加载项目根目录 `.env`（`load_dotenv(ROOT / ".env")`）

4. 项目环境变量文件
- 文件：`.env`
- 已配置：`ZHIPU_API_KEY`、`ZHIPU_MODEL=glm-4.7-flash`、`ZHIPU_BASE_URL`

## 4. 当前已知风险/注意事项

- 即使配置了 Key，模型请求仍可能被服务端限流（429），此时系统会回退到规则回复。
- `.env` 含敏感信息，禁止提交到公共仓库。
- 当前工作区有未提交改动：`.DS_Store`（可忽略或清理）。

## 5. 接续工作计划（建议）

1. 增强可观测性
- 在 `graph.py` 中区分记录：
  - LLM 正常回复
  - LLM 调用异常（限流/鉴权/网络）
  - 规则回退触发原因

2. 优化前端提示
- 当后端回退时，前端展示“模型限流，已切换本地规则”的明确提示。

3. 配置治理
- 新增 `.gitignore` 规则确保 `.env`、`__pycache__`、`.pytest_cache` 不入库。

4. 启动与验证脚本化
- 增加 `scripts/run_local.sh`：
  - 启动服务
  - 健康检查
  - 一次 `/api/chat` 冒烟测试

## 6. 快速操作命令（下一位接手者）

```bash
cd /path/to/<project_root>
PYTHONPATH=src uvicorn rigforge.main:app --host 127.0.0.1 --port 8000
```

```bash
cd /path/to/<project_root>
PYTHONPATH=src pytest -q
```
