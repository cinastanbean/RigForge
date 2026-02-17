"""
RigForge 主程序 - RigForge Main Application

FastAPI 应用程序，提供 Web API 和前端服务。
FastAPI application providing Web API and frontend services.

主要功能 Main Functions:
1. 初始化配件数据库和聊天服务
2. 提供 RESTful API 接口
3. 支持多种数据源模式（京东、Newegg、混合）
4. 会话管理和指标记录
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .csv_runtime_db import rebuild_runtime_db
from .db import SQLitePartsRepository
from .graph import RigForgeGraph
from .schemas import ChatRequest
from .service import ChatService
from .tools import Toolset, pick_build_from_candidates

# 项目根目录 - Project Root Directory
ROOT = Path(__file__).resolve().parents[2]
"""
项目根目录 - Project Root Directory

RigForge 项目的根目录路径。
Root directory path of the RigForge project.
"""

# 前端目录 - Frontend Directory
FRONTEND_DIR = ROOT / "frontend"
"""
前端目录 - Frontend Directory

存放前端静态文件的目录路径。
Directory path storing frontend static files.
"""

# 指标数据库路径 - Metrics Database Path
METRICS_DB_PATH = ROOT / "data" / "metrics.db"
"""
指标数据库路径 - Metrics Database Path

存储聊天指标和统计信息的数据库文件路径。
Database file path storing chat metrics and statistics.
"""

# 运行时配件数据库路径 - Runtime Parts Database Path
RUNTIME_PARTS_DB_PATH = ROOT / "data" / "agent_parts.db"
"""
运行时配件数据库路径 - Runtime Parts Database Path

从 CSV 文件构建的 SQLite 数据库，用于快速查询配件信息。
SQLite database built from CSV files for fast parts information queries.
"""

# CSV 数据文件路径 - CSV Data File Paths
CSV_JD_PATH = ROOT / "data" / "data_jd.csv"
"""
京东 CSV 数据文件路径 - JD CSV Data File Path

京东配件数据的 CSV 文件路径。
CSV file path for JD parts data.
"""

CSV_NEWEGG_PATH = ROOT / "data" / "data_newegg.csv"
"""
Newegg CSV 数据文件路径 - Newegg CSV Data File Path

Newegg 配件数据的 CSV 文件路径。
CSV file path for Newegg parts data.
"""

# 加载环境变量 - Load Environment Variables
load_dotenv(ROOT / ".env")
"""
加载 .env 文件中的环境变量。
Load environment variables from .env file.
"""


def _env_int(name: str, default: int) -> int:
    """
    从环境变量读取整数 - Read Integer from Environment Variable
    
    从环境变量中读取整数值，如果读取失败则返回默认值。
    Read integer value from environment variable, return default if reading fails.
    
    参数 Parameters:
        name: 环境变量名称
              Environment variable name
        default: 默认值
                 Default value
    
    返回 Returns:
        环境变量值或默认值
        Environment variable value or default
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 会话存储配置 - Session Storage Configuration
SESSION_STORE = os.getenv("SESSION_STORE", "sqlite").strip().lower()
"""
会话存储方式 - Session Storage Method

会话数据的存储方式，可选值：memory、sqlite、redis。
Storage method for session data, options: memory, sqlite, redis.
"""

SESSION_REDIS_URL = os.getenv("SESSION_REDIS_URL", "redis://127.0.0.1:6379/0").strip()
"""
Redis 连接 URL - Redis Connection URL

当 SESSION_STORE="redis" 时使用的 Redis 连接地址。
Redis connection address used when SESSION_STORE="redis".
"""

SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 604800)
"""
会话过期时间（秒）- Session TTL in Seconds

会话数据的过期时间，默认为 7 天（604800 秒）。
Expiration time for session data, default is 7 days (604800 seconds).
"""

SESSION_CLEANUP_INTERVAL_SECONDS = _env_int("SESSION_CLEANUP_INTERVAL_SECONDS", 3600)
"""
会话清理间隔（秒）- Session Cleanup Interval in Seconds

清理过期会话的时间间隔，默认为 1 小时（3600 秒）。
Time interval for cleaning expired sessions, default is 1 hour (3600 seconds).
"""

# 构建数据模式 - Build Data Mode
BuildDataMode = Literal["jd_newegg", "jd", "newegg"]
"""
构建数据模式 - Build Data Mode

配件数据源模式：
Build data source modes:
- jd_newegg: 混合京东和 Newegg 数据
- jd: 仅使用京东数据
- newegg: 仅使用 Newegg 数据
"""


def _bootstrap_parts_repo() -> None:
    """
    初始化配件仓库 - Bootstrap Parts Repository
    
    检查 CSV 数据文件是否存在，如果数据库不存在或 CSV 文件比数据库新，则重建数据库。
    Check if CSV data files exist, rebuild database if database doesn't exist or CSV files are newer than database.
    
    异常 Raises:
        RuntimeError: 如果必需的 CSV 数据文件缺失
                      If required CSV data files are missing
    """
    # 检查 CSV 文件是否存在 - Check if CSV files exist
    if not CSV_JD_PATH.exists() or not CSV_NEWEGG_PATH.exists():
        missing = []
        if not CSV_JD_PATH.exists():
            missing.append(str(CSV_JD_PATH))
        if not CSV_NEWEGG_PATH.exists():
            missing.append(str(CSV_NEWEGG_PATH))
        raise RuntimeError(f"required CSV data file missing: {', '.join(missing)}")

    # 检查是否需要重建数据库（CSV 文件比数据库新，或数据库不存在）
    # Check if database needs to be rebuilt (CSV files are newer than database, or database doesn't exist)
    need_rebuild = False
    if not RUNTIME_PARTS_DB_PATH.exists():
        need_rebuild = True
    else:
        csv_mtime = max(CSV_JD_PATH.stat().st_mtime, CSV_NEWEGG_PATH.stat().st_mtime)
        db_mtime = RUNTIME_PARTS_DB_PATH.stat().st_mtime
        if csv_mtime > db_mtime:
            need_rebuild = True

    # 如果需要重建数据库 - If database needs to be rebuilt
    if need_rebuild:
        result = rebuild_runtime_db(
            db_path=RUNTIME_PARTS_DB_PATH,
            jd_csv=CSV_JD_PATH,
            newegg_csv=CSV_NEWEGG_PATH,
        )
        print(f"[RigForge] Database rebuilt: {result['rows_total']} parts loaded")
    

def _normalize_mode(mode: str | None) -> BuildDataMode:
    """
    标准化数据模式 - Normalize Data Mode
    
    将输入的模式字符串标准化为有效的 BuildDataMode。
    Normalize input mode string to valid BuildDataMode.
    
    参数 Parameters:
        mode: 输入的模式字符串
              Input mode string
    
    返回 Returns:
        标准化后的模式
        Normalized mode
    """
    if mode in {"jd_newegg", "jd", "newegg"}:
        return mode
    return "jd_newegg"


def _build_graph_service(repo: SQLitePartsRepository, mode: BuildDataMode) -> ChatService:
    """
    构建图服务 - Build Graph Service
    
    根据配件仓库和数据模式，创建聊天服务实例。
    Create chat service instance based on parts repository and data mode.
    
    参数 Parameters:
        repo: 配件仓库实例
              Parts repository instance
        mode: 数据模式
              Data mode
    
    返回 Returns:
        聊天服务实例
        Chat service instance
    """
    source_label = {
        "jd_newegg": "csv(jd+newegg)",
        "jd": "csv(jd)",
        "newegg": "csv(newegg)",
    }[mode]
    
    # 创建工具集 - Create toolset
    toolset = Toolset(
        repo,
        build_data_source=source_label,
        build_data_version="latest",
        build_data_mode=mode,
    )
    
    # 创建图引擎 - Create graph engine
    graph = RigForgeGraph(toolset)
    
    # 创建聊天服务 - Create chat service
    return ChatService(
        graph,
        metrics_db_path=METRICS_DB_PATH,
        session_store=SESSION_STORE if SESSION_STORE in {"memory", "sqlite", "redis"} else "sqlite",
        session_redis_url=SESSION_REDIS_URL,
        session_ttl_seconds=SESSION_TTL_SECONDS,
        session_cleanup_interval_seconds=SESSION_CLEANUP_INTERVAL_SECONDS,
    )


def _live_preview_build(mode: BuildDataMode, result) -> None:
    """
    实时预览构建 - Live Preview Build
    
    根据当前需求生成配置方案预览。
    Generate build plan preview based on current requirements.
    
    参数 Parameters:
        mode: 数据模式
              Data mode
        result: 聊天结果对象
                Chat result object
    """
    graph = services[mode].graph
    preview = pick_build_from_candidates(result.requirements, graph.tool_map["search_parts"])
    result.build = preview
    
    # 计算预估功耗 - Calculate estimated power consumption
    skus = [
        p.sku
        for p in [
            preview.cpu,
            preview.motherboard,
            preview.memory,
            preview.storage,
            preview.gpu,
            preview.psu,
            preview.case,
            preview.cooler,
        ]
        if p is not None
    ]
    if skus:
        result.estimated_power = graph.tool_map["estimate_power"].invoke({"parts": skus})


# 初始化配件仓库 - Initialize Parts Repository
_bootstrap_parts_repo()
"""
初始化配件仓库 - Initialize Parts Repository

启动时初始化配件仓库，确保数据库是最新的。
Initialize parts repository on startup to ensure database is up-to-date.
"""

# 创建配件仓库实例 - Create Parts Repository Instances
repo_all = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH)
"""
所有数据源仓库 - All Data Sources Repository

包含京东和 Newegg 所有数据的配件仓库。
Parts repository containing all data from JD and Newegg.
"""

repo_jd = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH, source_sites={"jd"})
"""
京东数据源仓库 - JD Data Source Repository

仅包含京东数据的配件仓库。
Parts repository containing only JD data.
"""

repo_newegg = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH, source_sites={"newegg"})
"""
Newegg 数据源仓库 - Newegg Data Source Repository

仅包含 Newegg 数据的配件仓库。
Parts repository containing only Newegg data.
"""

# 创建聊天服务实例 - Create Chat Service Instances
services: Dict[BuildDataMode, ChatService] = {
    "jd_newegg": _build_graph_service(repo_all, "jd_newegg"),
    "jd": _build_graph_service(repo_jd, "jd"),
    "newegg": _build_graph_service(repo_newegg, "newegg"),
}
"""
聊天服务字典 - Chat Services Dictionary

根据数据模式存储不同的聊天服务实例。
Store different chat service instances based on data mode.
"""

service = services["jd_newegg"]
"""
默认聊天服务 - Default Chat Service

默认使用的聊天服务（混合数据源）。
Default chat service to use (mixed data sources).
"""

# 创建 FastAPI 应用 - Create FastAPI Application
app = FastAPI(title="RigForge｜锐格锻造坊")
"""
FastAPI 应用实例 - FastAPI Application Instance

RigForge 的 FastAPI 应用程序实例。
FastAPI application instance for RigForge.
"""

# 添加 CORS 中间件 - Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
CORS 中间件配置 - CORS Middleware Configuration

允许跨域请求，方便前端调用 API。
Allow cross-origin requests for easy frontend API calls.
"""

# 挂载静态文件 - Mount Static Files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
"""
静态文件挂载 - Static Files Mount

将前端目录挂载到 /static 路径。
Mount frontend directory to /static path.
"""


@app.get("/")
def index():
    """
    首页路由 - Index Route
    
    返回前端首页 HTML 文件。
    Return frontend index HTML file.
    
    返回 Returns:
        前端 HTML 文件
        Frontend HTML file
    """
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    """
    网站图标路由 - Favicon Route
    
    返回网站图标文件。
    Return website favicon file.
    
    返回 Returns:
        网站图标文件
        Website favicon file
    """
    return FileResponse(FRONTEND_DIR / "favicon.ico")


@app.post("/api/chat")
def chat(payload: ChatRequest):
    """
    聊天 API - Chat API
    
    处理用户聊天消息，返回 AI 回复和配置方案。
    Process user chat message, return AI response and build plan.
    
    参数 Parameters:
        payload: 聊天请求对象
                Chat request object
    
    返回 Returns:
        聊天响应结果
        Chat response result
    """
    mode = _normalize_mode(payload.build_data_mode)
    routed_session_id = f"{mode}:{payload.session_id}"
    active_service = service if mode == "jd_newegg" else services[mode]
    result = active_service.chat(routed_session_id, payload.message, payload.interaction_mode, payload.enthusiasm_level)
    _live_preview_build(mode, result)
    return result.model_dump()


@app.get("/api/parts")
def list_parts():
    """
    配件列表 API - Parts List API
    
    返回所有配件的列表。
    Return list of all parts.
    
    返回 Returns:
        配件列表
        List of parts
    """
    return [p.model_dump() for p in repo_all.all_parts()]


@app.get("/api/metrics")
def metrics():
    """
    指标 API - Metrics API
    
    返回聊天服务的指标和统计信息。
    Return chat service metrics and statistics.
    
    返回 Returns:
        指标和统计信息
        Metrics and statistics
    """
    return service.metrics()
