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

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
METRICS_DB_PATH = ROOT / "data" / "metrics.db"
RUNTIME_PARTS_DB_PATH = ROOT / "data" / "agent_parts.db"
CSV_JD_PATH = ROOT / "data" / "data_jd.csv"
CSV_NEWEGG_PATH = ROOT / "data" / "data_newegg.csv"

load_dotenv(ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


SESSION_STORE = os.getenv("SESSION_STORE", "sqlite").strip().lower()
SESSION_REDIS_URL = os.getenv("SESSION_REDIS_URL", "redis://127.0.0.1:6379/0").strip()
SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 604800)
SESSION_CLEANUP_INTERVAL_SECONDS = _env_int("SESSION_CLEANUP_INTERVAL_SECONDS", 3600)

BuildDataMode = Literal["jd_newegg", "jd", "newegg"]


def _bootstrap_parts_repo() -> None:
    if not CSV_JD_PATH.exists() or not CSV_NEWEGG_PATH.exists():
        missing = []
        if not CSV_JD_PATH.exists():
            missing.append(str(CSV_JD_PATH))
        if not CSV_NEWEGG_PATH.exists():
            missing.append(str(CSV_NEWEGG_PATH))
        raise RuntimeError(f"required CSV data file missing: {', '.join(missing)}")

    # 检查是否需要重建数据库（CSV 文件比数据库新，或数据库不存在）
    need_rebuild = False
    if not RUNTIME_PARTS_DB_PATH.exists():
        need_rebuild = True
    else:
        csv_mtime = max(CSV_JD_PATH.stat().st_mtime, CSV_NEWEGG_PATH.stat().st_mtime)
        db_mtime = RUNTIME_PARTS_DB_PATH.stat().st_mtime
        if csv_mtime > db_mtime:
            need_rebuild = True

    if need_rebuild:
        result = rebuild_runtime_db(
            db_path=RUNTIME_PARTS_DB_PATH,
            jd_csv=CSV_JD_PATH,
            newegg_csv=CSV_NEWEGG_PATH,
        )
        print(f"[RigForge] Database rebuilt: {result['rows_total']} parts loaded")
    

def _normalize_mode(mode: str | None) -> BuildDataMode:
    if mode in {"jd_newegg", "jd", "newegg"}:
        return mode
    return "jd_newegg"


def _build_graph_service(repo: SQLitePartsRepository, mode: BuildDataMode) -> ChatService:
    source_label = {
        "jd_newegg": "csv(jd+newegg)",
        "jd": "csv(jd)",
        "newegg": "csv(newegg)",
    }[mode]
    toolset = Toolset(
        repo,
        build_data_source=source_label,
        build_data_version="latest",
        build_data_mode=mode,
    )
    graph = RigForgeGraph(toolset)
    return ChatService(
        graph,
        metrics_db_path=METRICS_DB_PATH,
        session_store=SESSION_STORE if SESSION_STORE in {"memory", "sqlite", "redis"} else "sqlite",
        session_redis_url=SESSION_REDIS_URL,
        session_ttl_seconds=SESSION_TTL_SECONDS,
        session_cleanup_interval_seconds=SESSION_CLEANUP_INTERVAL_SECONDS,
    )


def _live_preview_build(mode: BuildDataMode, result) -> None:
    graph = services[mode].graph
    preview = pick_build_from_candidates(result.requirements, graph.tool_map["search_parts"])
    result.build = preview
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


_bootstrap_parts_repo()
repo_all = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH)
repo_jd = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH, source_sites={"jd"})
repo_newegg = SQLitePartsRepository(RUNTIME_PARTS_DB_PATH, source_sites={"newegg"})

services: Dict[BuildDataMode, ChatService] = {
    "jd_newegg": _build_graph_service(repo_all, "jd_newegg"),
    "jd": _build_graph_service(repo_jd, "jd"),
    "newegg": _build_graph_service(repo_newegg, "newegg"),
}
service = services["jd_newegg"]

app = FastAPI(title="RigForge｜锐格锻造坊")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(FRONTEND_DIR / "favicon.ico")


@app.post("/api/chat")
def chat(payload: ChatRequest):
    mode = _normalize_mode(payload.build_data_mode)
    routed_session_id = f"{mode}:{payload.session_id}"
    active_service = service if mode == "jd_newegg" else services[mode]
    result = active_service.chat(routed_session_id, payload.message, payload.enthusiasm_level)
    _live_preview_build(mode, result)
    return result.model_dump()


@app.get("/api/parts")
def list_parts():
    return [p.model_dump() for p in repo_all.all_parts()]


@app.get("/api/metrics")
def metrics():
    return service.metrics()
