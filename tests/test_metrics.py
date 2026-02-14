import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.main import app
from rigforge.service import ChatService
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_service_metrics_grouped_by_enthusiasm_level(tmp_path):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    metrics_db = tmp_path / "metrics.db"
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph, metrics_db_path=metrics_db)

    service.chat("s1", "预算10000，2k游戏", enthusiasm_level="high")
    service.chat("s1", "不要显示器，1tb，尽量静音")
    service.chat("s2", "预算8000，办公，1080p，不要显示器，1tb，静音", enthusiasm_level="standard")

    metrics = service.metrics()
    assert metrics["total_sessions"] == 2
    assert metrics["total_turns"] == 3
    assert metrics["by_enthusiasm_level"]["high"]["sessions"] == 1
    assert metrics["by_enthusiasm_level"]["standard"]["sessions"] == 1
    assert "fallback" in metrics


def test_metrics_persist_after_service_restart(tmp_path):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    metrics_db = tmp_path / "metrics.db"

    graph1 = RigForgeGraph(Toolset(repo))
    graph1.llm = None
    service1 = ChatService(graph1, metrics_db_path=metrics_db)
    service1.chat("s1", "预算9000，2k游戏", enthusiasm_level="high")

    graph2 = RigForgeGraph(Toolset(repo))
    graph2.llm = None
    service2 = ChatService(graph2, metrics_db_path=metrics_db)
    metrics = service2.metrics()
    assert metrics["total_sessions"] == 1
    assert metrics["total_turns"] == 1


def test_metrics_endpoint_returns_200():
    client = TestClient(app)
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_sessions" in body
    assert "by_enthusiasm_level" in body


def test_sqlite_session_ttl_expired_state_not_loaded(tmp_path):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    metrics_db = tmp_path / "metrics.db"
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(
        graph,
        metrics_db_path=metrics_db,
        session_store="sqlite",
        session_ttl_seconds=5,
        session_cleanup_interval_seconds=3600,
    )

    service.chat("ttl-s1", "预算9000，2k游戏", enthusiasm_level="high")
    with sqlite3.connect(metrics_db) as conn:
        conn.execute(
            """
            UPDATE session_state
            SET updated_at = datetime('now', '-999 seconds')
            WHERE session_id = ?
            """,
            ("ttl-s1",),
        )
        conn.commit()

    service.sessions.pop("ttl-s1", None)
    service.chat("ttl-s1", "你好")
    assert service.sessions["ttl-s1"].turns == 1


def test_sqlite_cleanup_removes_expired_session_rows(tmp_path):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    metrics_db = tmp_path / "metrics.db"
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(
        graph,
        metrics_db_path=metrics_db,
        session_store="sqlite",
        session_ttl_seconds=10,
        session_cleanup_interval_seconds=3600,
    )

    requirements = json.dumps({}, ensure_ascii=False)
    history = json.dumps([], ensure_ascii=False)
    template_history = json.dumps({}, ensure_ascii=False)
    with sqlite3.connect(metrics_db) as conn:
        conn.execute(
            """
            INSERT INTO session_state (
                session_id,
                requirements_json,
                history_json,
                enthusiasm_level,
                model_provider,
                model_status_detail,
                template_history_json,
                turns,
                has_recommendation,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-999 seconds'))
            """,
            ("expired-s1", requirements, history, "standard", None, "", template_history, 0, 0),
        )
        conn.commit()

    with sqlite3.connect(metrics_db) as conn:
        before = conn.execute("SELECT COUNT(*) FROM session_state WHERE session_id = ?", ("expired-s1",)).fetchone()[0]
    assert before == 1

    service._cleanup_expired_sessions(force=True)

    with sqlite3.connect(metrics_db) as conn:
        after = conn.execute("SELECT COUNT(*) FROM session_state WHERE session_id = ?", ("expired-s1",)).fetchone()[0]
    assert after == 0
