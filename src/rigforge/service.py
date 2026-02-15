from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency at runtime
    Redis = None

from .graph import RigForgeGraph
from .schemas import ChatResponse, UserRequirements


@dataclass
class SessionState:
    requirements: UserRequirements = field(default_factory=UserRequirements)
    history: List[Dict[str, str]] = field(default_factory=list)
    enthusiasm_level: Literal["standard", "high"] = "standard"
    model_provider: Literal["zhipu", "openrouter", "rules"] | None = None
    model_status_detail: str = ""
    template_history: Dict[str, List[int]] = field(default_factory=dict)
    turns: int = 0
    has_recommendation: bool = False


class ChatService:
    def __init__(
        self,
        graph: RigForgeGraph,
        metrics_db_path: Path | None = None,
        session_store: Literal["memory", "sqlite", "redis"] = "sqlite",
        session_redis_url: str | None = None,
        session_ttl_seconds: int | None = 7 * 24 * 3600,
        session_cleanup_interval_seconds: int = 3600,
    ):
        self.graph = graph
        self.sessions: Dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._session_locks: Dict[str, threading.Lock] = {}
        self._session_last_seen: Dict[str, float] = {}
        self._lock_last_seen: Dict[str, float] = {}
        self._cleanup_lock = threading.Lock()
        self._last_session_cleanup_monotonic = 0.0
        self._last_memory_cleanup_monotonic = 0.0
        self.metrics_db_path = metrics_db_path
        self.session_store = session_store
        self._redis_client: Optional[Redis] = None
        self.session_ttl_seconds = max(0, int(session_ttl_seconds or 0))
        self.session_cleanup_interval_seconds = max(1, int(session_cleanup_interval_seconds))

        if self.session_store == "redis":
            if Redis is None:
                raise RuntimeError("session_store=redis requires 'redis' package installed.")
            if not session_redis_url:
                raise ValueError("session_store=redis requires session_redis_url.")
            self._redis_client = Redis.from_url(session_redis_url, decode_responses=True)
            # startup connectivity check for fail-fast behavior
            self._redis_client.ping()

        if self.metrics_db_path:
            self._init_metrics_table()
            if self.session_store == "sqlite":
                self._init_session_table()
                self._cleanup_expired_sessions(force=True)

        self._cached_model_provider: Literal["zhipu", "openrouter", "rules"] | None = None
        self._cached_model_status_detail: str = ""
        self._initialize_model_provider()

    def _initialize_model_provider(self):
        provider, detail = self.graph.select_provider_for_session()
        self._cached_model_provider = provider
        self._cached_model_status_detail = detail

    def chat(
        self,
        session_id: str,
        message: str,
        enthusiasm_level: Literal["standard", "high"] | None = None,
    ) -> ChatResponse:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            now = time.monotonic()
            self._session_last_seen[session_id] = now
            session = self.sessions.get(session_id)
            if session is None:
                session = self._load_session_state(session_id)
            if session is None:
                session = SessionState()
            self.sessions[session_id] = session
            if enthusiasm_level in ("standard", "high"):
                session.enthusiasm_level = enthusiasm_level
            if session.model_provider is None:
                session.model_provider = self._cached_model_provider
                session.model_status_detail = self._cached_model_status_detail

            last_assistant_reply = ""
            for item in reversed(session.history):
                if item.get("role") == "assistant":
                    last_assistant_reply = item.get("content", "")
                    break

            next_turn = session.turns + 1
            result = self.graph.invoke(
                message,
                requirements=session.requirements,
                enthusiasm_level=session.enthusiasm_level,
                turn_number=next_turn,
                last_assistant_reply=last_assistant_reply,
                model_provider=session.model_provider,
                template_history=session.template_history,
            )
            session.requirements = result.requirements
            session.template_history = result.template_history
            session.turns = next_turn
            if result.build.cpu is not None:
                session.has_recommendation = True
            session.history.append({"role": "user", "content": message})
            session.history.append({"role": "assistant", "content": result.reply})
            self._record_metric_event(
                session_id=session_id,
                enthusiasm_level=session.enthusiasm_level,
                turn_number=session.turns,
                has_recommendation=session.has_recommendation,
                response_mode=result.response_mode,
                fallback_reason=result.fallback_reason,
            )
            self._save_session_state(session_id, session)
            self._cleanup_expired_sessions()
            self._cleanup_in_memory_cache()
            result.model_status_detail = session.model_status_detail
            return result

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._sessions_lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            self._lock_last_seen[session_id] = time.monotonic()
            return lock

    def metrics(self) -> dict:
        if self.metrics_db_path:
            return self._metrics_from_db()

        total_sessions = len(self.sessions)
        total_turns = sum(s.turns for s in self.sessions.values())
        by_level: Dict[str, dict] = {}

        for level in ("standard", "high"):
            sessions = [s for s in self.sessions.values() if s.enthusiasm_level == level]
            count = len(sessions)
            turns = sum(s.turns for s in sessions)
            recommendation_sessions = sum(1 for s in sessions if s.has_recommendation)
            rate = round(recommendation_sessions / count, 4) if count else 0.0
            by_level[level] = {
                "sessions": count,
                "turns": turns,
                "avg_turns_per_session": round(turns / count, 2) if count else 0.0,
                "recommendation_sessions": recommendation_sessions,
                "recommendation_rate": rate,
            }

        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "avg_turns_per_session": round(total_turns / total_sessions, 2) if total_sessions else 0.0,
            "by_enthusiasm_level": by_level,
        }

    def _init_metrics_table(self) -> None:
        assert self.metrics_db_path is not None
        self.metrics_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    enthusiasm_level TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    has_recommendation INTEGER NOT NULL,
                    response_mode TEXT NOT NULL,
                    fallback_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _init_session_table(self) -> None:
        assert self.metrics_db_path is not None
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    requirements_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    enthusiasm_level TEXT NOT NULL,
                    model_provider TEXT,
                    model_status_detail TEXT NOT NULL,
                    template_history_json TEXT NOT NULL,
                    turns INTEGER NOT NULL,
                    has_recommendation INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _load_session_state(self, session_id: str) -> SessionState | None:
        if self.session_store == "memory":
            return None
        if self.session_store == "redis":
            assert self._redis_client is not None
            payload = self._redis_client.get(self._redis_key(session_id))
            if not payload:
                return None
            data = json.loads(payload)
            return SessionState(
                requirements=UserRequirements.model_validate(data["requirements"]),
                history=data["history"],
                enthusiasm_level=data["enthusiasm_level"],
                model_provider=data.get("model_provider"),
                model_status_detail=data.get("model_status_detail", ""),
                template_history=data.get("template_history", {}),
                turns=int(data.get("turns", 0)),
                has_recommendation=bool(data.get("has_recommendation", False)),
            )

        if not self.metrics_db_path:
            return None
        base_sql = """
            SELECT
                requirements_json,
                history_json,
                enthusiasm_level,
                model_provider,
                model_status_detail,
                template_history_json,
                turns,
                has_recommendation
            FROM session_state
            WHERE session_id = ?
        """
        params: tuple = (session_id,)
        if self.session_ttl_seconds > 0:
            base_sql += " AND updated_at >= datetime('now', ?)"
            params = (session_id, f"-{self.session_ttl_seconds} seconds")
        with sqlite3.connect(self.metrics_db_path) as conn:
            row = conn.execute(base_sql, params).fetchone()
        if row is None:
            return None
        return SessionState(
            requirements=UserRequirements.model_validate(json.loads(row[0])),
            history=json.loads(row[1]),
            enthusiasm_level=row[2],
            model_provider=row[3],
            model_status_detail=row[4] or "",
            template_history=json.loads(row[5]),
            turns=int(row[6]),
            has_recommendation=bool(row[7]),
        )

    def _save_session_state(self, session_id: str, session: SessionState) -> None:
        if self.session_store == "memory":
            return
        if self.session_store == "redis":
            assert self._redis_client is not None
            payload = {
                "requirements": session.requirements.model_dump(),
                "history": session.history,
                "enthusiasm_level": session.enthusiasm_level,
                "model_provider": session.model_provider,
                "model_status_detail": session.model_status_detail,
                "template_history": session.template_history,
                "turns": session.turns,
                "has_recommendation": session.has_recommendation,
            }
            key = self._redis_key(session_id)
            value = json.dumps(payload, ensure_ascii=False)
            if self.session_ttl_seconds > 0:
                self._redis_client.set(key, value, ex=self.session_ttl_seconds)
            else:
                self._redis_client.set(key, value)
            return

        if not self.metrics_db_path:
            return
        with sqlite3.connect(self.metrics_db_path) as conn:
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    requirements_json = excluded.requirements_json,
                    history_json = excluded.history_json,
                    enthusiasm_level = excluded.enthusiasm_level,
                    model_provider = excluded.model_provider,
                    model_status_detail = excluded.model_status_detail,
                    template_history_json = excluded.template_history_json,
                    turns = excluded.turns,
                    has_recommendation = excluded.has_recommendation,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    json.dumps(session.requirements.model_dump(), ensure_ascii=False),
                    json.dumps(session.history, ensure_ascii=False),
                    session.enthusiasm_level,
                    session.model_provider,
                    session.model_status_detail,
                    json.dumps(session.template_history, ensure_ascii=False),
                    session.turns,
                    1 if session.has_recommendation else 0,
                ),
            )
            conn.commit()

    @staticmethod
    def _redis_key(session_id: str) -> str:
        return f"rigforge:session:{session_id}"

    def _cleanup_expired_sessions(self, force: bool = False) -> None:
        if self.session_store != "sqlite":
            return
        if not self.metrics_db_path:
            return
        if self.session_ttl_seconds <= 0:
            return

        now = time.monotonic()
        if not force and (now - self._last_session_cleanup_monotonic) < self.session_cleanup_interval_seconds:
            return

        with self._cleanup_lock:
            now = time.monotonic()
            if not force and (now - self._last_session_cleanup_monotonic) < self.session_cleanup_interval_seconds:
                return
            with sqlite3.connect(self.metrics_db_path) as conn:
                conn.execute(
                    """
                    DELETE FROM session_state
                    WHERE updated_at < datetime('now', ?)
                    """,
                    (f"-{self.session_ttl_seconds} seconds",),
                )
                conn.commit()
            self._last_session_cleanup_monotonic = now

    def _cleanup_in_memory_cache(self, force: bool = False) -> None:
        if self.session_ttl_seconds <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_memory_cleanup_monotonic) < self.session_cleanup_interval_seconds:
            return
        with self._cleanup_lock:
            now = time.monotonic()
            if not force and (now - self._last_memory_cleanup_monotonic) < self.session_cleanup_interval_seconds:
                return
            expire_before = now - float(self.session_ttl_seconds)
            with self._sessions_lock:
                stale_sessions = [sid for sid, seen in self._session_last_seen.items() if seen < expire_before]
                for sid in stale_sessions:
                    self.sessions.pop(sid, None)
                    self._session_last_seen.pop(sid, None)
                    self._lock_last_seen.pop(sid, None)
                    lock = self._session_locks.get(sid)
                    if lock is not None and not lock.locked():
                        self._session_locks.pop(sid, None)

                stale_locks = [sid for sid, seen in self._lock_last_seen.items() if seen < expire_before]
                for sid in stale_locks:
                    lock = self._session_locks.get(sid)
                    if lock is not None and not lock.locked():
                        self._session_locks.pop(sid, None)
                        self._lock_last_seen.pop(sid, None)
            self._last_memory_cleanup_monotonic = now

    def _record_metric_event(
        self,
        session_id: str,
        enthusiasm_level: Literal["standard", "high"],
        turn_number: int,
        has_recommendation: bool,
        response_mode: str,
        fallback_reason: str | None,
    ) -> None:
        if not self.metrics_db_path:
            return
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute(
                """
                INSERT INTO chat_events (
                    session_id, enthusiasm_level, turn_number, has_recommendation, response_mode, fallback_reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    enthusiasm_level,
                    turn_number,
                    1 if has_recommendation else 0,
                    response_mode,
                    fallback_reason,
                ),
            )
            conn.commit()

    def _metrics_from_db(self) -> dict:
        assert self.metrics_db_path is not None
        with sqlite3.connect(self.metrics_db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    e.session_id AS session_id,
                    latest.level AS enthusiasm_level,
                    COUNT(*) AS turns,
                    MAX(e.has_recommendation) AS has_recommendation
                FROM chat_events e
                JOIN (
                    SELECT
                        ce.session_id AS session_id,
                        ce.enthusiasm_level AS level
                    FROM chat_events ce
                    JOIN (
                        SELECT session_id, MAX(id) AS max_id
                        FROM chat_events
                        GROUP BY session_id
                    ) latest_id
                    ON latest_id.max_id = ce.id
                ) latest
                ON latest.session_id = e.session_id
                GROUP BY e.session_id, latest.level
                """
            ).fetchall()

            fallback_rows = conn.execute(
                """
                SELECT
                    COALESCE(fallback_reason, 'none') AS reason,
                    COUNT(*) AS count
                FROM chat_events
                WHERE response_mode = 'fallback'
                GROUP BY COALESCE(fallback_reason, 'none')
                """
            ).fetchall()

        total_sessions = len(rows)
        total_turns = sum(int(r[2]) for r in rows)
        by_level: Dict[str, dict] = {
            "standard": {
                "sessions": 0,
                "turns": 0,
                "avg_turns_per_session": 0.0,
                "recommendation_sessions": 0,
                "recommendation_rate": 0.0,
            },
            "high": {
                "sessions": 0,
                "turns": 0,
                "avg_turns_per_session": 0.0,
                "recommendation_sessions": 0,
                "recommendation_rate": 0.0,
            },
        }
        for session_id, level, turns, has_recommendation in rows:
            _ = session_id
            info = by_level[level]
            info["sessions"] += 1
            info["turns"] += int(turns)
            info["recommendation_sessions"] += int(has_recommendation)

        for level in ("standard", "high"):
            sessions = by_level[level]["sessions"]
            turns = by_level[level]["turns"]
            rec_sessions = by_level[level]["recommendation_sessions"]
            by_level[level]["avg_turns_per_session"] = round(turns / sessions, 2) if sessions else 0.0
            by_level[level]["recommendation_rate"] = round(rec_sessions / sessions, 4) if sessions else 0.0

        fallback_by_reason = {reason: int(count) for reason, count in fallback_rows}

        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "avg_turns_per_session": round(total_turns / total_sessions, 2) if total_sessions else 0.0,
            "by_enthusiasm_level": by_level,
            "fallback": {
                "total": int(sum(fallback_by_reason.values())),
                "by_reason": fallback_by_reason,
            },
        }
