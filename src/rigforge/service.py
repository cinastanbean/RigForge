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

from .graph import RigForgeGraph, PerformanceTracker
from .schemas import ChatResponse, UserRequirements


@dataclass
class SessionState:
    """
    会话状态数据类 - Session State Dataclass
    
    表示单个会话的状态信息，包括用户需求、对话历史、交互模式等。
    Represents state information of a single session, including user requirements, conversation history, interaction mode, etc.
    
    字段说明 Field Descriptions:
    - requirements: 用户需求对象
    - history: 对话历史记录
    - interaction_mode: 交互模式（对话/组件选择）
    - enthusiasm_level: 热情度（标准/高）
    - model_provider: 模型提供商
    - model_status_detail: 模型状态详情
    - template_history: 模板使用历史
    - turns: 对话轮次
    - has_recommendation: 是否已生成推荐
    """
    requirements: UserRequirements = field(default_factory=UserRequirements)
    """
    用户需求 - User Requirements
    
    当前的用户需求状态。
    Current user requirements state.
    """
    history: List[Dict[str, str]] = field(default_factory=list)
    """
    对话历史 - Conversation History
    
    对话历史记录列表，每条记录包含 role 和 content。
    List of conversation history records, each containing role and content.
    """
    interaction_mode: Literal["chat", "component"] = "chat"
    """
    交互模式 - Interaction Mode
    
    交互模式：chat（对话模式）、component（组件选择模式）。
    Interaction mode: chat or component.
    """
    enthusiasm_level: Literal["standard", "high"] = "standard"
    """
    热情度 - Enthusiasm Level
    
    回复的热情度：standard（标准）、high（高）。
    Reply enthusiasm level: standard or high.
    """
    model_provider: Literal["zhipu", "openrouter", "rules"] | None = None
    """
    模型提供商 - Model Provider
    
    使用的模型提供商：zhipu、openrouter 或 rules。
    Model provider used: zhipu, openrouter, or rules.
    """
    model_status_detail: str = ""
    """
    模型状态详情 - Model Status Detail
    
    模型状态的详细信息。
    Detailed information about model status.
    """
    template_history: Dict[str, List[int]] = field(default_factory=dict)
    """
    模板历史 - Template History
    
    模板使用历史记录。
    Template usage history.
    """
    turns: int = 0
    """
    对话轮次 - Conversation Turns
    
    已进行的对话轮次数。
    Number of conversation turns completed.
    """
    has_recommendation: bool = False
    """
    是否已推荐 - Has Recommendation
    
    标记是否已生成推荐配置。
    Flag indicating if recommendation has been generated.
    """


class ChatService:
    """
    聊天服务类 - Chat Service Class
    
    负责管理聊天会话、处理用户消息、维护会话状态和记录指标。
    Manages chat sessions, processes user messages, maintains session state, and records metrics.
    
    主要功能 Main Functions:
    1. 会话管理：创建、加载、保存、清理会话
    2. 消息处理：调用图引擎处理用户消息
    3. 状态维护：维护会话状态和对话历史
    4. 指标记录：记录聊天事件和统计信息
    5. 并发控制：使用锁机制保证线程安全
    
    会话存储方式 Session Storage:
    - memory: 内存存储（默认，仅用于开发/测试）
    - sqlite: SQLite 数据库存储（推荐用于生产环境）
    - redis: Redis 存储（适用于分布式环境）
    """
    
    def __init__(
        self,
        graph: RigForgeGraph,
        metrics_db_path: Path | None = None,
        session_store: Literal["memory", "sqlite", "redis"] = "sqlite",
        session_redis_url: str | None = None,
        session_ttl_seconds: int | None = 7 * 24 * 3600,
        session_cleanup_interval_seconds: int = 3600,
    ):
        """
        初始化聊天服务 - Initialize chat service
        
        参数 Parameters:
            graph: 图引擎实例，用于处理用户消息
                   Graph engine instance for processing user messages
            metrics_db_path: 指标数据库路径，如果为 None 则不记录指标
                            Path to metrics database, if None then no metrics are recorded
            session_store: 会话存储方式，可选 "memory"、"sqlite" 或 "redis"
                          Session storage method, can be "memory", "sqlite", or "redis"
            session_redis_url: Redis 连接 URL，仅当 session_store="redis" 时使用
                              Redis connection URL, only used when session_store="redis"
            session_ttl_seconds: 会话过期时间（秒），默认为 7 天
                                Session TTL in seconds, default is 7 days
            session_cleanup_interval_seconds: 会话清理间隔（秒），默认为 1 小时
                                           Session cleanup interval in seconds, default is 1 hour
        """
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

        # 初始化 Redis 客户端 - Initialize Redis client
        if self.session_store == "redis":
            if Redis is None:
                raise RuntimeError("session_store=redis requires 'redis' package installed.")
            if not session_redis_url:
                raise ValueError("session_store=redis requires session_redis_url.")
            self._redis_client = Redis.from_url(session_redis_url, decode_responses=True)
            # startup connectivity check for fail-fast behavior
            self._redis_client.ping()

        # 初始化指标数据库 - Initialize metrics database
        if self.metrics_db_path:
            self._init_metrics_table()
            if self.session_store == "sqlite":
                self._init_session_table()
                self._cleanup_expired_sessions(force=True)

        # 缓存模型提供商信息 - Cache model provider information
        self._cached_model_provider: Literal["zhipu", "openrouter", "rules"] | None = None
        self._cached_model_status_detail: str = ""
        self._initialize_model_provider()

    def _initialize_model_provider(self):
        """
        初始化模型提供商 - Initialize model provider
        
        在服务启动时检查并缓存模型提供商信息，避免每次会话都检查。
        Check and cache model provider information at service startup to avoid checking for every session.
        """
        provider, detail = self.graph.select_provider_for_session()
        self._cached_model_provider = provider
        self._cached_model_status_detail = detail

    def chat(
        self,
        session_id: str,
        message: str,
        interaction_mode: Literal["chat", "component"] | None = None,
        enthusiasm_level: Literal["standard", "high"] | None = None,
    ) -> ChatResponse:
        """
        处理聊天消息 - Process chat message
        
        处理用户消息，更新会话状态，调用图引擎生成回复，并记录指标。
        Process user message, update session state, call graph engine to generate reply, and record metrics.
        
        处理流程 Processing Flow:
        1. 获取会话锁，确保线程安全
        2. 加载或创建会话状态
        3. 更新会话配置（交互模式、热情度）
        4. 获取上一条助手回复
        5. 调用图引擎处理消息
        6. 更新会话状态（需求、历史、轮次）
        7. 记录指标事件
        8. 保存会话状态
        9. 清理过期会话和内存缓存
        10. 返回聊天响应
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
            message: 用户消息
                     User message
            interaction_mode: 交互模式，如果为 None 则使用会话的默认模式
                            Interaction mode, if None then use session's default mode
            enthusiasm_level: 热情度，如果为 None 则使用会话的默认热情度
                            Enthusiasm level, if None then use session's default level
        
        返回 Returns:
            聊天响应，包含回复、需求、配置方案等信息
            Chat response, including reply, requirements, build plan, etc.
        """
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            now = time.monotonic()
            self._session_last_seen[session_id] = now
            
            # 创建性能跟踪器 - Create performance tracker
            tracker = PerformanceTracker(f"chat session {session_id}")
            
            # 加载或创建会话 - Load or create session
            session = self.sessions.get(session_id)
            if session is None:
                session = self._load_session_state(session_id)
            if session is None:
                session = SessionState()
            self.sessions[session_id] = session
            
            # 更新会话配置 - Update session configuration
            if interaction_mode in ("chat", "component"):
                session.interaction_mode = interaction_mode
            if enthusiasm_level in ("standard", "high"):
                session.enthusiasm_level = enthusiasm_level
            if session.model_provider is None:
                session.model_provider = self._cached_model_provider
                session.model_status_detail = self._cached_model_status_detail

            # 获取上一条助手回复 - Get last assistant reply
            last_assistant_reply = ""
            for item in reversed(session.history):
                if item.get("role") == "assistant":
                    last_assistant_reply = item.get("content", "")
                    break

            # 调用图引擎处理消息 - Call graph engine to process message
            next_turn = session.turns + 1
            result = self.graph.invoke(
                message,
                requirements=session.requirements,
                enthusiasm_level=session.enthusiasm_level,
                turn_number=next_turn,
                last_assistant_reply=last_assistant_reply,
                model_provider=session.model_provider,
                template_history=session.template_history,
                interaction_mode=session.interaction_mode,
            )
            
            # 更新会话状态 - Update session state
            session.requirements = result.requirements
            session.template_history = result.template_history
            session.turns = next_turn
            if result.build.cpu is not None:
                session.has_recommendation = True
            session.history.append({"role": "user", "content": message})
            session.history.append({"role": "assistant", "content": result.reply})
            
            # 完成性能跟踪 - Finish performance tracking
            tracker.finish()
            
            # 记录指标事件 - Record metric event
            self._record_metric_event(
                session_id=session_id,
                enthusiasm_level=session.enthusiasm_level,
                turn_number=session.turns,
                has_recommendation=session.has_recommendation,
                response_mode=result.response_mode,
                fallback_reason=result.fallback_reason,
            )
            
            # 保存会话状态 - Save session state
            self._save_session_state(session_id, session)
            
            # 清理过期会话和内存缓存 - Cleanup expired sessions and memory cache
            self._cleanup_expired_sessions()
            self._cleanup_in_memory_cache()
            
            # 返回结果 - Return result
            result.model_status_detail = session.model_status_detail
            return result

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """
        获取会话锁 - Get session lock
        
        为指定会话获取或创建锁，用于保证线程安全。
        Get or create lock for specified session, used to ensure thread safety.
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
        
        返回 Returns:
            会话锁对象
            Session lock object
        """
        with self._sessions_lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            self._lock_last_seen[session_id] = time.monotonic()
            return lock

    def metrics(self) -> dict:
        """
        获取指标统计 - Get metrics statistics
        
        返回聊天服务的指标统计信息，包括总会话数、总轮次、平均轮次等。
        Returns metrics statistics of chat service, including total sessions, total turns, average turns, etc.
        
        返回 Returns:
            指标统计字典
            Metrics statistics dictionary
        """
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
        """
        初始化指标表 - Initialize metrics table
        
        在 SQLite 数据库中创建聊天事件表，用于记录指标。
        Create chat events table in SQLite database for recording metrics.
        """
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
        """
        初始化会话表 - Initialize session table
        
        在 SQLite 数据库中创建会话状态表，用于持久化会话状态。
        Create session state table in SQLite database for persisting session state.
        """
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
        """
        加载会话状态 - Load session state
        
        从存储中加载会话状态。
        Load session state from storage.
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
        
        返回 Returns:
            会话状态对象，如果不存在则返回 None
            Session state object, or None if not found
        """
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
        """
        保存会话状态 - Save session state
        
        将会话状态保存到存储中。
        Save session state to storage.
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
            session: 会话状态对象
                     Session state object
        """
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
        """
        生成 Redis 键 - Generate Redis key
        
        为会话 ID 生成 Redis 键。
        Generate Redis key for session ID.
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
        
        返回 Returns:
            Redis 键字符串
            Redis key string
        """
        return f"rigforge:session:{session_id}"

    def _cleanup_expired_sessions(self, force: bool = False) -> None:
        """
        清理过期会话 - Cleanup expired sessions
        
        清理 SQLite 数据库中的过期会话。
        Cleanup expired sessions in SQLite database.
        
        参数 Parameters:
            force: 是否强制清理，忽略时间间隔
                   Whether to force cleanup, ignoring time interval
        """
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
        """
        清理内存缓存 - Cleanup in-memory cache
        
        清理内存中的过期会话和锁。
        Cleanup expired sessions and locks in memory.
        
        参数 Parameters:
            force: 是否强制清理，忽略时间间隔
                   Whether to force cleanup, ignoring time interval
        """
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
        """
        记录指标事件 - Record metric event
        
        记录聊天事件到指标数据库。
        Record chat event to metrics database.
        
        参数 Parameters:
            session_id: 会话 ID
                       Session ID
            enthusiasm_level: 热情度
                              Enthusiasm level
            turn_number: 轮次号
                         Turn number
            has_recommendation: 是否已推荐
                                Whether recommendation has been generated
            response_mode: 响应模式
                            Response mode
            fallback_reason: 回退原因
                             Fallback reason
        """
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
        """
        从数据库获取指标 - Get metrics from database
        
        从 SQLite 数据库中读取并计算指标统计信息。
        Read and calculate metrics statistics from SQLite database.
        
        返回 Returns:
            指标统计字典
            Metrics statistics dictionary
        """
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
