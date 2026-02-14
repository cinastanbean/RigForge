from pathlib import Path

from fastapi.testclient import TestClient

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.main import app
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


class BrokenLLM:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("rate limited")


class BrokenChain:
    def __or__(self, _other):
        return self

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("rate limited")


def test_compose_reply_falls_back_when_llm_fails(monkeypatch):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    g = RigForgeGraph(Toolset(repo))
    g.llm = BrokenLLM()

    monkeypatch.setattr("rigforge.graph.ChatPromptTemplate.from_messages", lambda *_args, **_kwargs: BrokenChain())

    first = g.invoke("预算10000，2K游戏")
    second = g.invoke("不需要显示器，1TB，尽量静音", requirements=first.requirements)
    assert "推荐方案已生成" in second.reply
    assert second.response_mode == "fallback"
    assert second.fallback_reason == "rate_limited"


def test_api_still_returns_200_when_model_errors(monkeypatch):
    class BrokenService:
        def chat(self, _sid, _msg, _enthusiasm_level=None):
            from rigforge.schemas import ChatResponse, UserRequirements, BuildPlan
            return ChatResponse(reply="fallback", requirements=UserRequirements(), build=BuildPlan())

    monkeypatch.setattr("rigforge.main.service", BrokenService())
    client = TestClient(app)
    resp = client.post("/api/chat", json={"session_id": "x", "message": "hi"})
    assert resp.status_code == 200
