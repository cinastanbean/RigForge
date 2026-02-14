from pathlib import Path

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.service import ChatService
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_provider_is_selected_once_per_session(monkeypatch):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    calls = {"count": 0}

    def fake_select():
        calls["count"] += 1
        return "zhipu", "智谱可用"

    monkeypatch.setattr(graph, "select_provider_for_session", fake_select)

    first = service.chat("s-provider", "预算9000，2k游戏")
    second = service.chat("s-provider", "好的")

    assert calls["count"] == 1
    assert first.session_model_provider == "zhipu"
    assert second.session_model_provider == "zhipu"
