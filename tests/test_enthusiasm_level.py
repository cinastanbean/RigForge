from pathlib import Path

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.service import ChatService
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_graph_response_contains_selected_enthusiasm_level():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    out = graph.invoke("想配台电脑", enthusiasm_level="high")

    assert out.enthusiasm_level == "high"


def test_service_persists_enthusiasm_level_per_session():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s1", "预算9000，2K游戏", enthusiasm_level="high")
    second = service.chat("s1", "不需要显示器，1TB，尽量静音")

    assert first.enthusiasm_level == "high"
    assert second.enthusiasm_level == "high"


def test_high_enthusiasm_decays_after_two_turns():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    first = graph.invoke("想配台电脑", enthusiasm_level="high", turn_number=1)
    third = graph.invoke("想配台电脑", enthusiasm_level="high", turn_number=3)

    assert any(k in first.reply for k in ["太棒了", "非常好", "赞", "很棒", "太好了", "好极了"])
    assert any(k in third.reply for k in ["收到", "明白了", "了解", "好", "下一步"])
