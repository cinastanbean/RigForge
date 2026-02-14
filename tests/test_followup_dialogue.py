from pathlib import Path

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.schemas import RequirementUpdate, UserRequirements
from rigforge.service import ChatService
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_first_turn_asks_proactive_questions():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    out = graph.invoke("想配一台电脑")

    assert out.build.total_price() == 0
    assert "预算" in out.reply or "用途" in out.reply


def test_can_reach_recommendation_after_collecting_fields():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    r1 = graph.invoke("预算10000，主要2K游戏")
    r2 = graph.invoke("不需要显示器，1TB就够，尽量静音", requirements=r1.requirements)

    assert r2.build.cpu is not None
    assert r2.requirements.monitor_set is True
    assert r2.requirements.storage_set is True
    assert r2.requirements.noise_set is True


def test_stop_followup_intent_goes_straight_to_recommendation():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    r1 = graph.invoke("预算10000，主要2K游戏")
    r2 = graph.invoke("其他我都不知道，你直接推荐吧", requirements=r1.requirements)

    assert r2.build.cpu is not None
    assert r2.build.total_price() > 0


def test_monitor_included_in_budget_phrase_is_recognized():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    out = graph.invoke("把显示器也算进预算里")

    assert out.requirements.need_monitor is True
    assert out.requirements.monitor_set is True
    assert "显示器也算进预算里" not in out.reply


def test_direct_recommend_still_requires_budget_and_use_case():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    out = graph.invoke("其他都不知道，你直接推荐")

    assert out.build.total_price() == 0
    assert "预算" in out.reply or "用途" in out.reply


def test_high_cooperation_can_ask_two_questions():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    out = graph.invoke("预算9000，2K游戏", enthusiasm_level="standard")

    # High cooperation turn may ask up to two key missing items.
    assert out.reply.count("\n") >= 2


def test_semantic_values_without_set_flags_should_not_repeat_questions(monkeypatch):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    def fake_extract(_text, _current):
        return RequirementUpdate(need_monitor=True, need_quiet=True)

    monkeypatch.setattr(graph.extractor, "extract", fake_extract)
    out = graph.invoke("把显示器也算进预算里，希望风扇噪音尽量小", requirements=UserRequirements())

    assert out.requirements.monitor_set is True
    assert out.requirements.noise_set is True
    assert "显示器" not in out.reply
    assert "静音" not in out.reply


def test_monitor_phrase_still_works_when_extractor_returns_empty(monkeypatch):
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    def fake_extract(_text, _current):
        return RequirementUpdate()

    monkeypatch.setattr(graph.extractor, "extract", fake_extract)
    out = graph.invoke("把显示器也算进预算里")

    assert out.requirements.need_monitor is True
    assert out.requirements.monitor_set is True


def test_short_yes_answer_maps_to_monitor_from_previous_question():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s-monitor", "预算9000，2k游戏")
    second = service.chat("s-monitor", "要")

    assert "显示器" in first.reply
    assert second.requirements.need_monitor is True
    assert second.requirements.monitor_set is True
    assert "要不要把显示器也算进预算里" not in second.reply


def test_hao_de_answer_maps_to_monitor_from_previous_question():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s-monitor-hao-de", "预算9000，2k游戏")
    second = service.chat("s-monitor-hao-de", "好的。")

    assert "显示器" in first.reply
    assert second.requirements.need_monitor is True
    assert second.requirements.monitor_set is True
    assert "要不要把显示器也算进预算里" not in second.reply


def test_generic_continue_switches_to_next_missing_question():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s-generic-continue", "预算9000")
    second = service.chat("s-generic-continue", "好的")

    assert "主要做什么" in first.reply or "游戏、办公、剪辑" in first.reply
    assert "主要做什么" not in second.reply
    assert "分辨率" in second.reply or "刷新率" in second.reply


def test_rule_templates_do_not_repeat_in_same_session():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s-template-repeat", "好的")
    second = service.chat("s-template-repeat", "好的")

    assert first.reply != second.reply


def test_numeric_budget_reply_after_budget_question_is_applied():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None
    service = ChatService(graph)

    first = service.chat("s-budget-numeric", "主要2k游戏")
    second = service.chat("s-budget-numeric", "9000")

    assert "预算" in first.reply
    assert second.requirements.budget_set is True
    assert second.requirements.budget_max == 9000
    assert second.requirements.budget_min == int(9000 * 0.85)
    assert "预算范围" not in second.reply


def test_cpu_preference_intel_is_applied_to_cpu_selection():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    out = graph.invoke("预算10000，2K游戏，不需要显示器，1TB，静音，Intel")

    assert out.requirements.cpu_preference == "Intel"
    assert out.build.cpu is not None
    assert out.build.cpu.brand.lower() == "intel"


def test_cpu_preference_lowercase_intel_is_normalized_and_applied():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))
    graph.llm = None

    out = graph.invoke("预算10000，2K游戏，不需要显示器，1TB，静音，intel")

    assert out.requirements.cpu_preference == "Intel"
    assert out.build.cpu is not None
    assert out.build.cpu.brand.lower() == "intel"
