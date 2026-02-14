from pathlib import Path

from rigforge.db import PartsRepository
from rigforge.graph import RigForgeGraph
from rigforge.schemas import UserRequirements
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_cheaper_intent_reduces_budget_and_sets_budget_priority():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    first = graph.invoke("预算10000，2K游戏")
    second = graph.invoke("我想便宜点", requirements=first.requirements)

    assert second.requirements.priority == "budget"
    assert second.requirements.budget_max < first.requirements.budget_max


def test_resolution_only_2k_does_not_override_existing_budget():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    graph = RigForgeGraph(Toolset(repo))

    req = UserRequirements(
        budget_min=8500,
        budget_max=10000,
        budget_set=True,
        use_case=["gaming"],
        use_case_set=True,
        resolution="1080p",
        resolution_set=False,
    )
    out = graph.invoke("2k", requirements=req)

    assert out.requirements.budget_max == 10000
    assert out.requirements.budget_min == 8500
    assert out.requirements.resolution == "1440p"
    assert out.requirements.resolution_set is True
