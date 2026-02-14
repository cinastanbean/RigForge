from pathlib import Path

from rigforge.db import PartsRepository
from rigforge.tools import Toolset


ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_detects_socket_mismatch():
    repo = PartsRepository(ROOT / "data" / "parts.json")
    tools = Toolset(repo).register()

    issues = tools["check_compatibility"].invoke(
        {
            "cpu_sku": "CPU-7600",
            "motherboard_sku": "MB-B760",
            "memory_sku": "RAM-32-6000",
            "gpu_sku": "GPU-4070S",
            "psu_sku": "PSU-750G",
            "case_sku": "CASE-ATX-AIR",
            "cooler_sku": "COOLER-AG620",
        }
    )

    assert any("socket mismatch" in i for i in issues)
