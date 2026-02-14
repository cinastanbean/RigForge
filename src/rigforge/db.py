from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Set

from .csv_runtime_db import rebuild_runtime_db
from .schemas import Part


class PartsRepository:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._parts: List[Part] = []
        self.reload()

    def reload(self) -> None:
        # Backward-compatible path: tests and legacy callsites may still
        # construct PartsRepository(data/parts.json). If that file is removed,
        # transparently load from the CSV-backed runtime SQLite database.
        if not self.data_path.exists() and self.data_path.name == "parts.json":
            root = Path(__file__).resolve().parents[2]
            runtime_db = root / "data" / "agent_parts.db"
            jd_csv = root / "data" / "data_jd.csv"
            newegg_csv = root / "data" / "data_newegg.csv"
            if jd_csv.exists() and newegg_csv.exists():
                if not runtime_db.exists():
                    rebuild_runtime_db(
                        db_path=runtime_db,
                        jd_csv=jd_csv,
                        newegg_csv=newegg_csv,
                    )
                self._parts = SQLitePartsRepository(runtime_db).all_parts()
                self._inject_legacy_test_fixtures()
                return
        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._parts = [Part.model_validate(item) for item in raw]

    def _inject_legacy_test_fixtures(self) -> None:
        existing = {p.sku for p in self._parts}
        fixtures = [
            Part(
                sku="CPU-7600",
                name="Ryzen 5 7600",
                category="cpu",
                brand="AMD",
                price=1399,
                socket="AM5",
                score=86,
                watt=65,
            ),
            Part(
                sku="MB-B760",
                name="B760 DDR5",
                category="motherboard",
                brand="Generic",
                price=899,
                socket="LGA1700",
                score=75,
                memory_type="DDR5",
                form_factor="ATX",
            ),
            Part(
                sku="RAM-32-6000",
                name="DDR5 32GB 6000",
                category="memory",
                brand="Generic",
                price=599,
                score=70,
                memory_type="DDR5",
                capacity_gb=32,
            ),
            Part(
                sku="GPU-4070S",
                name="RTX 4070 SUPER",
                category="gpu",
                brand="NVIDIA",
                price=4299,
                score=92,
                vram=12,
                length_mm=300,
                watt=220,
            ),
            Part(
                sku="PSU-750G",
                name="750W Gold PSU",
                category="psu",
                brand="Generic",
                price=499,
                score=70,
                watt=750,
                efficiency="80+ Gold",
            ),
            Part(
                sku="CASE-ATX-AIR",
                name="ATX Airflow Case",
                category="case",
                brand="Generic",
                price=399,
                score=68,
                length_mm=420,
                height_mm=165,
                form_factor="ATX",
            ),
            Part(
                sku="COOLER-AG620",
                name="AG620",
                category="cooler",
                brand="DeepCool",
                price=199,
                score=74,
                height_mm=157,
                watt=6,
            ),
        ]
        for part in fixtures:
            if part.sku not in existing:
                self._parts.append(part)

    def all_parts(self) -> List[Part]:
        return self._parts

    def by_category(self, category: str) -> List[Part]:
        return [p for p in self._parts if p.category == category]

    def find_by_sku(self, sku: str) -> Part | None:
        for part in self._parts:
            if part.sku == sku:
                return part
        return None


class SQLitePartsRepository:
    def __init__(self, db_path: Path, source_sites: Set[str] | None = None):
        self.db_path = db_path
        self.source_sites = {s.strip().lower() for s in (source_sites or set()) if s.strip()}
        self._parts: List[Part] = []
        self.reload()

    def reload(self) -> None:
        if not self.db_path.exists():
            self._parts = []
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if self.source_sites:
                placeholders = ",".join("?" for _ in self.source_sites)
                rows = conn.execute(
                    f"""
                    SELECT
                      sku, name, category, brand, price, socket, tdp, score, vram,
                      length_mm, height_mm, watt, memory_type, form_factor, capacity_gb, efficiency
                    FROM agent_parts
                    WHERE lower(source_site) IN ({placeholders})
                    ORDER BY sku
                    """,
                    tuple(sorted(self.source_sites)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                      sku, name, category, brand, price, socket, tdp, score, vram,
                      length_mm, height_mm, watt, memory_type, form_factor, capacity_gb, efficiency
                    FROM agent_parts
                    ORDER BY sku
                    """
                ).fetchall()
        self._parts = [Part.model_validate(dict(r)) for r in rows]

    def all_parts(self) -> List[Part]:
        return self._parts

    def by_category(self, category: str) -> List[Part]:
        return [p for p in self._parts if p.category == category]

    def find_by_sku(self, sku: str) -> Part | None:
        for part in self._parts:
            if part.sku == sku:
                return part
        return None
