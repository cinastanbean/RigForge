from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Set

from .csv_runtime_db import rebuild_runtime_db
from .schemas import Part


class PartsRepository:
    """
    零件仓库类 - Parts Repository Class
    
    负责管理硬件配件数据，支持从 JSON 文件或 SQLite 数据库加载配件信息。
    Manages hardware parts data, supporting loading from JSON files or SQLite database.
    """
    
    def __init__(self, data_path: Path):
        """
        初始化配件仓库 - Initialize parts repository
        
        参数 Parameters:
            data_path: 配件数据文件路径，可以是 JSON 文件或 SQLite 数据库路径
                       Path to parts data file, can be JSON file or SQLite database path
        """
        self.data_path = data_path
        self._parts: List[Part] = []
        self.reload()

    def reload(self) -> None:
        """
        重新加载配件数据 - Reload parts data
        
        从数据源重新加载配件信息。如果数据源是 JSON 文件，直接加载。
        如果数据源是 SQLite 数据库，从数据库加载所有配件。
        Reload parts data from data source. If data source is JSON file, load directly.
        If data source is SQLite database, load all parts from database.
        """
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
        """
        注入遗留测试固件 - Inject legacy test fixtures
        
        为了向后兼容性和测试目的，注入一些测试用的配件数据。
        For backward compatibility and testing purposes, inject some test parts data.
        """
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
        """
        获取所有配件 - Get all parts
        
        返回 Returns:
            所有配件列表
            List of all parts
        """
        return self._parts

    def by_category(self, category: str) -> List[Part]:
        """
        按类别获取配件 - Get parts by category
        
        参数 Parameters:
            category: 配件类别，如 "cpu", "gpu", "memory" 等
                      Part category, such as "cpu", "gpu", "memory", etc.
        
        返回 Returns:
            指定类别的配件列表
            List of parts in the specified category
        """
        return [p for p in self._parts if p.category == category]

    def find_by_sku(self, sku: str) -> Part | None:
        """
        按 SKU 查找配件 - Find part by SKU
        
        参数 Parameters:
            sku: 配件 SKU 编号
                 Part SKU number
        
        返回 Returns:
            找到的配件，如果不存在则返回 None
            Found part, or None if not found
        """
        for part in self._parts:
            if part.sku == sku:
                return part
        return None


class SQLitePartsRepository:
    """
    SQLite 配件仓库类 - SQLite Parts Repository Class
    
    负责从 SQLite 数据库加载和管理硬件配件数据。
    Manages hardware parts data from SQLite database.
    """
    
    def __init__(self, db_path: Path, source_sites: Set[str] | None = None):
        """
        初始化 SQLite 配件仓库 - Initialize SQLite parts repository
        
        参数 Parameters:
            db_path: SQLite 数据库文件路径
                    SQLite database file path
            source_sites: 数据源站点集合，如 {"jd", "newegg"}，如果为 None 则加载所有站点
                          Set of data source sites, such as {"jd", "newegg"}, if None then load all sites
        """
        self.db_path = db_path
        self.source_sites = {s.strip().lower() for s in (source_sites or set()) if s.strip()}
        self._parts: List[Part] = []
        self.reload()

    def reload(self) -> None:
        """
        重新加载配件数据 - Reload parts data
        
        从 SQLite 数据库重新加载配件信息。
        如果指定了 source_sites，只加载指定站点的配件。
        Reload parts data from SQLite database.
        If source_sites is specified, only load parts from specified sites.
        """
        if not self.db_path.exists():
            self._parts = []
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if self.source_sites:
                # 只加载指定站点的配件 - Only load parts from specified sites
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
                # 加载所有站点的配件 - Load parts from all sites
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
        """
        获取所有配件 - Get all parts
        
        返回 Returns:
            所有配件列表
            List of all parts
        """
        return self._parts

    def by_category(self, category: str) -> List[Part]:
        """
        按类别获取配件 - Get parts by category
        
        参数 Parameters:
            category: 配件类别，如 "cpu", "gpu", "memory" 等
                      Part category, such as "cpu", "gpu", "memory", etc.
        
        返回 Returns:
            指定类别的配件列表
            List of parts in the specified category
        """
        return [p for p in self._parts if p.category == category]

    def find_by_sku(self, sku: str) -> Part | None:
        """
        按 SKU 查找配件 - Find part by SKU
        
        参数 Parameters:
            sku: 配件 SKU 编号
                 Part SKU number
        
        返回 Returns:
            找到的配件，如果不存在则返回 None
            Found part, or None if not found
        """
        for part in self._parts:
            if part.sku == sku:
                return part
        return None
