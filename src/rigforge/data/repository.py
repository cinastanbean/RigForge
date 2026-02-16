"""数据仓库抽象"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Part


class PartsRepository:
    """配件仓库基类"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._parts: List[Part] = []
        self.reload()
    
    def reload(self) -> None:
        """重新加载数据"""
        raise NotImplementedError
    
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
    """SQLite 数据仓库"""
    
    def __init__(self, db_path: Path, source_sites: Set[str] | None = None):
        self.db_path = db_path
        self.source_sites = {s.strip().lower() for s in (source_sites or set()) if s.strip()}
        self._parts: List[Part] = []
        self.reload()
    
    def reload(self) -> None:
        from .models import Part
        
        if not self.db_path.exists():
            self._parts = []
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if self.source_sites:
                placeholders = ",".join("?" for _ in self.source_sites)
                rows = conn.execute(
                    f"""
                    SELECT sku, name, category, brand, price, socket, tdp, score, vram,
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
                    SELECT sku, name, category, brand, price, socket, tdp, score, vram,
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
