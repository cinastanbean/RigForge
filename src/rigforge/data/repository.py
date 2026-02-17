"""
数据仓库抽象 - Data Repository Abstraction

提供配件数据的访问接口，支持多种数据源。
Provide access interface for parts data, supporting multiple data sources.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Part


class PartsRepository:
    """
    配件仓库基类 - Parts Repository Base Class
    
    定义配件仓库的基本接口。
    Define basic interface for parts repository.
    """
    
    def __init__(self, data_path: Path):
        """
        初始化仓库 - Initialize Repository
        
        参数 Parameters:
            data_path: 数据文件路径
                       Data file path
        """
        self.data_path = data_path
        self._parts: List[Part] = []
        self.reload()
    
    def reload(self) -> None:
        """
        重新加载数据 - Reload Data
        
        从数据源重新加载配件数据。
        Reload parts data from data source.
        """
        raise NotImplementedError
    
    def all_parts(self) -> List[Part]:
        """
        获取所有配件 - Get All Parts
        
        返回 Returns:
            所有配件列表
            List of all parts
        """
        return self._parts
    
    def by_category(self, category: str) -> List[Part]:
        """
        按类别获取配件 - Get Parts by Category
        
        参数 Parameters:
            category: 配件类别
                      Part category
        
        返回 Returns:
            指定类别的配件列表
            List of parts in specified category
        """
        return [p for p in self._parts if p.category == category]
    
    def find_by_sku(self, sku: str) -> Part | None:
        """
        按 SKU 查找配件 - Find Part by SKU
        
        参数 Parameters:
            sku: 配件 SKU
                 Part SKU
        
        返回 Returns:
            找到的配件，未找到则返回 None
            Found part, or None if not found
        """
        for part in self._parts:
            if part.sku == sku:
                return part
        return None


class SQLitePartsRepository:
    """
    SQLite 数据仓库 - SQLite Data Repository
    
    从 SQLite 数据库加载配件数据。
    Load parts data from SQLite database.
    """
    
    def __init__(self, db_path: Path, source_sites: Set[str] | None = None):
        """
        初始化 SQLite 仓库 - Initialize SQLite Repository
        
        参数 Parameters:
            db_path: 数据库文件路径
                     Database file path
            source_sites: 数据源站点集合（可选，为空则加载所有站点）
                          Set of source sites (optional, empty means load all sites)
        """
        self.db_path = db_path
        self.source_sites = {s.strip().lower() for s in (source_sites or set()) if s.strip()}
        self._parts: List[Part] = []
        self.reload()
    
    def reload(self) -> None:
        """
        重新加载数据 - Reload Data
        
        从 SQLite 数据库重新加载配件数据。
        Reload parts data from SQLite database.
        """
        from .models import Part
        
        # 检查数据库文件是否存在 - Check if database file exists
        if not self.db_path.exists():
            self._parts = []
            return
        
        # 连接数据库并查询数据 - Connect to database and query data
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 根据数据源站点筛选 - Filter by source sites
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
        
        # 将查询结果转换为 Part 对象 - Convert query results to Part objects
        self._parts = [Part.model_validate(dict(r)) for r in rows]
    
    def all_parts(self) -> List[Part]:
        """
        获取所有配件 - Get All Parts
        
        返回 Returns:
            所有配件列表
            List of all parts
        """
        return self._parts
    
    def by_category(self, category: str) -> List[Part]:
        """
        按类别获取配件 - Get Parts by Category
        
        参数 Parameters:
            category: 配件类别
                      Part category
        
        返回 Returns:
            指定类别的配件列表
            List of parts in specified category
        """
        return [p for p in self._parts if p.category == category]
    
    def find_by_sku(self, sku: str) -> Part | None:
        """
        按 SKU 查找配件 - Find Part by SKU
        
        参数 Parameters:
            sku: 配件 SKU
                 Part SKU
        
        返回 Returns:
            找到的配件，未找到则返回 None
            Found part, or None if not found
        """
        for part in self._parts:
            if part.sku == sku:
                return part
        return None
