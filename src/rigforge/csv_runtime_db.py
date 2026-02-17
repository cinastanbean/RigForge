"""
CSV 运行时数据库构建器 - CSV Runtime Database Builder

从 CSV 文件构建 SQLite 运行时数据库，过滤服务器级产品并去重。
Build SQLite runtime database from CSV files, filter server-grade products and deduplicate.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable


# 运行时数据库表结构 SQL - Runtime Database Table Schema SQL
RUNTIME_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_parts (
  sku TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT NOT NULL,
  brand TEXT NOT NULL,
  price INTEGER NOT NULL,
  price_usd REAL,
  socket TEXT,
  tdp INTEGER,
  score INTEGER,
  vram INTEGER,
  length_mm INTEGER,
  height_mm INTEGER,
  watt INTEGER,
  memory_type TEXT,
  form_factor TEXT,
  capacity_gb INTEGER,
  efficiency TEXT,
  source_site TEXT,
  source_url TEXT,
  item_sku TEXT,
  cpu_socket TEXT,
  mb_socket TEXT,
  mb_form_factor TEXT,
  gpu_length_mm INTEGER,
  case_max_gpu_length_mm INTEGER,
  case_max_cpu_cooler_height_mm INTEGER,
  monitor_resolution TEXT,
  monitor_refresh_hz TEXT,
  storage_interface TEXT,
  storage_form_factor TEXT,
  pcie_version TEXT,
  specs_json TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

# 有效的配件类别 - Valid Part Categories
VALID_CATEGORIES = {"cpu", "motherboard", "memory", "storage", "gpu", "psu", "case", "cooler"}

# 服务器级配件关键词（需过滤）- Server-grade Part Keywords (to filter)
SERVER_KEYWORDS = {
    # 服务器 CPU - Server CPUs
    "xeon", "epyc", "threadripper", "opteron", "xeon phi",
    # 服务器主板 - Server motherboards
    "server board", "workstation", "rackmount",
    # ECC 内存标识 - ECC memory indicators
    "ecc", "registered", "rdimm", "lrdimm",
    # 企业级存储 - Enterprise storage
    "enterprise", "data center", "nas drive", "surveillance",
}


def _is_server_product(name: str, category: str) -> bool:
    """
    检测是否为服务器级产品，应过滤掉 - Detect if it's a server-grade product, should be filtered
    
    参数 Parameters:
        name: 产品名称
              Product name
        category: 产品类别
                  Product category
    
    返回 Returns:
        是否为服务器级产品
        Whether it's a server-grade product
    """
    name_lower = name.lower()
    for keyword in SERVER_KEYWORDS:
        if keyword in name_lower:
            return True
    
    # 特殊处理：高端消费级产品不应误杀
    # Special handling: high-end consumer products should not be incorrectly filtered
    # Threadripper PRO 是工作站级，普通 Threadripper 是 HEDT 消费级
    # Threadripper PRO is workstation-grade, regular Threadripper is HEDT consumer-grade
    if "threadripper pro" in name_lower:
        return True
    if "threadripper" in name_lower and "pro" not in name_lower:
        return False  # 普通 Threadripper 保留 - Keep regular Threadripper
    
    return False


# 整数字段 - Integer Fields
INT_FIELDS = {
    "price",
    "tdp",
    "score",
    "vram",
    "length_mm",
    "height_mm",
    "watt",
    "capacity_gb",
    "gpu_length_mm",
    "case_max_gpu_length_mm",
    "case_max_cpu_cooler_height_mm",
}

# 浮点数字段 - Float Fields
FLOAT_FIELDS = {
    "price_usd",
}

# 所有字段 - All Fields
ALL_FIELDS = [
    "sku",
    "name",
    "category",
    "brand",
    "price",
    "price_usd",
    "socket",
    "tdp",
    "score",
    "vram",
    "length_mm",
    "height_mm",
    "watt",
    "memory_type",
    "form_factor",
    "capacity_gb",
    "efficiency",
    "source_site",
    "source_url",
    "item_sku",
    "cpu_socket",
    "mb_socket",
    "mb_form_factor",
    "gpu_length_mm",
    "case_max_gpu_length_mm",
    "case_max_cpu_cooler_height_mm",
    "monitor_resolution",
    "monitor_refresh_hz",
    "storage_interface",
    "storage_form_factor",
    "pcie_version",
    "specs_json",
]


def _to_int(v: str) -> int:
    """
    转换为整数 - Convert to Integer
    
    参数 Parameters:
        v: 字符串值
           String value
    
    返回 Returns:
        转换后的整数值，失败返回 0
        Converted integer value, returns 0 on failure
    """
    try:
        return int(float(v or 0))
    except Exception:
        return 0


def _to_float(v: str) -> float:
    """
    转换为浮点数 - Convert to Float
    
    参数 Parameters:
        v: 字符串值
           String value
    
    返回 Returns:
        转换后的浮点数值，失败返回 0.0
        Converted float value, returns 0.0 on failure
    """
    try:
        return float(v or 0)
    except Exception:
        return 0.0


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, object]]:
    """
    迭代 CSV 行，过滤无效数据 - Iterate CSV rows, filter invalid data
    
    参数 Parameters:
        path: CSV 文件路径
              CSV file path
    
    返回 Returns:
        迭代器，产生过滤后的行字典
        Iterator yielding filtered row dictionaries
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 检查类别是否有效 - Check if category is valid
            category = (row.get("category") or "").strip().lower()
            if category not in VALID_CATEGORIES:
                continue
            
            # 检查必需字段 - Check required fields
            sku = (row.get("sku") or "").strip()
            name = (row.get("name") or "").strip()
            brand = (row.get("brand") or "").strip() or "Unknown"
            if not sku or not name:
                continue
            
            # 过滤服务器级产品 - Filter server-grade products
            if _is_server_product(name, category):
                continue
            
            # 检查价格是否有效 - Check if price is valid
            price_raw = (row.get("price") or "").strip()
            try:
                price = int(float(price_raw or 0))
            except:
                price = 0
            
            # 过滤价格为 0 或无效的产品 - Filter products with price 0 or invalid
            if price <= 0:
                continue

            # 转换字段类型 - Convert field types
            out: Dict[str, object] = {}
            for field in ALL_FIELDS:
                raw = (row.get(field) or "").strip()
                if field in INT_FIELDS:
                    out[field] = _to_int(raw)
                elif field in FLOAT_FIELDS:
                    out[field] = _to_float(raw)
                else:
                    out[field] = raw
            out["category"] = category
            out["brand"] = brand
            out["price"] = max(0, int(out.get("price") or 0))
            out["score"] = max(0, int(out.get("score") or 0))
            if not out.get("socket"):
                out["socket"] = (out.get("cpu_socket") or out.get("mb_socket") or "")
            yield out


def rebuild_runtime_db(db_path: Path, jd_csv: Path, newegg_csv: Path) -> dict:
    """
    重建运行时数据库，过滤服务器级产品并去重
    Rebuild Runtime Database, filter server-grade products and deduplicate
    
    去重策略 Deduplication Strategy:
    - 相同产品（同名同品牌）只保留一条记录
      Keep only one record for same product (same name and brand)
    - 优先保留有 source_url 的记录（真实商品链接）
      Prioritize records with source_url (real product link)
    - 其次保留有 item_sku 的记录
      Next prioritize records with item_sku
    
    参数 Parameters:
        db_path: 数据库文件路径
                 Database file path
        jd_csv: 京东 CSV 文件路径
                JD CSV file path
        newegg_csv: NewEgg CSV 文件路径
                    NewEgg CSV file path
    
    返回 Returns:
        构建统计信息字典
        Build statistics dictionary
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    inserted = 0
    dedup_count = 0
    
    # 用于去重：(name_normalized, brand) -> best_row
    # For deduplication: (name_normalized, brand) -> best_row
    # name_normalized 是名称的标准化形式（去除空格、转小写）
    # name_normalized is normalized form of name (remove spaces, lowercase)
    seen_products: Dict[tuple, Dict[str, object]] = {}
    
    # 先收集所有行，应用过滤和去重
    # First collect all rows, apply filtering and deduplication
    all_rows = []
    for src in (jd_csv, newegg_csv):
        for row in _iter_csv_rows(src):
            all_rows.append(row)
    
    # 去重处理 - Deduplication processing
    for row in all_rows:
        # 生成标准化名称用于比较 - Generate normalized name for comparison
        name_normalized = row.get("name", "").lower().replace(" ", "").replace("-", "")
        brand_normalized = row.get("brand", "").lower().strip()
        key = (name_normalized, brand_normalized)
        
        source_url = row.get("source_url", "") or ""
        item_sku = row.get("item_sku", "") or ""
        
        if key not in seen_products:
            seen_products[key] = row
        else:
            # 检查是否应该替换现有记录 - Check if should replace existing record
            existing = seen_products[key]
            existing_url = existing.get("source_url", "") or ""
            existing_sku = existing.get("item_sku", "") or ""
            
            # 新记录有 URL，旧记录没有 -> 替换
            # New record has URL, old record doesn't -> replace
            if source_url and not existing_url:
                seen_products[key] = row
                dedup_count += 1
            # 新记录有 item_sku，旧记录没有 -> 替换
            # New record has item_sku, old record doesn't -> replace
            elif not source_url and not existing_url and item_sku and not existing_sku:
                seen_products[key] = row
                dedup_count += 1
            # 都没有/都有，保留先来的
            # Both have or both don't have, keep the first one
            else:
                dedup_count += 1
    
    # 写入数据库 - Write to database
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS agent_parts")
        conn.execute(RUNTIME_TABLE_SQL)

        for row in seen_products.values():
            conn.execute(
                """
                INSERT INTO agent_parts (
                  sku, name, category, brand, price, price_usd, socket, tdp, score, vram,
                  length_mm, height_mm, watt, memory_type, form_factor, capacity_gb,
                  efficiency, source_site, source_url, item_sku, cpu_socket, mb_socket,
                  mb_form_factor, gpu_length_mm, case_max_gpu_length_mm,
                  case_max_cpu_cooler_height_mm, monitor_resolution, monitor_refresh_hz,
                  storage_interface, storage_form_factor, pcie_version, specs_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    row["sku"],
                    row["name"],
                    row["category"],
                    row["brand"],
                    row["price"],
                    row["price_usd"],
                    row["socket"],
                    row["tdp"],
                    row["score"],
                    row["vram"],
                    row["length_mm"],
                    row["height_mm"],
                    row["watt"],
                    row["memory_type"],
                    row["form_factor"],
                    row["capacity_gb"],
                    row["efficiency"],
                    row["source_site"],
                    row["source_url"],
                    row["item_sku"],
                    row["cpu_socket"],
                    row["mb_socket"],
                    row["mb_form_factor"],
                    row["gpu_length_mm"],
                    row["case_max_gpu_length_mm"],
                    row["case_max_cpu_cooler_height_mm"],
                    row["monitor_resolution"],
                    row["monitor_refresh_hz"],
                    row["storage_interface"],
                    row["storage_form_factor"],
                    row["pcie_version"],
                    row["specs_json"],
                ),
            )
            inserted += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM agent_parts").fetchone()[0]

    # 返回统计信息 - Return statistics
    return {
        "db": str(db_path),
        "rows_processed": inserted,
        "rows_total": int(total),
        "rows_from_csv": len(all_rows),
        "deduplicated": dedup_count,
        "jd_csv": str(jd_csv),
        "newegg_csv": str(newegg_csv),
    }
