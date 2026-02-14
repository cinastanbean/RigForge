from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable


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


VALID_CATEGORIES = {"cpu", "motherboard", "memory", "storage", "gpu", "psu", "case", "cooler"}


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

FLOAT_FIELDS = {
    "price_usd",
}


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
    try:
        return int(float(v or 0))
    except Exception:
        return 0


def _to_float(v: str) -> float:
    try:
        return float(v or 0)
    except Exception:
        return 0.0


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("category") or "").strip().lower()
            if category not in VALID_CATEGORIES:
                continue
            sku = (row.get("sku") or "").strip()
            name = (row.get("name") or "").strip()
            brand = (row.get("brand") or "").strip() or "Unknown"
            if not sku or not name:
                continue

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
    db_path.parent.mkdir(parents=True, exist_ok=True)
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS agent_parts")
        conn.execute(RUNTIME_TABLE_SQL)

        for src in (jd_csv, newegg_csv):
            for row in _iter_csv_rows(src):
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
                    ON CONFLICT(sku) DO UPDATE SET
                      name=excluded.name,
                      category=excluded.category,
                      brand=excluded.brand,
                      price=excluded.price,
                      price_usd=excluded.price_usd,
                      socket=excluded.socket,
                      tdp=excluded.tdp,
                      score=excluded.score,
                      vram=excluded.vram,
                      length_mm=excluded.length_mm,
                      height_mm=excluded.height_mm,
                      watt=excluded.watt,
                      memory_type=excluded.memory_type,
                      form_factor=excluded.form_factor,
                      capacity_gb=excluded.capacity_gb,
                      efficiency=excluded.efficiency,
                      source_site=excluded.source_site,
                      source_url=excluded.source_url,
                      item_sku=excluded.item_sku,
                      cpu_socket=excluded.cpu_socket,
                      mb_socket=excluded.mb_socket,
                      mb_form_factor=excluded.mb_form_factor,
                      gpu_length_mm=excluded.gpu_length_mm,
                      case_max_gpu_length_mm=excluded.case_max_gpu_length_mm,
                      case_max_cpu_cooler_height_mm=excluded.case_max_cpu_cooler_height_mm,
                      monitor_resolution=excluded.monitor_resolution,
                      monitor_refresh_hz=excluded.monitor_refresh_hz,
                      storage_interface=excluded.storage_interface,
                      storage_form_factor=excluded.storage_form_factor,
                      pcie_version=excluded.pcie_version,
                      specs_json=excluded.specs_json,
                      updated_at=CURRENT_TIMESTAMP
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

    return {
        "db": str(db_path),
        "rows_processed": inserted,
        "rows_total": int(total),
        "jd_csv": str(jd_csv),
        "newegg_csv": str(newegg_csv),
    }
