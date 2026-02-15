# 数据输入契约（Agent 运行时）

## 1. 目标

定义 RigForge（锐格锻造坊）Agent 的运行时数据输入契约，保证推荐、校验与估算模块字段一致。

## 2. 输入文件

1. `data/data_jd.csv` - 京东数据源
2. `data/data_newegg.csv` - Newegg 数据源

说明：服务启动时读取两份 CSV 并重建 `data/agent_parts.db`（SQLite 运行时数据库）。

## 3. 字段约定（32 字段）

### 核心字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `sku` | TEXT | 主键，唯一标识 |
| `name` | TEXT | 产品名称 |
| `category` | TEXT | 类目（见第4节） |
| `brand` | TEXT | 品牌 |
| `price` | INTEGER | 价格（人民币） |
| `price_usd` | REAL | 美元价格 |

### 兼容性字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `socket` | TEXT | CPU/主板插槽 |
| `cpu_socket` | TEXT | CPU 插槽（备用） |
| `mb_socket` | TEXT | 主板插槽（备用） |
| `tdp` | INTEGER | 热设计功耗 |

### 性能/规格字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | INTEGER | 性能评分 |
| `vram` | INTEGER | 显存容量（GB） |
| `length_mm` | INTEGER | GPU 长度 |
| `height_mm` | INTEGER | 散热器高度 |
| `watt` | INTEGER | 功耗（W） |

### 内存/存储字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `memory_type` | TEXT | DDR4/DDR5 |
| `form_factor` | TEXT | 板型（ATX/mATX） |
| `capacity_gb` | INTEGER | 容量（GB） |

### 机箱/电源字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `efficiency` | TEXT | 电源效率（80+ Gold 等） |
| `gpu_length_mm` | INTEGER | GPU 长度（机箱兼容） |
| `case_max_gpu_length_mm` | INTEGER | 机箱最大 GPU 长度 |
| `case_max_cpu_cooler_height_mm` | INTEGER | 机箱最大散热器高度 |

### 数据来源字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `source_site` | TEXT | 来源站点（jd/newegg） |
| `source_url` | TEXT | 原始链接 |
| `item_sku` | TEXT | 原始 SKU |

### 扩展字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `monitor_resolution` | TEXT | 显示器分辨率 |
| `monitor_refresh_hz` | TEXT | 刷新率 |
| `storage_interface` | TEXT | 存储接口 |
| `storage_form_factor` | TEXT | 存储规格 |
| `pcie_version` | TEXT | PCIe 版本 |
| `specs_json` | TEXT | JSON 格式详细规格 |

## 4. 类目范围

| 类目 | 说明 |
|------|------|
| `cpu` | 处理器 |
| `cooler` | 散热器 |
| `motherboard` | 主板 |
| `memory` | 内存 |
| `storage` | 存储 |
| `gpu` | 显卡 |
| `psu` | 电源 |
| `case` | 机箱 |

## 5. 约束

1. `price` 必须为非负整数
2. `category` 必须落在类目范围内
3. `sku` 全局唯一
4. 缺失字段允许为空，但必须保留列头
5. `specs_json` 中的逗号需要正确转义

## 6. 启动检查

1. 两份 CSV 必须存在
2. 任一 CSV 不存在时，服务启动失败
3. 启动时自动重建 `agent_parts.db`

## 7. 数据质量

当前数据统计（2026-02）：
- `data_jd.csv`: 990 条记录
- `data_newegg.csv`: 1091 条记录
- 已知问题：存在同一产品多个 SKU 的重复情况（JD 44 组，Newegg 147 组）
