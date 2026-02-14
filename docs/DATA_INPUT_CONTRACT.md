# 数据输入契约（Agent 运行时）


## 1. 目标

定义 RigForge（锐格锻造坊）Agent 的运行时数据输入契约，保证推荐、校验与估算模块字段一致。

## 2. 输入文件

1. `data/data_jd.csv`
2. `data/data_newegg.csv`

说明：服务启动时读取两份 CSV 并重建 `data/agent_parts.db`。

## 3. 字段约定（统一）

1. `sku`
2. `name`
3. `category`
4. `brand`
5. `price`
6. `price_usd`
7. `socket`
8. `score`
9. `vram`
10. `length_mm`
11. `height_mm`
12. `watt`
13. `memory_type`
14. `form_factor`
15. `capacity_gb`
16. `efficiency`
17. `source`
18. `data_version`

## 4. 类目范围

1. `cpu`
2. `cooler`
3. `motherboard`
4. `memory`
5. `storage`
6. `gpu`
7. `psu`
8. `case`
9. `monitor`

## 5. 约束

1. `price` 必须为非负整数。
2. `category` 必须落在类目范围内。
3. `sku` 在单源内唯一。
4. 缺失字段允许为空，但必须保留列头。

## 6. 启动检查

1. 两份 CSV 必须存在。
2. 任一 CSV 不存在时，服务启动失败并返回缺失文件清单。
