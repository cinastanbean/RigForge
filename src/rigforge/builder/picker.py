"""配件选择模块"""

from __future__ import annotations

from typing import List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import Part, BuildPlan, UserRequirements


def pick_build_from_candidates(
    requirements: "UserRequirements",
    search_parts_fn: Callable,
    budget_allocation: Optional[dict] = None,
) -> "BuildPlan":
    """从候选配件中选择并组装配置方案
    
    Args:
        requirements: 用户需求
        search_parts_fn: 配件搜索函数
        budget_allocation: 预算分配（可选，默认自动计算）
        
    Returns:
        BuildPlan 实例
    """
    from ..schemas import BuildPlan, Part
    from .budget import allocate_budget
    
    # 计算预算分配
    if budget_allocation is None:
        allocation = allocate_budget(
            requirements.budget_max,
            requirements.use_case
        )
        budgets = allocation.to_dict()
    else:
        budgets = budget_allocation
    
    build = BuildPlan()
    
    def _norm(value: str) -> str:
        return value.strip().lower()
    
    def _brand_matches(item: "Part", brand: str) -> bool:
        brand_norm = _norm(brand)
        if not brand_norm:
            return False
        return _norm(item.brand) == brand_norm or f" {brand_norm} " in f" {_norm(item.name)} "
    
    def choose(
        category: str,
        predicate: Optional[Callable[["Part"], bool]] = None,
        strict_predicate: bool = False
    ) -> Optional["Part"]:
        """选择指定类别的配件"""
        # 构建搜索参数
        prefer_brands = list(requirements.prefer_brands)
        exclude_brands = list(requirements.brand_blacklist)
        
        # CPU 特殊处理
        cpu_preference = (requirements.cpu_preference or "").strip()
        if category == "cpu" and cpu_preference:
            prefer_brands = [cpu_preference]
            exclude_brands = [
                b for b in exclude_brands 
                if _norm(b) != _norm(cpu_preference)
            ]
        
        # 搜索配件
        budget = max(budgets.get(category, 500), 200)
        raw = search_parts_fn.invoke({
            "category": category,
            "budget_max": budget,
            "prefer_brands": prefer_brands,
            "exclude_brands": exclude_brands,
        })
        
        if not raw:
            # 扩大预算重试
            raw = search_parts_fn.invoke({
                "category": category,
                "budget_max": max(requirements.budget_max, 200),
                "prefer_brands": prefer_brands,
                "exclude_brands": exclude_brands,
            })
        
        if not raw:
            return None
        
        items = [Part.model_validate(item) for item in raw]
        
        # CPU 品牌偏好过滤
        if category == "cpu" and cpu_preference:
            items = [p for p in items if _brand_matches(p, cpu_preference)]
            if not items:
                return None
        
        # 排序
        if requirements.priority == "budget":
            items.sort(key=lambda p: p.price)
        elif requirements.priority == "performance":
            items.sort(key=lambda p: (-p.score, p.price))
        
        # 应用谓词过滤
        if predicate:
            for item in items:
                if predicate(item):
                    return item
            if strict_predicate:
                return None
            return items[0] if items else None
        
        return items[0] if items else None
    
    # 按依赖顺序选择配件
    build.cpu = choose("cpu")
    
    build.motherboard = choose(
        "motherboard",
        predicate=lambda p: build.cpu is not None and p.socket == build.cpu.socket
    )
    
    build.memory = choose(
        "memory",
        predicate=lambda p: build.motherboard is not None and p.memory_type == build.motherboard.memory_type,
        strict_predicate=True
    )
    
    build.storage = choose("storage")
    build.gpu = choose("gpu")
    
    # 计算电源需求
    estimated_draw = 0
    if build.cpu:
        estimated_draw += build.cpu.watt
    if build.gpu:
        estimated_draw += build.gpu.watt
    target_psu = int((estimated_draw + 120) * 1.35)
    
    build.psu = choose("psu", predicate=lambda p: p.watt >= target_psu)
    
    build.case = choose(
        "case",
        predicate=lambda p: (
            build.motherboard is not None
            and not (build.motherboard.form_factor == "ATX" and p.form_factor == "mATX")
            and (build.gpu is None or p.length_mm >= build.gpu.length_mm)
        )
    )
    
    build.cooler = choose(
        "cooler",
        predicate=lambda p: build.case is None or p.height_mm <= build.case.height_mm
    )
    
    return build
