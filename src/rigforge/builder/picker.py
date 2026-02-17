"""
配件选择模块 - Part Selection Module

从候选配件中智能选择并组装配置方案。
Intelligently select and assemble build plan from candidate parts.
"""

from __future__ import annotations

from typing import List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import Part, BuildPlan, UserRequirements


def pick_build_from_candidates(
    requirements: "UserRequirements",
    search_parts_fn: Callable,
    budget_allocation: Optional[dict] = None,
) -> "BuildPlan":
    """
    从候选配件中选择并组装配置方案 - Pick Build Plan from Candidate Parts
    
    根据用户需求和候选配件，智能选择最佳的硬件配置方案。
    Intelligently select the best hardware configuration plan based on user requirements and candidate parts.
    
    选择策略 Selection Strategy:
    1. 根据总预算分配各类配件的预算
    2. 依次选择各类配件，考虑品牌偏好、性能优先级和兼容性
    3. 确保配件之间的兼容性（插槽、尺寸、功耗等）
    4. 返回完整的配置方案
    
    参数 Parameters:
        requirements: 用户需求对象，包含预算、用途、品牌偏好等
                     User requirements object, including budget, use case, brand preferences, etc.
        search_parts_fn: 配件搜索函数，用于获取候选配件
                         Function for searching parts, used to get candidate parts
        budget_allocation: 预算分配字典，如果为 None 则自动计算
                          Budget allocation dictionary, if None then calculate automatically
    
    返回 Returns:
        完整的配置方案
        Complete build plan
    """
    from ..schemas import BuildPlan, Part
    from .budget import allocate_budget
    
    # 步骤 1: 计算预算分配 - Step 1: Calculate budget allocation
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
        """
        标准化字符串 - Normalize String
        
        将字符串转换为小写并去除首尾空格。
        Convert string to lowercase and trim leading/trailing spaces.
        """
        return value.strip().lower()
    
    def _brand_matches(item: "Part", brand: str) -> bool:
        """
        检查品牌是否匹配 - Check if Brand Matches
        
        检查配件的品牌或名称中是否包含指定品牌。
        Check if part's brand or name contains the specified brand.
        """
        brand_norm = _norm(brand)
        if not brand_norm:
            return False
        return _norm(item.brand) == brand_norm or f" {brand_norm} " in f" {_norm(item.name)} "
    
    def choose(
        category: str,
        predicate: Optional[Callable[["Part"], bool]] = None,
        strict_predicate: bool = False
    ) -> Optional["Part"]:
        """
        选择指定类别的配件 - Choose Part by Category
        
        根据类别、预算和约束条件选择最佳配件。
        Select the best part based on category, budget, and constraints.
        
        参数 Parameters:
            category: 配件类别
                      Part category
            predicate: 可选的谓词函数，用于进一步过滤配件
                      Optional predicate function for further filtering parts
            strict_predicate: 是否严格应用谓词，如果为 True 且没有匹配项则返回 None
                             Whether to strictly apply predicate, if True and no match then return None
        
        返回 Returns:
            选中的配件，如果没有找到则返回 None
            Selected part, or None if not found
        """
        # 构建搜索参数 - Build search parameters
        prefer_brands = list(requirements.prefer_brands)
        exclude_brands = list(requirements.brand_blacklist)
        
        # CPU 特殊处理 - CPU special handling
        cpu_preference = (requirements.cpu_preference or "").strip()
        if category == "cpu" and cpu_preference:
            # 显式的 CPU 偏好应该比通用品牌历史更强
            # Explicit CPU preference should be stronger than generic brand history.
            prefer_brands = [cpu_preference]
            exclude_brands = [
                b for b in exclude_brands 
                if _norm(b) != _norm(cpu_preference)
            ]
        
        # 搜索配件 - Search parts
        budget = max(budgets.get(category, 500), 200)
        raw = search_parts_fn.invoke({
            "category": category,
            "budget_max": budget,
            "prefer_brands": prefer_brands,
            "exclude_brands": exclude_brands,
        })
        
        if not raw:
            # 扩大预算重试 - Expand budget and retry
            raw = search_parts_fn.invoke({
                "category": category,
                "budget_max": max(requirements.budget_max, 200),
                "prefer_brands": prefer_brands,
                "exclude_brands": exclude_brands,
            })
        
        if not raw:
            return None
        
        items = [Part.model_validate(item) for item in raw]
        
        # CPU 品牌偏好过滤 - CPU brand preference filtering
        if category == "cpu" and cpu_preference:
            items = [p for p in items if _brand_matches(p, cpu_preference)]
            if not items:
                return None
        
        # 排序 - Sort
        if requirements.priority == "budget":
            items.sort(key=lambda p: p.price)
        elif requirements.priority == "performance":
            items.sort(key=lambda p: (-p.score, p.price))
        
        # 应用谓词过滤 - Apply predicate filtering
        if predicate:
            for item in items:
                if predicate(item):
                    return item
            if strict_predicate:
                return None
            return items[0] if items else None
        
        return items[0] if items else None
    
    # 步骤 2: 按依赖顺序选择配件 - Step 2: Choose parts in dependency order
    
    # 选择 CPU - Choose CPU
    build.cpu = choose("cpu")
    
    # 选择主板（必须与 CPU 插槽匹配）- Choose motherboard (must match CPU socket)
    build.motherboard = choose(
        "motherboard",
        predicate=lambda p: build.cpu is not None and p.socket == build.cpu.socket
    )
    
    # 选择内存（必须与主板内存类型匹配，严格匹配）- Choose memory (must match motherboard memory type, strict match)
    build.memory = choose(
        "memory",
        predicate=lambda p: build.motherboard is not None and p.memory_type == build.motherboard.memory_type,
        strict_predicate=True
    )
    
    # 选择存储 - Choose storage
    build.storage = choose("storage")
    
    # 选择显卡 - Choose GPU
    build.gpu = choose("gpu")
    
    # 计算电源需求 - Calculate PSU requirements
    estimated_draw = 0
    if build.cpu:
        estimated_draw += build.cpu.watt
    if build.gpu:
        estimated_draw += build.gpu.watt
    target_psu = int((estimated_draw + 120) * 1.35)
    
    # 选择电源（功率必须足够）- Choose PSU (wattage must be sufficient)
    build.psu = choose("psu", predicate=lambda p: p.watt >= target_psu)
    
    # 选择机箱（必须与主板规格匹配，且能容纳显卡）- Choose case (must match motherboard form factor and fit GPU)
    build.case = choose(
        "case",
        predicate=lambda p: (
            build.motherboard is not None
            and not (build.motherboard.form_factor == "ATX" and p.form_factor == "mATX")
            and (build.gpu is None or p.length_mm >= build.gpu.length_mm)
        )
    )
    
    # 选择散热器（高度必须适合机箱）- Choose cooler (height must fit case)
    build.cooler = choose(
        "cooler",
        predicate=lambda p: build.case is None or p.height_mm <= build.case.height_mm
    )
    
    return build
