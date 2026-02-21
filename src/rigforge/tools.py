from __future__ import annotations

from typing import List, Protocol

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .schemas import BuildPlan, Part, UserRequirements


class PartsRepoProtocol(Protocol):
    """
    配件仓库协议 - Parts Repository Protocol
    
    定义配件仓库的接口规范，确保不同的配件仓库实现具有统一的方法签名。
    Defines the interface specification for parts repository, ensuring different implementations have consistent method signatures.
    """
    def all_parts(self) -> List[Part]: ...
    def by_category(self, category: str) -> List[Part]: ...
    def find_by_sku(self, sku: str) -> Part | None: ...


class SearchPartsInput(BaseModel):
    """
    搜索配件输入模型 - Search Parts Input Model
    
    定义搜索配件时需要的输入参数。
    Defines input parameters required for searching parts.
    """
    category: str = Field(description="Part category such as cpu, gpu, motherboard")
    budget_max: int = Field(description="Max acceptable price for this category")
    prefer_brands: List[str] = Field(default_factory=list)
    exclude_brands: List[str] = Field(default_factory=list)


class CompatibilityInput(BaseModel):
    """
    兼容性检查输入模型 - Compatibility Check Input Model
    
    定义检查硬件兼容性时需要的输入参数，包含所有核心硬件的 SKU。
    Defines input parameters required for checking hardware compatibility, including SKUs of all core hardware.
    """
    cpu_sku: str
    motherboard_sku: str
    memory_sku: str
    gpu_sku: str
    psu_sku: str
    case_sku: str
    cooler_sku: str


class RecommendationContextInput(BaseModel):
    """
    推荐上下文输入模型 - Recommendation Context Input Model
    
    定义生成推荐上下文时需要的输入参数，包括预算、用途和分辨率。
    Defines input parameters required for generating recommendation context, including budget, use case, and resolution.
    """
    budget_min: int
    budget_max: int
    use_case: List[str] = Field(default_factory=list)
    resolution: str = "1080p"


class Toolset:
    """
    工具集类 - Toolset Class
    
    管理所有推荐系统相关的工具函数，包括配件搜索、功耗估算、兼容性检查和推荐上下文生成。
    Manages all recommendation system related tool functions, including parts search, power estimation, compatibility check, and recommendation context generation.
    
    工具列表 Tool List:
    - search_parts: 按类别和预算搜索配件
    - estimate_power: 估算系统功耗
    - check_compatibility: 检查硬件兼容性
    - recommendation_context: 生成推荐上下文（权重配置）
    """
    
    def __init__(
        self,
        repo: PartsRepoProtocol,
        *,
        build_data_source: str = "csv(jd+newegg)",
        build_data_version: str = "v0",
        build_data_mode: str = "jd_newegg",
    ):
        """
        初始化工具集 - Initialize toolset
        
        参数 Parameters:
            repo: 配件仓库实例，用于查询配件数据
                  Parts repository instance for querying parts data
            build_data_source: 配件数据源，默认为 "csv(jd+newegg)"
                               Parts data source, default is "csv(jd+newegg)"
            build_data_version: 配件数据版本，默认为 "v0"
                               Parts data version, default is "v0"
            build_data_mode: 配件数据模式，默认为 "jd_newegg"
                            Parts data mode, default is "jd_newegg"
        """
        self.repo = repo
        self.build_data_source = build_data_source
        self.build_data_version = build_data_version
        self.build_data_mode = build_data_mode

    def register(self):
        """
        注册所有工具函数 - Register all tool functions
        
        返回 Returns:
            工具字典，包含所有已注册的工具函数
            Dictionary of tools, containing all registered tool functions
        """
        repo = self.repo

        @tool("search_parts", args_schema=SearchPartsInput)
        def search_parts(
            category: str,
            budget_max: int,
            prefer_brands: List[str] | None = None,
            exclude_brands: List[str] | None = None,
        ) -> List[dict]:
            """
            搜索配件 - Search Parts
            
            按类别和预算搜索配件，支持品牌偏好和排除品牌，按性能评分排序。
            Search parts by category and budget, supporting brand preferences and exclusions, sorted by performance score.
            
            处理流程 Processing Flow:
            1. 从仓库获取指定类别的所有配件
            2. 过滤出价格在预算范围内的配件
            3. 排除黑名单中的品牌
            4. 如果有品牌偏好，优先显示偏好品牌
            5. 按性能评分和价格排序
            6. 返回前 5 个最佳匹配
            
            参数 Parameters:
                category: 配件类别，如 "cpu", "gpu", "motherboard" 等
                          Part category, such as "cpu", "gpu", "motherboard", etc.
                budget_max: 该类别的最大预算
                           Max budget for this category
                prefer_brands: 偏好品牌列表，优先选择这些品牌
                              List of preferred brands, prioritize these brands
                exclude_brands: 排除品牌列表，不选择这些品牌
                               List of excluded brands, do not select these brands
            
            返回 Returns:
                配件字典列表，每个字典包含配件的详细信息
                List of part dictionaries, each containing detailed part information
            """
            prefer_brands = prefer_brands or []
            exclude_brands = exclude_brands or []

            def _norm(value: str) -> str:
                return value.strip().lower()

            # 品牌中英文映射 - Brand Chinese-English mapping
            BRAND_ALIASES = {
                "intel": {"intel", "英特尔"},
                "amd": {"amd", "锐龙", "超威"},
                "nvidia": {"nvidia", "英伟达"},
            }

            def _brand_matches(part: Part, brand_norm: str) -> bool:
                if not brand_norm:
                    return False
                part_brand = _norm(part.brand)
                part_name = _norm(part.name)

                # 直接匹配 - Direct match
                if part_brand == brand_norm:
                    return True

                # 名称中包含品牌 - Brand in name
                if f" {brand_norm} " in f" {part_name} ":
                    return True

                # 通过别名映射匹配 - Match via brand aliases
                for canonical, aliases in BRAND_ALIASES.items():
                    if brand_norm in aliases or brand_norm == canonical:
                        if part_brand in aliases:
                            return True
                        # 检查名称中是否包含任一别名
                        for alias in aliases:
                            if f" {alias} " in f" {part_name} ":
                                return True

                return False

            prefer_brands_norm = {_norm(b) for b in prefer_brands if b and b.strip()}
            exclude_brands_norm = {_norm(b) for b in exclude_brands if b and b.strip()}

            # 步骤 1: 从仓库获取指定类别的所有配件 - Step 1: Get all parts in category from repository
            all_parts = repo.by_category(category)

            # 步骤 2-3: 过滤价格和品牌 - Step 2-3: Filter by price and brand
            candidates = [
                p
                for p in all_parts
                if p.price <= budget_max
                and not any(_brand_matches(p, blocked) for blocked in exclude_brands_norm)
            ]

            # 步骤 4-5: 过滤和排序 - Step 4-5: Filter and Sort
            if prefer_brands_norm:
                # 只保留偏好品牌的配件 - Only keep preferred brand parts
                candidates = [
                    p for p in candidates
                    if any(_brand_matches(p, preferred) for preferred in prefer_brands_norm)
                ]
                # 按性能评分和价格排序 - Sort by performance score and price
                candidates.sort(key=lambda p: (-p.score, p.price))
            else:
                # 按性能评分和价格排序 - Sort by performance score and price
                candidates.sort(key=lambda p: (-p.score, p.price))

            # 步骤 6: 返回前 20 个 - Step 6: Return top 20
            return [c.model_dump() for c in candidates[:20]]

        @tool("estimate_power")
        def estimate_power(parts: List[str]) -> int:
            """
            估算系统功耗 - Estimate System Power
            
            通过累加所有配件的功耗并添加余量来估算系统总功耗。
            Estimate total system power draw by summing all parts' wattage and adding headroom.
            
            处理流程 Processing Flow:
            1. 累加所有配件的功耗
            2. 乘以 1.35 作为余量（考虑峰值功耗和效率损耗）
            3. 返回估算的总功耗
            
            参数 Parameters:
                parts: 配件 SKU 列表
                       List of part SKUs
            
            返回 Returns:
                估算的系统总功耗（瓦特）
                Estimated total system power draw (watts)
            """
            watts = 0
            for sku in parts:
                part = repo.find_by_sku(sku)
                if part:
                    watts += part.watt
            return int(watts * 1.35)

        @tool("check_compatibility", args_schema=CompatibilityInput)
        def check_compatibility(
            cpu_sku: str,
            motherboard_sku: str,
            memory_sku: str,
            gpu_sku: str,
            psu_sku: str,
            case_sku: str,
            cooler_sku: str,
        ) -> List[str]:
            """
            检查硬件兼容性 - Check Hardware Compatibility
            
            验证核心硬件之间的兼容性约束，包括接口、尺寸、功耗等。
            Validate core hardware compatibility constraints, including interfaces, dimensions, power, etc.
            
            检查项 Check Items:
            1. CPU 和主板插槽匹配
            2. 内存类型和主板匹配
            3. 显卡长度和机箱匹配
            4. 散热器高度和机箱匹配
            5. 电源功率是否足够
            6. 主板和机箱规格匹配
            
            参数 Parameters:
                cpu_sku: CPU SKU
                motherboard_sku: 主板 SKU
                memory_sku: 内存 SKU
                gpu_sku: 显卡 SKU
                psu_sku: 电源 SKU
                case_sku: 机箱 SKU
                cooler_sku: 散热器 SKU
            
            返回 Returns:
                兼容性问题列表，如果为空则表示兼容
                List of compatibility issues, empty if compatible
            """
            issues: List[str] = []
            cpu = repo.find_by_sku(cpu_sku)
            motherboard = repo.find_by_sku(motherboard_sku)
            memory = repo.find_by_sku(memory_sku)
            gpu = repo.find_by_sku(gpu_sku)
            psu = repo.find_by_sku(psu_sku)
            case = repo.find_by_sku(case_sku)
            cooler = repo.find_by_sku(cooler_sku)

            if not all([cpu, motherboard, memory, gpu, psu, case, cooler]):
                return ["build parts are incomplete"]

            # 检查 1: CPU 和主板插槽匹配 - Check 1: CPU and motherboard socket match
            if cpu.socket != motherboard.socket:
                issues.append("CPU and motherboard socket mismatch")

            # 检查 2: 内存类型和主板匹配 - Check 2: Memory type and motherboard match
            if motherboard.memory_type and memory.memory_type:
                if motherboard.memory_type != memory.memory_type:
                    issues.append("Memory generation mismatch with motherboard")

            # 检查 3: 显卡长度和机箱匹配 - Check 3: GPU length and case match
            if gpu.length_mm > case.length_mm:
                issues.append("GPU is too long for selected case")

            # 检查 4: 散热器高度和机箱匹配 - Check 4: Cooler height and case match
            if cooler.height_mm > case.height_mm and cooler.height_mm > 100:
                issues.append("Air cooler height exceeds case limit")

            # 检查 5: 电源功率是否足够 - Check 5: PSU wattage is sufficient
            needed = int((cpu.watt + gpu.watt + 120) * 1.35)
            if psu.watt < needed:
                issues.append("PSU wattage may be insufficient")

            # 检查 6: 主板和机箱规格匹配 - Check 6: Motherboard and case form factor match
            if motherboard.form_factor and case.form_factor:
                if motherboard.form_factor == "ATX" and case.form_factor == "mATX":
                    issues.append("ATX motherboard cannot fit mATX case")

            return issues

        @tool("recommendation_context", args_schema=RecommendationContextInput)
        def recommendation_context(
            budget_min: int,
            budget_max: int,
            use_case: List[str],
            resolution: str,
        ) -> dict:
            """
            生成推荐上下文 - Generate Recommendation Context
            
            根据用户目标生成价格、GPU 和 CPU 的目标权重配置。
            Generate target weights for price, GPU, and CPU according to user goal.
            
            处理流程 Processing Flow:
            1. 根据用途调整 GPU 和 CPU 权重
               - 游戏用途增加 GPU 权重
               - 视频剪辑或 AI 用途增加 CPU 权重
            2. 返回预算范围、分辨率和权重配置
            
            参数 Parameters:
                budget_min: 最小预算
                           Min budget
                budget_max: 最大预算
                           Max budget
                use_case: 用途列表，如 ["gaming"], ["video_editing", "ai"]
                         Use case list, such as ["gaming"], ["video_editing", "ai"]
                resolution: 分辨率，如 "1080p", "1440p", "4k"
                            Resolution, such as "1080p", "1440p", "4k"
            
            返回 Returns:
                推荐上下文字典，包含预算范围、分辨率和权重配置
                Recommendation context dictionary, including budget range, resolution, and weight configuration
            """
            # 根据用途调整权重 - Adjust weights based on use case
            gpu_weight = 0.35
            cpu_weight = 0.25
            if "gaming" in use_case:
                gpu_weight += 0.15
            if "video_editing" in use_case or "ai" in use_case:
                cpu_weight += 0.1
            return {
                "budget_min": budget_min,
                "budget_max": budget_max,
                "resolution": resolution,
                "gpu_weight": round(gpu_weight, 2),
                "cpu_weight": round(cpu_weight, 2),
            }

        return {
            "search_parts": search_parts,
            "estimate_power": estimate_power,
            "check_compatibility": check_compatibility,
            "recommendation_context": recommendation_context,
        }


def pick_build_from_candidates(
    req: UserRequirements,
    search_parts_tool,
) -> BuildPlan:
    """
    从候选配件中选择配置方案 - Pick Build Plan from Candidate Parts
    
    根据用户需求和候选配件，智能选择最佳的硬件配置方案。
    Intelligently select the best hardware configuration plan based on user requirements and candidate parts.
    
    处理流程 Processing Flow:
    1. 根据总预算分配各类配件的预算
       - CPU: 20%
       - 主板: 13%
       - 内存: 8%
       - 存储: 8%
       - 显卡: 32%
       - 电源: 8%
       - 机箱: 6%
       - 散热器: 5%
    2. 依次选择各类配件，考虑品牌偏好、性能优先级和兼容性
    3. 确保配件之间的兼容性（插槽、尺寸、功耗等）
    4. 返回完整的配置方案
    
    参数 Parameters:
        req: 用户需求对象，包含预算、用途、品牌偏好等
             User requirements object, including budget, use case, brand preferences, etc.
        search_parts_tool: 搜索配件的工具函数
                          Tool function for searching parts
    
    返回 Returns:
        完整的配置方案
        Complete build plan
    """
    # 步骤 1: 分配预算 - Step 1: Allocate budget
    budgets = {
        "cpu": int(req.budget_max * 0.2),
        "motherboard": int(req.budget_max * 0.13),
        "memory": int(req.budget_max * 0.08),
        "storage": int(req.budget_max * 0.08),
        "gpu": int(req.budget_max * 0.32),
        "psu": int(req.budget_max * 0.08),
        "case": int(req.budget_max * 0.06),
        "cooler": int(req.budget_max * 0.05),
    }

    build = BuildPlan()

    def choose(category: str, predicate=None, strict_predicate: bool = False, is_fallback: bool = False) -> Part | None:
        """
        选择配件 - Choose Part

        根据类别、预算和约束条件选择最佳配件。
        Select the best part based on category, budget, and constraints.

        参数 Parameters:
            category: 配件类别
                      Part category
            predicate: 可选的谓词函数，用于进一步过滤配件
                      Optional predicate function for further filtering parts
            strict_predicate: 是否严格应用谓词，如果为 True 且没有匹配项则返回 None
                             Whether to strictly apply predicate, if True and no match then return None
            is_fallback: 是否为兜底模式，如果是则忽略品牌偏好，选择最便宜的
                         Whether this is fallback mode, if True ignore brand preferences and choose cheapest

        返回 Returns:
            选中的配件，如果没有找到则返回 None
            Selected part, or None if not found
        """
        def _norm(value: str) -> str:
            return value.strip().lower()

        def _brand_matches(item: Part, brand: str) -> bool:
            brand_norm = _norm(brand)
            if not brand_norm:
                return False
            item_brand_norm = _norm(item.brand)
            item_name_norm = _norm(item.name)
            matches = item_brand_norm == brand_norm or f" {brand_norm} " in f" {item_name_norm} "
            return matches

        def _has_preferred_brand(raw_items: List[dict], preferred: str) -> bool:
            items_local = [Part.model_validate(item) for item in raw_items]
            return any(_brand_matches(item, preferred) for item in items_local)

        # 兜底模式处理 - Fallback mode handling
        if is_fallback:
            category_prefer_brands = []
            category_exclude_brands = []
            cpu_preference = ""
        else:
            category_prefer_brands = list(req.prefer_brands)
            category_exclude_brands = list(req.brand_blacklist)
            cpu_preference = (req.cpu_preference or "").strip()

            # CPU 品牌偏好处理 - CPU brand preference handling
            if category == "cpu" and cpu_preference:
                category_prefer_brands = [cpu_preference]
                category_exclude_brands = [
                    b for b in category_exclude_brands if b.strip().lower() != cpu_preference.lower()
                ]

        # 第一次搜索：使用类别预算 - First search: use category budget
        raw = search_parts_tool.invoke(
            {
                "category": category,
                "budget_max": max(budgets[category], 200),
                "prefer_brands": category_prefer_brands,
                "exclude_brands": category_exclude_brands,
            }
        )

        # 判断是否需要扩展预算 - Check if budget expansion is needed
        should_expand_budget = (
            category == "cpu"
            and bool(cpu_preference)
            and raw
            and not _has_preferred_brand(raw, cpu_preference)
        )
        if not raw or should_expand_budget:
            # 第二次搜索：扩展预算 - Second search: expand budget
            raw = search_parts_tool.invoke(
                {
                    "category": category,
                    "budget_max": max(req.budget_max, 200),
                    "prefer_brands": category_prefer_brands,
                    "exclude_brands": category_exclude_brands,
                }
            )
        if not raw:
            return None
        items = [Part.model_validate(item) for item in raw]

        # CPU 品牌匹配 - CPU brand matching
        if category == "cpu" and cpu_preference:
            matched_cpu_brand = [
                item for item in items if _brand_matches(item, cpu_preference)
            ]
            if matched_cpu_brand:
                items = matched_cpu_brand
            else:
                return None

        # 根据优先级排序 - Sort by priority
        if req.priority == "budget" or is_fallback:
            items.sort(key=lambda p: p.price)
        elif req.priority == "performance":
            items.sort(key=lambda p: (-p.score, p.price))

        # 应用谓词过滤 - Apply predicate filtering
        if predicate:
            for item in items:
                if predicate(item):
                    return item
            if strict_predicate:
                return None
            return items[0]

        return items[0]

    # 步骤 2: 依次选择各类配件 - Step 2: Choose parts in order

    # 选择 CPU - Choose CPU
    build.cpu = choose("cpu")

    # CPU兜底机制：如果CPU没找到，选择最便宜的
    if build.cpu is None:
        build.cpu = choose("cpu", is_fallback=True)
    
    # 选择主板（必须与 CPU 插槽匹配）- Choose motherboard (must match CPU socket)
    # 如果预算内找不到匹配的主板，扩大预算重试
    # If no matching motherboard found within budget, expand budget and retry
    build.motherboard = choose(
        "motherboard",
        predicate=(lambda p: build.cpu is not None and p.socket == build.cpu.socket),
        strict_predicate=True,
    )
    
    # 如果主板没找到，扩大预算重试
    if build.motherboard is None and build.cpu:
        original_budget = budgets.get("motherboard", 0)
        expanded_budget = max(original_budget * 2, req.budget_max)
        budgets["motherboard"] = expanded_budget
        build.motherboard = choose(
            "motherboard",
            predicate=(lambda p: build.cpu is not None and p.socket == build.cpu.socket),
            strict_predicate=True,
        )
        budgets["motherboard"] = original_budget

    # 主板兜底机制：如果还是没找到，选择最便宜的主板
    if build.motherboard is None:
        build.motherboard = choose("motherboard", is_fallback=True)
    
    # 选择内存（必须与主板内存类型匹配，严格匹配）- Choose memory (must match motherboard memory type, strict match)
    # 如果主板memory_type为空，则不使用严格匹配，允许选择任意内存
    # If motherboard memory_type is empty, don't use strict matching, allow any memory
    if build.motherboard and build.motherboard.memory_type:
        build.memory = choose(
            "memory",
            predicate=(
                lambda p: build.motherboard is not None
                and p.memory_type == build.motherboard.memory_type
            ),
            strict_predicate=True,
        )
        
        # 内存兜底机制：如果没找到匹配的内存，扩大预算重试
        if build.memory is None:
            original_budget = budgets.get("memory", 0)
            expanded_budget = max(original_budget * 2, req.budget_max)
            budgets["memory"] = expanded_budget
            build.memory = choose(
                "memory",
                predicate=(
                    lambda p: build.motherboard is not None
                    and p.memory_type == build.motherboard.memory_type
                ),
                strict_predicate=True,
            )
            budgets["memory"] = original_budget

        # 内存兜底机制：如果还是没找到，选择最便宜的内存
        if build.memory is None:
            build.memory = choose("memory", is_fallback=True)
    else:
        # 主板没有指定内存类型，选择最佳内存（优先DDR5）
        build.memory = choose("memory")

        # 内存兜底机制：如果没找到，选择最便宜的
        if build.memory is None:
            build.memory = choose("memory", is_fallback=True)
    
    # 选择存储 - Choose storage
    build.storage = choose("storage")
    
    # 存储兜底机制：如果没找到，选择最便宜的
    if build.storage is None:
        build.storage = choose("storage", is_fallback=True)

    # 选择显卡 - Choose GPU
    build.gpu = choose("gpu")

    # 显卡兜底机制：如果没找到，可以为空（低端配置）

    # 计算目标电源功率 - Calculate target PSU wattage
    estimated_draw = 0
    if build.cpu:
        estimated_draw += build.cpu.watt
    if build.gpu:
        estimated_draw += build.gpu.watt
    target_psu = int((estimated_draw + 120) * 1.35)

    # 选择电源（功率必须足够）- Choose PSU (wattage must be sufficient)
    # 如果预算内找不到足够功率的电源，扩大预算重试
    # If no PSU with sufficient wattage found within budget, expand budget and retry
    build.psu = choose("psu", predicate=(lambda p: p.watt >= target_psu))
    
    # 如果电源没找到，扩大预算重试
    if build.psu is None:
        original_budget = budgets.get("psu", 0)
        expanded_budget = max(original_budget * 2, req.budget_max)
        budgets["psu"] = expanded_budget
        build.psu = choose("psu", predicate=(lambda p: p.watt >= target_psu))
        budgets["psu"] = original_budget

    # 电源兜底机制：如果还是没找到，选择最便宜的电源
    if build.psu is None:
        build.psu = choose("psu", is_fallback=True)
    
    # 选择机箱（必须与主板规格匹配，且能容纳显卡）- Choose case (must match motherboard form factor and fit GPU)
    build.case = choose(
        "case",
        predicate=(
            lambda p: build.motherboard is not None
            and not (
                build.motherboard.form_factor == "ATX" and p.form_factor == "mATX"
            )
            and (build.gpu is None or p.length_mm >= build.gpu.length_mm)
        ),
    )
    
    # 机箱兜底机制：如果没找到，扩大预算重试
    if build.case is None:
        original_budget = budgets.get("case", 0)
        expanded_budget = max(original_budget * 2, req.budget_max)
        budgets["case"] = expanded_budget
        build.case = choose(
            "case",
            predicate=(
                lambda p: build.motherboard is not None
                and not (
                    build.motherboard.form_factor == "ATX" and p.form_factor == "mATX"
                )
                and (build.gpu is None or p.length_mm >= build.gpu.length_mm)
            ),
        )
        budgets["case"] = original_budget

    # 机箱兜底机制：如果还是没找到，选择最便宜的机箱
    if build.case is None:
        build.case = choose("case", is_fallback=True)
    
    # 选择散热器（高度必须适合机箱）- Choose cooler (height must fit case)
    # 如果预算内找不到合适的散热器，扩大预算重试
    # If no suitable cooler found within budget, expand budget and retry
    build.cooler = choose(
        "cooler",
        predicate=(lambda p: build.case is None or p.height_mm <= build.case.height_mm),
    )
    
    # 如果散热器没找到，扩大预算重试
    if build.cooler is None:
        original_budget = budgets.get("cooler", 0)
        expanded_budget = max(original_budget * 2, req.budget_max)
        budgets["cooler"] = expanded_budget
        build.cooler = choose(
            "cooler",
            predicate=(lambda p: build.case is None or p.height_mm <= build.case.height_mm),
        )
        budgets["cooler"] = original_budget

    # 散热器兜底机制：如果还是没找到，选择最便宜的散热器
    if build.cooler is None:
        build.cooler = choose("cooler", is_fallback=True)

    return build
