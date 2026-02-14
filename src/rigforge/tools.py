from __future__ import annotations

from typing import List, Protocol

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .schemas import BuildPlan, Part, UserRequirements


class PartsRepoProtocol(Protocol):
    def all_parts(self) -> List[Part]: ...
    def by_category(self, category: str) -> List[Part]: ...
    def find_by_sku(self, sku: str) -> Part | None: ...


class SearchPartsInput(BaseModel):
    category: str = Field(description="Part category such as cpu, gpu, motherboard")
    budget_max: int = Field(description="Max acceptable price for this category")
    prefer_brands: List[str] = Field(default_factory=list)
    exclude_brands: List[str] = Field(default_factory=list)


class CompatibilityInput(BaseModel):
    cpu_sku: str
    motherboard_sku: str
    memory_sku: str
    gpu_sku: str
    psu_sku: str
    case_sku: str
    cooler_sku: str


class RecommendationContextInput(BaseModel):
    budget_min: int
    budget_max: int
    use_case: List[str] = Field(default_factory=list)
    resolution: str = "1080p"


class Toolset:
    def __init__(
        self,
        repo: PartsRepoProtocol,
        *,
        build_data_source: str = "csv(jd+newegg)",
        build_data_version: str = "v0",
        build_data_mode: str = "jd_newegg",
    ):
        self.repo = repo
        self.build_data_source = build_data_source
        self.build_data_version = build_data_version
        self.build_data_mode = build_data_mode

    def register(self):
        repo = self.repo

        @tool("search_parts", args_schema=SearchPartsInput)
        def search_parts(
            category: str,
            budget_max: int,
            prefer_brands: List[str] | None = None,
            exclude_brands: List[str] | None = None,
        ) -> List[dict]:
            """Search parts by category and budget, sorted by performance score."""
            prefer_brands = prefer_brands or []
            exclude_brands = exclude_brands or []

            def _norm(value: str) -> str:
                return value.strip().lower()

            def _brand_matches(part: Part, brand_norm: str) -> bool:
                if not brand_norm:
                    return False
                part_brand = _norm(part.brand)
                part_name = _norm(part.name)
                return part_brand == brand_norm or f" {brand_norm} " in f" {part_name} "

            prefer_brands_norm = {_norm(b) for b in prefer_brands if b and b.strip()}
            exclude_brands_norm = {_norm(b) for b in exclude_brands if b and b.strip()}
            candidates = [
                p
                for p in repo.by_category(category)
                if p.price <= budget_max
                and not any(_brand_matches(p, blocked) for blocked in exclude_brands_norm)
            ]
            if prefer_brands_norm:
                candidates.sort(
                    key=lambda p: (
                        not any(_brand_matches(p, preferred) for preferred in prefer_brands_norm),
                        -p.score,
                        p.price,
                    )
                )
            else:
                candidates.sort(key=lambda p: (-p.score, p.price))
            return [c.model_dump() for c in candidates[:5]]

        @tool("estimate_power")
        def estimate_power(parts: List[str]) -> int:
            """Estimate system power draw by summing part watt usage and adding headroom."""
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
            """Validate core hardware compatibility constraints."""
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

            if cpu.socket != motherboard.socket:
                issues.append("CPU and motherboard socket mismatch")

            if motherboard.memory_type and memory.memory_type:
                if motherboard.memory_type != memory.memory_type:
                    issues.append("Memory generation mismatch with motherboard")

            if gpu.length_mm > case.length_mm:
                issues.append("GPU is too long for selected case")

            if cooler.height_mm > case.height_mm and cooler.height_mm > 100:
                issues.append("Air cooler height exceeds case limit")

            needed = int((cpu.watt + gpu.watt + 120) * 1.35)
            if psu.watt < needed:
                issues.append("PSU wattage may be insufficient")

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
            """Generate target weights for price, GPU, and CPU according to user goal."""
            gpu_weight = 0.35
            cpu_weight = 0.25
            if "gaming" in use_case:
                gpu_weight += 0.15
            if "video_editing" in use_case or "ai" in use_case:
                cpu_weight += 0.1
            target = {
                "budget_min": budget_min,
                "budget_max": budget_max,
                "resolution": resolution,
                "gpu_weight": round(gpu_weight, 2),
                "cpu_weight": round(cpu_weight, 2),
            }
            return target

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

    def choose(category: str, predicate=None, strict_predicate: bool = False) -> Part | None:
        def _norm(value: str) -> str:
            return value.strip().lower()

        def _brand_matches(item: Part, brand: str) -> bool:
            brand_norm = _norm(brand)
            if not brand_norm:
                return False
            return _norm(item.brand) == brand_norm or f" {brand_norm} " in f" {_norm(item.name)} "

        def _has_preferred_brand(raw_items: List[dict], preferred: str) -> bool:
            items_local = [Part.model_validate(item) for item in raw_items]
            return any(_brand_matches(item, preferred) for item in items_local)

        is_fallback = False
        category_prefer_brands = list(req.prefer_brands)
        category_exclude_brands = list(req.brand_blacklist)
        cpu_preference = (req.cpu_preference or "").strip()
        if category == "cpu" and cpu_preference:
            # Explicit CPU preference should be stronger than generic brand history.
            category_prefer_brands = [cpu_preference]
            category_exclude_brands = [
                b for b in category_exclude_brands if b.strip().lower() != cpu_preference.lower()
            ]
        raw = search_parts_tool.invoke(
            {
                "category": category,
                "budget_max": max(budgets[category], 200),
                "prefer_brands": category_prefer_brands,
                "exclude_brands": category_exclude_brands,
            }
        )
        should_expand_budget = (
            category == "cpu"
            and bool(cpu_preference)
            and raw
            and not _has_preferred_brand(raw, cpu_preference)
        )
        if not raw or should_expand_budget:
            is_fallback = True
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
        if category == "cpu" and cpu_preference:
            matched_cpu_brand = [
                item for item in items if _brand_matches(item, cpu_preference)
            ]
            if matched_cpu_brand:
                items = matched_cpu_brand
            else:
                # Do not silently violate an explicit CPU brand preference.
                return None
        if req.priority == "budget" or is_fallback:
            items.sort(key=lambda p: p.price)
        elif req.priority == "performance":
            items.sort(key=lambda p: (-p.score, p.price))
        if predicate:
            for item in items:
                if predicate(item):
                    return item
            if strict_predicate:
                return None
        return items[0]

    build.cpu = choose("cpu")
    build.motherboard = choose(
        "motherboard",
        predicate=(lambda p: build.cpu is not None and p.socket == build.cpu.socket),
    )
    build.memory = choose(
        "memory",
        predicate=(
            lambda p: build.motherboard is not None
            and p.memory_type == build.motherboard.memory_type
        ),
        strict_predicate=True,
    )
    build.storage = choose("storage")
    build.gpu = choose("gpu")

    estimated_draw = 0
    if build.cpu:
        estimated_draw += build.cpu.watt
    if build.gpu:
        estimated_draw += build.gpu.watt
    target_psu = int((estimated_draw + 120) * 1.35)

    build.psu = choose("psu", predicate=(lambda p: p.watt >= target_psu))
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
    build.cooler = choose(
        "cooler",
        predicate=(lambda p: build.case is None or p.height_mm <= build.case.height_mm),
    )

    return build
