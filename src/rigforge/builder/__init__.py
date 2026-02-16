"""Builder 模块：配置生成与兼容性检查"""

from .compatibility import check_compatibility, validate_build
from .budget import allocate_budget, BudgetAllocation
from .picker import pick_build_from_candidates

__all__ = [
    "check_compatibility",
    "validate_build",
    "allocate_budget",
    "BudgetAllocation",
    "pick_build_from_candidates",
]
