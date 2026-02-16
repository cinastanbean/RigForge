"""预算分配模块"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BudgetAllocation:
    """预算分配结果"""
    cpu: int
    motherboard: int
    memory: int
    storage: int
    gpu: int
    psu: int
    case: int
    cooler: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "cpu": self.cpu,
            "motherboard": self.motherboard,
            "memory": self.memory,
            "storage": self.storage,
            "gpu": self.gpu,
            "psu": self.psu,
            "case": self.case,
            "cooler": self.cooler,
        }


# 默认预算分配比例
DEFAULT_BUDGET_WEIGHTS = {
    "cpu": 0.20,
    "motherboard": 0.13,
    "memory": 0.08,
    "storage": 0.08,
    "gpu": 0.32,
    "psu": 0.08,
    "case": 0.06,
    "cooler": 0.05,
}


def allocate_budget(
    budget_max: int,
    use_case: List[str] | None = None,
    custom_weights: Dict[str, float] | None = None,
) -> BudgetAllocation:
    """根据预算和用途分配各配件预算
    
    Args:
        budget_max: 总预算上限
        use_case: 用途列表 (gaming/video_editing/ai/office)
        custom_weights: 自定义权重，覆盖默认值
        
    Returns:
        BudgetAllocation 实例
    """
    weights = DEFAULT_BUDGET_WEIGHTS.copy()
    
    # 根据用途调整权重
    if use_case:
        if "gaming" in use_case:
            weights["gpu"] += 0.05
            weights["cpu"] -= 0.03
        if "video_editing" in use_case or "ai" in use_case:
            weights["cpu"] += 0.05
            weights["memory"] += 0.03
            weights["gpu"] -= 0.03
    
    # 应用自定义权重
    if custom_weights:
        for key, value in custom_weights.items():
            if key in weights:
                weights[key] = value
    
    # 计算各配件预算
    return BudgetAllocation(
        cpu=int(budget_max * weights["cpu"]),
        motherboard=int(budget_max * weights["motherboard"]),
        memory=int(budget_max * weights["memory"]),
        storage=int(budget_max * weights["storage"]),
        gpu=int(budget_max * weights["gpu"]),
        psu=int(budget_max * weights["psu"]),
        case=int(budget_max * weights["case"]),
        cooler=int(budget_max * weights["cooler"]),
    )
