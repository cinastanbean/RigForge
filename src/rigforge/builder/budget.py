"""
预算分配模块 - Budget Allocation Module

根据总预算和用户用途，智能分配各类配件的预算。
Intelligently allocate budget for each part category based on total budget and user use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BudgetAllocation:
    """
    预算分配结果 - Budget Allocation Result
    
    表示各类配件的预算分配结果。
    Represents budget allocation result for each part category.
    
    字段说明 Field Descriptions:
    - cpu: CPU 预算
    - motherboard: 主板预算
    - memory: 内存预算
    - storage: 存储预算
    - gpu: 显卡预算
    - psu: 电源预算
    - case: 机箱预算
    - cooler: 散热器预算
    """
    cpu: int
    """
    CPU 预算 - CPU Budget
    
    CPU 配件的预算金额（元）。
    Budget amount for CPU part (CNY).
    """
    motherboard: int
    """
    主板预算 - Motherboard Budget
    
    主板配件的预算金额（元）。
    Budget amount for motherboard part (CNY).
    """
    memory: int
    """
    内存预算 - Memory Budget
    
    内存配件的预算金额（元）。
    Budget amount for memory part (CNY).
    """
    storage: int
    """
    存储预算 - Storage Budget
    
    存储配件的预算金额（元）。
    Budget amount for storage part (CNY).
    """
    gpu: int
    """
    显卡预算 - GPU Budget
    
    显卡配件的预算金额（元）。
    Budget amount for GPU part (CNY).
    """
    psu: int
    """
    电源预算 - PSU Budget
    
    电源配件的预算金额（元）。
    Budget amount for PSU part (CNY).
    """
    case: int
    """
    机箱预算 - Case Budget
    
    机箱配件的预算金额（元）。
    Budget amount for case part (CNY).
    """
    cooler: int
    """
    散热器预算 - Cooler Budget
    
    散热器配件的预算金额（元）。
    Budget amount for cooler part (CNY).
    """
    
    def to_dict(self) -> Dict[str, int]:
        """
        转换为字典 - Convert to Dictionary
        
        将预算分配结果转换为字典格式。
        Convert budget allocation result to dictionary format.
        
        返回 Returns:
            预算分配字典
            Budget allocation dictionary
        """
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


# 默认预算分配比例 - Default Budget Allocation Weights
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
"""
默认预算分配权重 - Default Budget Allocation Weights

各类配件的默认预算分配比例。
Default budget allocation ratios for each part category.

权重说明 Weight Descriptions:
- cpu: 20% - CPU 占总预算的 20%
- motherboard: 13% - 主板占总预算的 13%
- memory: 8% - 内存占总预算的 8%
- storage: 8% - 存储占总预算的 8%
- gpu: 32% - 显卡占总预算的 32%（通常是最重要的配件）
- psu: 8% - 电源占总预算的 8%
- case: 6% - 机箱占总预算的 6%
- cooler: 5% - 散热器占总预算的 5%
"""


def allocate_budget(
    budget_max: int,
    use_case: List[str] | None = None,
    custom_weights: Dict[str, float] | None = None,
) -> BudgetAllocation:
    """
    分配预算 - Allocate Budget
    
    根据总预算和用户用途，智能分配各类配件的预算。
    Intelligently allocate budget for each part category based on total budget and user use cases.
    
    分配策略 Allocation Strategy:
    1. 使用默认权重作为基础
    2. 根据用途调整权重：
       - 游戏用途：增加显卡预算，减少 CPU 预算
       - 视频剪辑/AI 用途：增加 CPU 和内存预算，减少显卡预算
    3. 应用自定义权重（如果提供）
    4. 计算各类配件的预算金额
    
    参数 Parameters:
        budget_max: 总预算上限（元）
                    Total budget limit (CNY)
        use_case: 用途列表，如 ["gaming"], ["video_editing", "ai"]
                  Use case list, such as ["gaming"], ["video_editing", "ai"]
        custom_weights: 自定义权重，覆盖默认值
                        Custom weights to override default values
    
    返回 Returns:
        预算分配结果对象
        Budget allocation result object
    """
    weights = DEFAULT_BUDGET_WEIGHTS.copy()
    
    # 根据用途调整权重 - Adjust weights based on use case
    if use_case:
        if "gaming" in use_case:
            # 游戏用途：增加显卡预算，减少 CPU 预算
            # Gaming use case: increase GPU budget, decrease CPU budget
            weights["gpu"] += 0.05
            weights["cpu"] -= 0.03
        if "video_editing" in use_case or "ai" in use_case:
            # 视频剪辑/AI 用途：增加 CPU 和内存预算，减少显卡预算
            # Video editing/AI use case: increase CPU and memory budget, decrease GPU budget
            weights["cpu"] += 0.05
            weights["memory"] += 0.03
            weights["gpu"] -= 0.03
    
    # 应用自定义权重 - Apply custom weights
    if custom_weights:
        for key, value in custom_weights.items():
            if key in weights:
                weights[key] = value
    
    # 计算各配件预算 - Calculate budget for each part
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
