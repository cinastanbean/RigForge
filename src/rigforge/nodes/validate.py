"""
兼容性验证节点 - Compatibility Validation Node

验证硬件配置方案的完整性和兼容性。
Validate completeness and compatibility of hardware build plan.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models import BuildPlan, UserRequirements


def validate_build(
    build: "BuildPlan",
    requirements: "UserRequirements",
) -> List[str]:
    """
    验证配置方案 - Validate Build Plan
    
    验证配置方案的完整性和兼容性。
    Validate completeness and compatibility of build plan.
    
    参数 Parameters:
        build: 配置方案
               Build plan
        requirements: 用户需求
                     User requirements
    
    返回 Returns:
        问题列表，空列表表示验证通过
        List of issues, empty list means validation passed
    """
    from ..builder.compatibility import validate_build as do_validate
    
    return do_validate(build, requirements)


def validate_build_node(state: dict) -> dict:
    """
    兼容性验证节点入口函数 - Compatibility Validation Node Entry Function
    
    此函数将被 graph.py 调用。
    This function will be called by graph.py.
    
    参数 Parameters:
        state: 当前状态字典
               Current state dictionary
    
    返回 Returns:
        更新后的状态字典，包含兼容性问题和预估功耗
        Updated state dictionary containing compatibility issues and estimated power
    """
    build = state.get("build")
    requirements = state.get("requirements")
    
    # 验证配置方案 - Validate build plan
    issues = validate_build(build, requirements)
    
    # 估算功耗 - Estimate power consumption
    estimated_power = 0
    if build:
        if build.cpu:
            estimated_power += build.cpu.watt
        if build.gpu:
            estimated_power += build.gpu.watt
        # 基础功耗 + 其他配件约 120W + 35% 余量
        # Base power + other parts ~120W + 35% headroom
        estimated_power = int((estimated_power + 120) * 1.35)
    
    return {
        "compatibility_issues": issues,
        "estimated_power": estimated_power,
    }
