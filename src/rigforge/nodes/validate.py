"""兼容性验证节点"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models import BuildPlan, UserRequirements


def validate_build(
    build: "BuildPlan",
    requirements: "UserRequirements",
) -> List[str]:
    """验证配置方案
    
    Args:
        build: 配置方案
        requirements: 用户需求
        
    Returns:
        问题列表，空列表表示验证通过
    """
    from ..builder.compatibility import validate_build as do_validate
    
    return do_validate(build, requirements)


def validate_build_node(state: dict) -> dict:
    """兼容性验证节点入口函数"""
    build = state.get("build")
    requirements = state.get("requirements")
    
    issues = validate_build(build, requirements)
    
    # 估算功耗
    estimated_power = 0
    if build:
        if build.cpu:
            estimated_power += build.cpu.watt
        if build.gpu:
            estimated_power += build.gpu.watt
        estimated_power = int((estimated_power + 120) * 1.35)
    
    return {
        "compatibility_issues": issues,
        "estimated_power": estimated_power,
    }
