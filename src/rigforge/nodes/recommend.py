"""配置推荐节点"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models import UserRequirements, BuildPlan


def recommend_build(
    requirements: "UserRequirements",
    search_parts_tool,
) -> "BuildPlan":
    """生成配置方案
    
    Args:
        requirements: 用户需求
        search_parts_tool: 配件搜索工具
        
    Returns:
        BuildPlan 实例
    """
    from ..builder.picker import pick_build_from_candidates
    
    return pick_build_from_candidates(requirements, search_parts_tool)


def recommend_build_node(state: dict) -> dict:
    """配置推荐节点入口函数"""
    requirements = state.get("requirements")
    search_parts_tool = state.get("search_parts_tool")
    
    build = recommend_build(requirements, search_parts_tool)
    
    return {"build": build}
