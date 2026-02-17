"""
配置推荐节点 - Build Recommendation Node

根据用户需求生成硬件配置方案。
Generate hardware build plan based on user requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models import UserRequirements, BuildPlan


def recommend_build(
    requirements: "UserRequirements",
    search_parts_tool,
) -> "BuildPlan":
    """
    生成配置方案 - Generate Build Plan
    
    根据用户需求和配件搜索工具，生成硬件配置方案。
    Generate hardware build plan based on user requirements and parts search tool.
    
    参数 Parameters:
        requirements: 用户需求
                     User requirements
        search_parts_tool: 配件搜索工具
                           Parts search tool
    
    返回 Returns:
        配置方案
        Build plan
    """
    from ..builder.picker import pick_build_from_candidates
    
    return pick_build_from_candidates(requirements, search_parts_tool)


def recommend_build_node(state: dict) -> dict:
    """
    配置推荐节点入口函数 - Build Recommendation Node Entry Function
    
    此函数将被 graph.py 调用。
    This function will be called by graph.py.
    
    参数 Parameters:
        state: 当前状态字典
               Current state dictionary
    
    返回 Returns:
        更新后的状态字典，包含配置方案
        Updated state dictionary containing build plan
    """
    requirements = state.get("requirements")
    search_parts_tool = state.get("search_parts_tool")
    
    build = recommend_build(requirements, search_parts_tool)
    
    return {"build": build}
