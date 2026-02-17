"""
回复组装节点 - Reply Composition Node

组装自然语言回复，包括追问回复和推荐回复。
Assemble natural language replies, including follow-up replies and recommendation replies.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from ..data.models import BuildPlan, UserRequirements


def compose_reply(
    build: Optional["BuildPlan"],
    requirements: "UserRequirements",
    issues: List[str],
    llm: Optional["ChatOpenAI"] = None,
    enthusiasm_level: str = "standard",
) -> str:
    """
    组装回复 - Compose Reply
    
    根据配置方案和用户需求，组装自然语言回复。
    Assemble natural language reply based on build plan and user requirements.
    
    参数 Parameters:
        build: 配置方案（可能为 None）
               Build plan (may be None)
        requirements: 用户需求
                     User requirements
        issues: 兼容性问题列表
                 List of compatibility issues
        llm: LLM 实例（可选）
             LLM instance (optional)
        enthusiasm_level: 热情程度
                           Enthusiasm level (standard/high)
    
    返回 Returns:
        回复文本
        Reply text
    """
    if build is None:
        # 追问场景 - Follow-up scenario
        return compose_followup_reply(requirements, llm, enthusiasm_level)
    else:
        # 推荐场景 - Recommendation scenario
        return compose_recommendation_reply(build, requirements, issues, llm, enthusiasm_level)


def compose_followup_reply(
    requirements: "UserRequirements",
    llm: Optional["ChatOpenAI"],
    enthusiasm_level: str,
) -> str:
    """
    组装追问回复 - Compose Follow-up Reply
    
    根据缺失的需求信息，生成追问回复。
    Generate follow-up reply based on missing requirement information.
    
    参数 Parameters:
        requirements: 用户需求
                     User requirements
        llm: LLM 实例（可选）
             LLM instance (optional)
        enthusiasm_level: 热情程度
                           Enthusiasm level
    
    返回 Returns:
        追问回复文本
        Follow-up reply text
    """
    # 简单的模板回复 - Simple template reply
    prefix = "太棒了！" if enthusiasm_level == "high" else "收到，"
    
    # 检查缺失的信息 - Check missing information
    missing = []
    if requirements.budget_max is None:
        missing.append("预算")
    if requirements.use_case is None:
        missing.append("用途")
    if requirements.resolution is None:
        missing.append("分辨率")
    
    # 根据缺失信息生成追问 - Generate follow-up based on missing information
    if missing:
        return f"{prefix}请问您的{missing[0]}是多少呢？"
    else:
        return f"{prefix}信息已收集完毕，正在为您生成配置方案..."


def compose_recommendation_reply(
    build: "BuildPlan",
    requirements: "UserRequirements",
    issues: List[str],
    llm: Optional["ChatOpenAI"],
    enthusiasm_level: str,
) -> str:
    """
    组装推荐回复 - Compose Recommendation Reply
    
    根据配置方案和兼容性问题，生成推荐回复。
    Generate recommendation reply based on build plan and compatibility issues.
    
    参数 Parameters:
        build: 配置方案
               Build plan
        requirements: 用户需求
                     User requirements
        issues: 兼容性问题列表
                 List of compatibility issues
        llm: LLM 实例（可选）
             LLM instance (optional)
        enthusiasm_level: 热情程度
                           Enthusiasm level
    
    返回 Returns:
        推荐回复文本
        Recommendation reply text
    """
    lines = []
    
    # 开场 - Opening
    if enthusiasm_level == "high":
        lines.append("太棒了！为您找到了一套很棒的配置！")
    else:
        lines.append("根据您的需求，推荐以下配置：")
    
    # 配置清单 - Build list
    lines.append("")
    if build.cpu:
        lines.append(f"🖥️ CPU: {build.cpu.name} - ¥{build.cpu.price}")
    if build.motherboard:
        lines.append(f"🔧 主板: {build.motherboard.name} - ¥{build.motherboard.price}")
    if build.memory:
        lines.append(f"💾 内存: {build.memory.name} - ¥{build.memory.price}")
    if build.gpu:
        lines.append(f"🎮 显卡: {build.gpu.name} - ¥{build.gpu.price}")
    if build.storage:
        lines.append(f"💿 存储: {build.storage.name} - ¥{build.storage.price}")
    if build.psu:
        lines.append(f"⚡ 电源: {build.psu.name} - ¥{build.psu.price}")
    if build.case:
        lines.append(f"📦 机箱: {build.case.name} - ¥{build.case.price}")
    if build.cooler:
        lines.append(f"❄️ 散热: {build.cooler.name} - ¥{build.cooler.price}")
    
    # 总价 - Total price
    lines.append(f"\n💰 总价: ¥{build.total_price()}")
    
    # 兼容性问题 - Compatibility issues
    if issues:
        lines.append("\n⚠️ 注意事项：")
        for issue in issues:
            lines.append(f"  - {issue}")
    
    return "\n".join(lines)


def compose_reply_node(state: dict) -> dict:
    """
    回复组装节点入口函数 - Reply Composition Node Entry Function
    
    此函数将被 graph.py 调用。
    This function will be called by graph.py.
    
    参数 Parameters:
        state: 当前状态字典
               Current state dictionary
    
    返回 Returns:
        更新后的状态字典，包含回复文本
        Updated state dictionary containing reply text
    """
    build = state.get("build")
    requirements = state.get("requirements")
    issues = state.get("compatibility_issues", [])
    llm = state.get("llm")
    enthusiasm_level = state.get("enthusiasm_level", "standard")
    
    reply = compose_reply(build, requirements, issues, llm, enthusiasm_level)
    
    return {"response_text": reply}
