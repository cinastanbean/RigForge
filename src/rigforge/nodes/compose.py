"""回复组装节点"""

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
    """组装回复
    
    Args:
        build: 配置方案（可能为 None）
        requirements: 用户需求
        issues: 兼容性问题
        llm: LLM 实例（可选）
        enthusiasm_level: 热情程度 (standard/high)
        
    Returns:
        回复文本
    """
    if build is None:
        # 追问场景
        return compose_followup_reply(requirements, llm, enthusiasm_level)
    else:
        # 推荐场景
        return compose_recommendation_reply(build, requirements, issues, llm, enthusiasm_level)


def compose_followup_reply(
    requirements: "UserRequirements",
    llm: Optional["ChatOpenAI"],
    enthusiasm_level: str,
) -> str:
    """组装追问回复"""
    # 简单的模板回复
    prefix = "太棒了！" if enthusiasm_level == "high" else "收到，"
    
    missing = []
    if requirements.budget_max is None:
        missing.append("预算")
    if requirements.use_case is None:
        missing.append("用途")
    if requirements.resolution is None:
        missing.append("分辨率")
    
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
    """组装推荐回复"""
    lines = []
    
    # 开场
    if enthusiasm_level == "high":
        lines.append("太棒了！为您找到了一套很棒的配置！")
    else:
        lines.append("根据您的需求，推荐以下配置：")
    
    # 配置清单
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
    
    # 总价
    lines.append(f"\n💰 总价: ¥{build.total_price()}")
    
    # 兼容性问题
    if issues:
        lines.append("\n⚠️ 注意事项：")
        for issue in issues:
            lines.append(f"  - {issue}")
    
    return "\n".join(lines)


def compose_reply_node(state: dict) -> dict:
    """回复组装节点入口函数"""
    build = state.get("build")
    requirements = state.get("requirements")
    issues = state.get("compatibility_issues", [])
    llm = state.get("llm")
    enthusiasm_level = state.get("enthusiasm_level", "standard")
    
    reply = compose_reply(build, requirements, issues, llm, enthusiasm_level)
    
    return {"response_text": reply}
