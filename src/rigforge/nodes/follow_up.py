"""追问生成节点"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models import UserRequirements


# 追问问题模板
FOLLOW_UP_QUESTIONS = {
    "budget": [
        "请问您的预算范围是多少呢？比如 5000-8000 元？",
        "您打算花多少钱配这台电脑？",
    ],
    "use_case": [
        "这台电脑主要用来做什么呢？游戏、办公还是视频剪辑？",
        "请问您的用途是什么？比如打游戏、剪辑视频、AI训练？",
    ],
    "resolution": [
        "您对显示器分辨率有什么要求吗？1080p、2K 还是 4K？",
        "请问您用的是什么分辨率显示器？",
    ],
    "storage": [
        "您需要多大的存储空间？512GB、1TB 还是 2TB？",
        "对硬盘容量有什么要求吗？",
    ],
    "noise": [
        "您对噪音敏感吗？需要静音配置吗？",
        "机箱需要特别静音吗？",
    ],
}


def generate_follow_up(
    requirements: "UserRequirements",
    missing_fields: List[str],
    avoid_field: str | None = None,
) -> List[str]:
    """生成追问问题列表
    
    Args:
        requirements: 当前已收集的需求
        missing_fields: 缺失的字段列表
        avoid_field: 需要避免的字段（上次已问过）
        
    Returns:
        问题列表
    """
    questions = []
    
    # 按优先级排序缺失字段
    priority_order = ["budget", "use_case", "resolution", "storage", "noise"]
    ordered_missing = sorted(
        missing_fields,
        key=lambda x: priority_order.index(x) if x in priority_order else 999
    )
    
    for field in ordered_missing:
        if field == avoid_field:
            continue
        
        if field in FOLLOW_UP_QUESTIONS:
            templates = FOLLOW_UP_QUESTIONS[field]
            # 简单选择第一个模板
            questions.append(templates[0])
            break  # 每次只问一个问题
    
    return questions


def generate_follow_up_node(state: dict) -> dict:
    """追问生成节点入口函数"""
    requirements = state.get("requirements")
    missing_fields = state.get("follow_up_questions", [])
    avoid_field = state.get("avoid_repeat_field")
    
    questions = generate_follow_up(requirements, missing_fields, avoid_field)
    
    return {"follow_up_questions": questions}
