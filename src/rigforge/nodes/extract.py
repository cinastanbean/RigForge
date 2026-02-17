"""
需求提取节点 - Requirement Extraction Node

从用户输入中提取和更新装机需求。
Extract and update PC build requirements from user input.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from ..data.models import UserRequirements, RequirementUpdate


class RequirementExtractor:
    """
    需求提取器 - Requirement Extractor
    
    从用户输入中提取和更新装机需求。
    Extract and update PC build requirements from user input.
    
    主要功能 Main Functions:
    1. 使用 LLM 从用户输入中提取需求信息
    2. 基于规则的需求提取（作为回退方案）
    3. 缓存提取结果以提高性能
    4. 构建已收集信息的摘要
    """
    
    def __init__(self):
        """
        初始化需求提取器 - Initialize Requirement Extractor
        
        创建缓存字典用于存储提取结果。
        Create cache dictionary for storing extraction results.
        """
        self._cache: Dict[str, RequirementUpdate] = {}
        """
        提取结果缓存 - Extraction Result Cache
        
        用于缓存已提取的需求更新结果，避免重复调用 LLM。
        Cache for storing extracted requirement updates to avoid redundant LLM calls.
        """
    
    def _cache_key(self, text: str, current: "UserRequirements") -> str:
        """
        生成缓存键 - Generate Cache Key
        
        根据用户输入和当前需求生成唯一的缓存键。
        Generate a unique cache key based on user input and current requirements.
        
        参数 Parameters:
            text: 用户输入文本
                  User input text
            current: 当前需求对象
                     Current requirements object
        
        返回 Returns:
            缓存键字符串
            Cache key string
        """
        return f"{text}:{current.model_dump_json()}"
    
    @staticmethod
    def build_collected_summary(req: "UserRequirements") -> str:
        """
        构建已收集信息的简洁摘要 - Build Concise Summary of Collected Information
        
        将已收集的需求信息转换为简洁的文本摘要。
        Convert collected requirements into a concise text summary.
        
        参数 Parameters:
            req: 需求对象
                 Requirements object
        
        返回 Returns:
            简洁的文本摘要
            Concise text summary
        """
        parts = []
        
        # 预算信息 - Budget information
        if req.budget_max is not None:
            budget_min = req.budget_min or req.budget_max
            parts.append(f"预算: {budget_min}-{req.budget_max}元")
        
        # 用途信息 - Use case information
        if req.use_case:
            use_case_map = {
                "gaming": "游戏",
                "video_editing": "视频剪辑",
                "ai": "AI开发",
                "office": "办公",
            }
            uses = [use_case_map.get(u, u) for u in req.use_case]
            parts.append(f"用途: {', '.join(uses)}")
        
        # 分辨率信息 - Resolution information
        if req.resolution:
            parts.append(f"分辨率: {req.resolution}")
        
        # CPU 信息 - CPU information
        if req.cpu_model:
            parts.append(f"CPU型号: {req.cpu_model}")
        elif req.cpu_preference:
            parts.append(f"CPU偏好: {req.cpu_preference}")
        
        # 显卡信息 - GPU information
        if req.gpu_model:
            parts.append(f"显卡型号: {req.gpu_model}")
        elif req.gpu_preference:
            parts.append(f"显卡偏好: {req.gpu_preference}")
        
        # 内存信息 - Memory information
        if req.memory_gb:
            mem_info = f"内存: {req.memory_gb}GB"
            if req.memory_type:
                mem_info += f" ({req.memory_type})"
            parts.append(mem_info)
        
        # 存储信息 - Storage information
        if req.storage_target_gb:
            parts.append(f"存储: {req.storage_target_gb}GB")
        
        # 静音需求 - Quiet requirement
        if req.need_quiet is not None:
            parts.append(f"静音: {'需要' if req.need_quiet else '不需要'}")
        
        # 品牌偏好 - Brand preferences
        if req.prefer_brands:
            parts.append(f"品牌偏好: {', '.join(req.prefer_brands)}")
        
        # 禁用品牌 - Brand blacklist
        if req.brand_blacklist:
            parts.append(f"禁用品牌: {', '.join(req.brand_blacklist)}")
        
        # 游戏列表 - Game titles
        if req.game_titles:
            parts.append(f"游戏: {', '.join(req.game_titles)}")
        
        # 优先级 - Priority
        if req.priority and req.priority != "balanced":
            priority_map = {"budget": "性价比优先", "performance": "性能优先"}
            parts.append(f"优先级: {priority_map.get(req.priority, req.priority)}")
        
        return " | ".join(parts) if parts else "暂无"
    
    def extract(
        self, 
        text: str, 
        current: "UserRequirements", 
        llm: Optional["ChatOpenAI"] = None
    ) -> "RequirementUpdate":
        """
        提取需求 - Extract Requirements
        
        从用户输入中提取需求信息。
        Extract requirement information from user input.
        
        提取策略 Extraction Strategy:
        1. 如果提供了 LLM，使用 LLM 提取需求
        2. 如果 LLM 提取失败，回退到基于规则的提取
        3. 使用缓存避免重复提取
        
        参数 Parameters:
            text: 用户输入文本
                  User input text
            current: 当前需求对象
                     Current requirements object
            llm: 可选的 LLM 实例
                Optional LLM instance
        
        返回 Returns:
            需求更新对象
            Requirement update object
        """
        from ..llm.providers import invoke_with_rate_limit, invoke_with_turn_timeout
        from ..data.models import RequirementUpdate
        
        # 如果没有提供 LLM，使用基于规则的提取
        # If LLM is not provided, use rule-based extraction
        if llm is None:
            return self._extract_with_rules(text, current)
        
        # 检查缓存 - Check cache
        cache_key = self._cache_key(text, current)
        if cache_key in self._cache:
            print("[CACHE] 命中缓存，跳过 LLM 调用")
            return self._cache[cache_key]
        
        from ..llm.prompts import REQUIREMENT_EXTRACTION_PROMPT
        
        # 构建 LLM 提示词 - Build LLM prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", REQUIREMENT_EXTRACTION_PROMPT),
            ("human", "已收集信息: {collected}\n用户本轮输入: {text}"),
        ])
        
        try:
            # 使用 LLM 提取需求 - Use LLM to extract requirements
            structured = llm.with_structured_output(RequirementUpdate)
            collected_summary = self.build_collected_summary(current)
            
            result = invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(
                    lambda: (prompt | structured).invoke({
                        "collected": collected_summary,
                        "text": text
                    })
                )
            )
            
            # 缓存结果 - Cache result
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            # LLM 提取失败，回退到基于规则的提取
            # LLM extraction failed, fallback to rule-based extraction
            print(f"[PERF] Extract failed, falling back to rules: {e}")
            return self._extract_with_rules(text, current)
    
    def _extract_with_rules(
        self, 
        text: str, 
        current: "UserRequirements"
    ) -> "RequirementUpdate":
        """
        基于规则的需求提取（回退方案）- Rule-based Requirement Extraction (Fallback)
        
        使用规则和正则表达式从用户输入中提取需求信息。
        Extract requirement information from user input using rules and regex.
        
        参数 Parameters:
            text: 用户输入文本
                  User input text
            current: 当前需求对象
                     Current requirements object
        
        返回 Returns:
            需求更新对象
            Requirement update object
        """
        from ..data.models import RequirementUpdate
        
        lower = text.lower()
        update = RequirementUpdate()
        
        # 预算提取 - Budget extraction
        values = []
        budget_tokens = ["预算", "价位", "价格", "花", "元", "块", "rmb", "￥", "¥"]
        if any(tok in lower for tok in budget_tokens):
            numbers = re.findall(r"(\d{4,6})", text)
            values.extend(int(n) for n in numbers)
            
            # 处理 X k 格式 - Handle X k format
            k_values = re.findall(r"(\d+(?:\.\d+)?)\s*k\b", lower)
            values.extend(int(float(n) * 1000) for n in k_values)
        
        if values:
            vals = sorted(values)
            if len(vals) >= 2:
                update.budget_min = vals[0]
                update.budget_max = vals[-1]
            else:
                update.budget_max = vals[0]
                update.budget_min = int(vals[0] * 0.85)
        
        # 用途提取 - Use case extraction
        use_case = []
        if any(k in lower for k in ["游戏", "gaming", "fps"]):
            use_case.append("gaming")
        if any(k in lower for k in ["剪辑", "pr", "video"]):
            use_case.append("video_editing")
        if any(k in lower for k in ["ai", "深度学习"]):
            use_case.append("ai")
        if any(k in lower for k in ["办公", "office"]):
            use_case.append("office")
        if use_case:
            update.use_case = sorted(set(use_case))
        
        # 分辨率提取 - Resolution extraction
        if "2k" in lower or "1440" in lower:
            update.resolution = "1440p"
        elif "4k" in lower:
            update.resolution = "4k"
        elif "1080" in lower:
            update.resolution = "1080p"
        
        # CPU 偏好 - CPU preference
        if "intel" in lower:
            update.cpu_preference = "Intel"
        elif "amd" in lower:
            update.cpu_preference = "AMD"
        
        # 静音需求 - Quiet requirement
        if "静音" in text:
            update.need_quiet = True
        if any(k in text for k in ["不在乎噪音", "噪音无所谓"]):
            update.need_quiet = False
        
        # 存储容量 - Storage capacity
        if "2t" in lower or "2tb" in lower:
            update.storage_target_gb = 2000
        elif "1t" in lower or "1tb" in lower:
            update.storage_target_gb = 1000
        elif "512" in lower:
            update.storage_target_gb = 512
        
        # 优先级 - Priority
        if any(k in text for k in ["便宜点", "省点", "省钱"]):
            update.priority = "budget"
        if any(k in text for k in ["性能高点", "性能优先", "拉满"]):
            update.priority = "performance"
        
        # 计算缺失字段 - Calculate missing fields
        missing = []
        if current.budget_max is None and update.budget_max is None:
            missing.append("budget")
        if current.use_case is None and update.use_case is None:
            missing.append("use_case")
        if current.resolution is None and update.resolution is None:
            missing.append("resolution")
        update.missing_fields = missing
        
        return update


def collect_requirements(state: dict) -> dict:
    """
    需求收集节点入口函数 - Requirement Collection Node Entry Function
    
    此函数将被 graph.py 调用。
    This function will be called by graph.py.
    
    参数 Parameters:
        state: 当前状态字典
               Current state dictionary
    
    返回 Returns:
        更新后的状态字典
        Updated state dictionary
    """
    # 此函数将被 graph.py 调用
    # 返回更新后的状态
    pass
