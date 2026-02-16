"""需求提取节点"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from ..data.models import UserRequirements, RequirementUpdate


class RequirementExtractor:
    """需求提取器"""
    
    def __init__(self):
        self._cache: Dict[str, RequirementUpdate] = {}
    
    def _cache_key(self, text: str, current: "UserRequirements") -> str:
        return f"{text}:{current.model_dump_json()}"
    
    @staticmethod
    def build_collected_summary(req: "UserRequirements") -> str:
        """构建已收集信息的简洁摘要"""
        parts = []
        
        if req.budget_max is not None:
            budget_min = req.budget_min or req.budget_max
            parts.append(f"预算: {budget_min}-{req.budget_max}元")
        
        if req.use_case:
            use_case_map = {
                "gaming": "游戏",
                "video_editing": "视频剪辑",
                "ai": "AI开发",
                "office": "办公",
            }
            uses = [use_case_map.get(u, u) for u in req.use_case]
            parts.append(f"用途: {', '.join(uses)}")
        
        if req.resolution:
            parts.append(f"分辨率: {req.resolution}")
        
        if req.cpu_model:
            parts.append(f"CPU型号: {req.cpu_model}")
        elif req.cpu_preference:
            parts.append(f"CPU偏好: {req.cpu_preference}")
        
        if req.gpu_model:
            parts.append(f"显卡型号: {req.gpu_model}")
        elif req.gpu_preference:
            parts.append(f"显卡偏好: {req.gpu_preference}")
        
        if req.memory_gb:
            mem_info = f"内存: {req.memory_gb}GB"
            if req.memory_type:
                mem_info += f" ({req.memory_type})"
            parts.append(mem_info)
        
        if req.storage_target_gb:
            parts.append(f"存储: {req.storage_target_gb}GB")
        
        if req.need_quiet is not None:
            parts.append(f"静音: {'需要' if req.need_quiet else '不需要'}")
        
        if req.prefer_brands:
            parts.append(f"品牌偏好: {', '.join(req.prefer_brands)}")
        
        if req.brand_blacklist:
            parts.append(f"禁用品牌: {', '.join(req.brand_blacklist)}")
        
        if req.game_titles:
            parts.append(f"游戏: {', '.join(req.game_titles)}")
        
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
        """提取需求"""
        from ..llm.providers import invoke_with_rate_limit, invoke_with_turn_timeout
        from ..data.models import RequirementUpdate
        
        if llm is None:
            return self._extract_with_rules(text, current)
        
        # 检查缓存
        cache_key = self._cache_key(text, current)
        if cache_key in self._cache:
            print("[CACHE] 命中缓存，跳过 LLM 调用")
            return self._cache[cache_key]
        
        from ..llm.prompts import REQUIREMENT_EXTRACTION_PROMPT
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", REQUIREMENT_EXTRACTION_PROMPT),
            ("human", "已收集信息: {collected}\n用户本轮输入: {text}"),
        ])
        
        try:
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
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"[PERF] Extract failed, falling back to rules: {e}")
            return self._extract_with_rules(text, current)
    
    def _extract_with_rules(
        self, 
        text: str, 
        current: "UserRequirements"
    ) -> "RequirementUpdate":
        """基于规则的需求提取（回退方案）"""
        from ..data.models import RequirementUpdate
        
        lower = text.lower()
        update = RequirementUpdate()
        
        # 预算提取
        values = []
        budget_tokens = ["预算", "价位", "价格", "花", "元", "块", "rmb", "￥", "¥"]
        if any(tok in lower for tok in budget_tokens):
            numbers = re.findall(r"(\d{4,6})", text)
            values.extend(int(n) for n in numbers)
            
            # 处理 X k 格式
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
        
        # 用途提取
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
        
        # 分辨率提取
        if "2k" in lower or "1440" in lower:
            update.resolution = "1440p"
        elif "4k" in lower:
            update.resolution = "4k"
        elif "1080" in lower:
            update.resolution = "1080p"
        
        # CPU 偏好
        if "intel" in lower:
            update.cpu_preference = "Intel"
        elif "amd" in lower:
            update.cpu_preference = "AMD"
        
        # 静音需求
        if "静音" in text:
            update.need_quiet = True
        if any(k in text for k in ["不在乎噪音", "噪音无所谓"]):
            update.need_quiet = False
        
        # 存储容量
        if "2t" in lower or "2tb" in lower:
            update.storage_target_gb = 2000
        elif "1t" in lower or "1tb" in lower:
            update.storage_target_gb = 1000
        elif "512" in lower:
            update.storage_target_gb = 512
        
        # 优先级
        if any(k in text for k in ["便宜点", "省点", "省钱"]):
            update.priority = "budget"
        if any(k in text for k in ["性能高点", "性能优先", "拉满"]):
            update.priority = "performance"
        
        # 计算缺失字段
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
    """需求收集节点入口函数"""
    # 此函数将被 graph.py 调用
    # 返回更新后的状态
    pass
