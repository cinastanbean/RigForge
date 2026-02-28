"""
RigForge 图引擎 - RigForge Graph Engine

管理聊天对话流程，包括需求收集、配置推荐和回复生成。
Manages chat conversation flow, including requirement collection, build recommendation, and reply generation.

主要功能 Main Functions:
1. 需求提取：从用户输入中提取装机需求
2. 路由决策：判断是继续提问还是推荐配置
3. 配置推荐：根据需求生成硬件配置方案
4. 兼容性检查：验证配置方案的兼容性
5. 回复生成：生成自然语言回复
6. 性能跟踪：记录各环节的性能指标
"""

from __future__ import annotations

import os
import re
import threading
import time
import json
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .schemas import BuildPlan, ChatResponse, RequirementUpdate, RequirementUpdateWithReply, UserRequirements
from .tools import Toolset, pick_build_from_candidates

# 全局锁与状态 - Global Lock and State
_LLM_CALL_LOCK = threading.Lock()
"""
LLM 调用全局锁 - LLM Call Global Lock

用于控制 LLM 调用的并发访问。
Used to control concurrent access to LLM calls.
"""

_LAST_LLM_CALL_AT = 0.0
"""
上次 LLM 调用时间 - Last LLM Call Time

记录上次 LLM 调用的时间戳，用于速率限制。
Record timestamp of last LLM call for rate limiting.
"""

_AUTO_LLM = object()
"""
自动 LLM 选择标记 - Auto LLM Selection Marker

用于标记 LLM 实例应该自动选择提供商。
Marker to indicate LLM instance should auto-select provider.
"""

_PROVIDER_FAIL_UNTIL: Dict[str, float] = {"zhipu": 0.0, "openrouter": 0.0}
"""
提供商失败时间 - Provider Failure Time

记录每个提供商的失败时间，用于熔断机制。
Record failure time for each provider for circuit breaker mechanism.
"""

# 品牌规范化映射 - Brand Canonicalization Mapping
_BRAND_CANON = {
    "intel": "Intel",
    "amd": "AMD",
    "nvidia": "NVIDIA",
}
"""
品牌规范化映射 - Brand Canonicalization Mapping

将品牌名称转换为标准格式。
Convert brand names to standard format.
"""


def _canon_brand(value: str) -> str:
    """
    规范化品牌名称 - Canonicalize Brand Name
    
    将品牌名称转换为标准格式。
    Convert brand name to standard format.
    
    参数 Parameters:
        value: 品牌名称
               Brand name
    
    返回 Returns:
        标准化的品牌名称
        Canonicalized brand name
    """
    key = value.strip().lower()
    return _BRAND_CANON.get(key, value.strip())


def _canon_brand_list(values: List[str] | None) -> List[str] | None:
    """
    规范化品牌列表 - Canonicalize Brand List
    
    将品牌列表转换为标准格式，去重。
    Convert brand list to standard format, remove duplicates.
    
    参数 Parameters:
        values: 品牌列表
                Brand list
    
    返回 Returns:
        标准化的品牌列表
        Canonicalized brand list
    """
    if values is None:
        return None
    out: List[str] = []
    seen = set()
    for raw in values:
        if raw is None:
            continue
        item = _canon_brand(str(raw))
        if not item:
            continue
        token = item.lower()
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    return out


def build_llm(provider: Literal["zhipu", "openrouter", "openai"], temperature: float):
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "0"))

    if provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            return None
        model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openrouter_key,
            base_url=base_url,
            timeout=None,
            max_retries=max_retries,
        )

    if provider == "zhipu":
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if not zhipu_key:
            return None
        model = os.getenv("ZHIPU_MODEL", "glm-4.7-flash")
        base_url = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=zhipu_key,
            base_url=base_url,
            timeout=None,
            max_retries=max_retries,
        )

    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return None
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openai_key,
            timeout=None,
            max_retries=max_retries,
        )

    return None


def invoke_with_rate_limit(invoke_fn):
    global _LAST_LLM_CALL_AT
    
    rate_limit_enabled = os.getenv("LLM_RATE_LIMIT_ENABLED", "false").lower() == "true"
    
    if not rate_limit_enabled:
        return invoke_fn()
    
    min_interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "1.0"))
    start = time.time()
    with _LLM_CALL_LOCK:
        now = time.monotonic()
        target_at = max(now, _LAST_LLM_CALL_AT + min_interval)
        _LAST_LLM_CALL_AT = target_at
    wait = max(0.0, target_at - time.monotonic())
    if wait > 0:
        time.sleep(wait)
    return invoke_fn()


class PerformanceTracker:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.sub_timers: Dict[str, float] = {}
        self.current_timer: Optional[str] = None
        self.current_start: Optional[float] = None
        
    def start(self, timer_name: str):
        if self.current_timer is not None:
            self.sub_timers[self.current_timer] = self.sub_timers.get(self.current_timer, 0) + (time.time() - self.current_start)
        self.current_timer = timer_name
        self.current_start = time.time()
        
    def end(self, timer_name: Optional[str] = None):
        if self.current_timer is not None:
            self.sub_timers[self.current_timer] = self.sub_timers.get(self.current_timer, 0) + (time.time() - self.current_start)
            self.current_timer = None
            self.current_start = None
        
        if timer_name:
            self.start(timer_name)
    
    def finish(self):
        if self.current_timer is not None:
            self.sub_timers[self.current_timer] = self.sub_timers.get(self.current_timer, 0) + (time.time() - self.current_start)
            self.current_timer = None
            self.current_start = None
        return time.time() - self.start_time


def invoke_with_turn_timeout(invoke_fn, timeout_seconds: Optional[float] = None):
    start = time.time()
    try:
        if timeout_seconds:
            import threading
            result = [None]
            exception = [None]
            
            def worker():
                try:
                    result[0] = invoke_fn()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")

            if exception[0]:
                raise exception[0]

            return result[0]
        else:
            result = invoke_fn()
            return result
    except Exception as err:
        raise


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    requirements: UserRequirements
    follow_up_questions: List[str]
    build: BuildPlan
    compatibility_issues: List[str]
    estimated_power: int
    route: Literal["ask_more", "recommend"]
    response_text: str
    enthusiasm_level: Literal["standard", "high"]
    response_mode: Literal["llm", "fallback"]
    fallback_reason: Optional[str]
    high_cooperation: bool
    turn_number: int
    last_assistant_reply: str
    model_provider: Literal["zhipu", "openrouter", "rules"]
    template_history: Dict[str, List[int]]
    avoid_repeat_field: Optional[str]
    interaction_mode: Literal["chat", "component"]


class RequirementExtractor:
    def __init__(self):
        self._cache: Dict[str, RequirementUpdate] = {}
    
    def _cache_key(self, text: str, current: UserRequirements) -> str:
        """生成缓存键"""
        return f"{text}:{current.model_dump_json()}"
    
    @staticmethod
    def _build_collected_summary(req: UserRequirements) -> str:
        """构建已收集信息的简洁摘要，用于发送给 LLM
        
        只包含已明确收集的信息，避免发送完整 JSON，减少 token 消耗
        """
        parts = []
        
        if req.budget_set:
            parts.append(f"预算: {req.budget_min}-{req.budget_max}元")
        
        if req.use_case_set and req.use_case:
            use_case_map = {
                "gaming": "游戏",
                "video_editing": "视频剪辑",
                "ai": "AI开发",
                "office": "办公",
            }
            uses = [use_case_map.get(u, u) for u in req.use_case]
            parts.append(f"用途: {', '.join(uses)}")
        
        if req.resolution_set:
            parts.append(f"分辨率: {req.resolution}")
        
        if req.cpu_set:
            if req.cpu_model:
                parts.append(f"CPU型号: {req.cpu_model}")
            elif req.cpu_preference:
                parts.append(f"CPU偏好: {req.cpu_preference}")
        
        if req.gpu_set:
            if req.gpu_model:
                parts.append(f"显卡型号: {req.gpu_model}")
            elif req.gpu_preference:
                parts.append(f"显卡偏好: {req.gpu_preference}")
        
        if req.memory_set and req.memory_gb:
            mem_info = f"内存: {req.memory_gb}GB"
            if req.memory_type:
                mem_info += f" ({req.memory_type})"
            parts.append(mem_info)
        
        if req.storage_set and req.storage_target_gb:
            parts.append(f"存储: {req.storage_target_gb}GB")
        
        if req.noise_set:
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
        
        if req.case_preference:
            parts.append(f"机箱偏好: {req.case_preference}")
        
        if not parts:
            return "暂无"
        
        return " | ".join(parts)
    
    def extract(self, text: str, current: UserRequirements, llm=None) -> RequirementUpdate:
        if llm is None:
            return self._extract_with_rules(text, current)

        # 检查缓存
        cache_key = self._cache_key(text, current)
        if cache_key in self._cache:
            return self._cache[cache_key]

        start = time.time()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是PC装机需求提取器。从用户输入中提取以下信息：\n"
                    "- 预算范围 (budget_min, budget_max): 整数\n"
                    "- 用途 (use_case): 列表，例如 [\"gaming\"] 或 [\"video_editing\", \"ai\"] 或 [\"office\"]\n"
                    "- 分辨率 (resolution): 字符串，例如 \"1080p\" 或 \"1440p\" 或 \"4k\"\n"
                    "- CPU偏好 (cpu_preference): 字符串，例如 \"Intel\" 或 \"AMD\"\n"
                    "- 显卡偏好 (gpu_preference): 字符串，例如 \"NVIDIA\" 或 \"AMD\" 或具体型号\n"
                    "- 内存容量 (memory_gb): 整数\n"
                    "- 存储容量 (storage_target_gb): 整数\n"
                    "- 静音需求 (need_quiet): 布尔值，true 或 false\n"
                    "- 品牌偏好 (prefer_brands): 列表，例如 [\"Intel\", \"NVIDIA\"]\n"
                    "- 禁用品牌 (brand_blacklist): 列表，例如 [\"某品牌\"]\n"
                    "- 优先级 (priority): 字符串，只能是 \"budget\"、\"balanced\" 或 \"performance\"\n\n"
                    "重要规则：\n"
                    "- 如果识别到某个字段，必须将对应的 *_set 字段设为 true\n"
                    "- 如果用户说\"办公\"，设置 use_case=[\"office\"] 和 use_case_set=true\n"
                    "- 如果用户说\"游戏\"，设置 use_case=[\"gaming\"] 和 use_case_set=true\n"
                    "- 如果用户说\"剪辑\"，设置 use_case=[\"video_editing\"] 和 use_case_set=true\n"
                    "- 如果用户说\"AI\"，设置 use_case=[\"ai\"] 和 use_case_set=true\n"
                    "- 如果用户说\"预算9000\"，设置 budget_min=9000, budget_max=9000, budget_set=true\n"
                    "- 如果用户说\"预算8000-10000\"，设置 budget_min=8000, budget_max=10000, budget_set=true\n"
                    "- 如果没有识别到某个字段，对应的字段设为 null，对应的 *_set 字段也设为 null\n\n"
                    "输出格式：\n"
                    "直接输出纯JSON格式，不要markdown标记，不要包含任何解释性文字。\n"
                    "JSON格式如下：\n"
                    "{{\n"
                    "  \"budget_min\": 整数或null,\n"
                    "  \"budget_max\": 整数或null,\n"
                    "  \"budget_set\": true或false或null,\n"
                    "  \"use_case\": 列表或null,\n"
                    "  \"use_case_set\": true或false或null,\n"
                    "  \"resolution\": 字符串或null,\n"
                    "  \"resolution_set\": true或false或null,\n"
                    "  \"cpu_preference\": 字符串或null,\n"
                    "  \"cpu_set\": true或false或null,\n"
                    "  \"gpu_preference\": 字符串或null,\n"
                    "  \"gpu_set\": true或false或null,\n"
                    "  \"memory_gb\": 整数或null,\n"
                    "  \"memory_set\": true或false或null,\n"
                    "  \"storage_target_gb\": 整数或null,\n"
                    "  \"storage_set\": true或false或null,\n"
                    "  \"need_quiet\": 布尔值或null,\n"
                    "  \"noise_set\": true或false或null,\n"
                    "  \"prefer_brands\": 列表或null,\n"
                    "  \"brand_blacklist\": 列表或null,\n"
                    "  \"priority\": 字符串或null\n"
                    "}}\n\n"
                    "示例：\n"
                    "用户输入：\"预算9000，办公\"\n"
                    "输出：\n"
                    "{{\n"
                    "  \"budget_min\": 9000,\n"
                    "  \"budget_max\": 9000,\n"
                    "  \"budget_set\": true,\n"
                    "  \"use_case\": [\"office\"],\n"
                    "  \"use_case_set\": true,\n"
                    "  \"resolution\": null,\n"
                    "  \"resolution_set\": null,\n"
                    "  \"cpu_preference\": null,\n"
                    "  \"cpu_set\": null,\n"
                    "  \"gpu_preference\": null,\n"
                    "  \"gpu_set\": null,\n"
                    "  \"memory_gb\": null,\n"
                    "  \"memory_set\": null,\n"
                    "  \"storage_target_gb\": null,\n"
                    "  \"storage_set\": null,\n"
                    "  \"need_quiet\": null,\n"
                    "  \"noise_set\": null,\n"
                    "  \"prefer_brands\": null,\n"
                    "  \"brand_blacklist\": null,\n"
                    "  \"priority\": null\n"
                    "}}",
                ),
                (
                    "human",
                    "已收集信息: {collected}\n用户本轮输入: {text}",
                ),
            ]
        )

        try:
            structured = llm.with_structured_output(RequirementUpdate)

            # 构建增量信息，只发送已收集的信息摘要，而非完整 JSON
            collected_summary = self._build_collected_summary(current)

            model_input = {
                "collected": collected_summary,
                "text": text
            }

            result = invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(
                    lambda: (prompt | structured).invoke(
                        model_input
                    )
                )
            )

            # 存入缓存
            self._cache[cache_key] = result

            return result
        except Exception as e:
            return self._extract_with_rules(text, current)

    def extract_and_reply(self, text: str, current: UserRequirements, llm=None, follow_up_questions: List[str] = None, enthusiasm_level: str = "standard", last_assistant_reply: Optional[str] = None) -> RequirementUpdateWithReply:
        if llm is None:
            update = self._extract_with_rules(text, current)
            reply = self._generate_fallback_reply(update, follow_up_questions, enthusiasm_level)
            return RequirementUpdateWithReply(requirement_update=update, reply=reply, should_continue=True)

        start = time.time()
        
        follow_style, recommend_style = self._enthusiasm_instructions(enthusiasm_level)
        
        collected_summary = self._build_collected_summary(current)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是专业的PC装机顾问，负责与用户对话收集需求信息。\n\n"
                    "重要说明：\n"
                    "- 你只负责通过自然语言与用户沟通，收集用户需求\n"
                    "- 你不负责推荐具体的硬件配置\n"
                    "- 硬件配置推荐由独立的算法环节负责，根据收集到的需求自动生成\n\n"
                    "你的任务：\n"
                    "1. 从用户输入中提取装机需求信息\n"
                    "2. 判断是否需要继续提问收集更多信息\n"
                    "3. 如果需要继续，生成自然、友好的回复并提出下一个最关键的问题（必须提出问题！）\n"
                    "4. 如果信息足够或用户拒绝继续，标记 should_continue=false，表示需求收集完成\n\n"
                    "需要收集的关键信息：\n"
                    "- 预算范围 (budget_min, budget_max): 整数\n"
                    "- 用途 (use_case): 列表，例如 [\"gaming\"] 或 [\"video_editing\", \"ai\"] 或 [\"office\"]\n"
                    "- 分辨率 (resolution): 字符串，例如 \"1080p\" 或 \"1440p\" 或 \"4k\"\n"
                    "- 游戏名称 (game_titles): 列表，例如 [\"LOL\", \"DOTA2\"]（当用途包含 gaming 时）\n"
                    "- CPU偏好 (cpu_preference): 字符串，例如 \"Intel\" 或 \"AMD\"\n"
                    "- 显卡偏好 (gpu_preference): 字符串，例如 \"NVIDIA\" 或 \"AMD\" 或具体型号\n"
                    "- 内存容量 (memory_gb): 整数\n"
                    "- 存储容量 (storage_target_gb): 整数\n"
                    "- 静音需求 (need_quiet): 布尔值，true 或 false\n"
                    "- 品牌偏好 (prefer_brands): 列表，例如 [\"Intel\", \"NVIDIA\"]\n"
                    "- 禁用品牌 (brand_blacklist): 列表，例如 [\"某品牌\"]\n"
                    "- 优先级 (priority): 字符串，只能是 \"budget\"、\"balanced\" 或 \"performance\"\n\n"
                    "重要规则：\n"
                    "- 如果识别到某个字段，必须将对应的 *_set 字段设为 true\n"
                    "- 如果用户说\"办公\"，设置 use_case=[\"office\"] 和 use_case_set=true\n"
                    "- 如果用户说\"游戏\"，设置 use_case=[\"gaming\"] 和 use_case_set=true\n"
                    "- 如果用户说\"剪辑\"，设置 use_case=[\"video_editing\"] 和 use_case_set=true\n"
                    "- 如果用户说\"AI\"，设置 use_case=[\"ai\"] 和 use_case_set=true\n"
                    "- 如果用户说\"预算9000\"，设置 budget_min=9000, budget_max=9000, budget_set=true\n"
                    "- 如果用户说\"预算8000-10000\"，设置 budget_min=8000, budget_max=10000, budget_set=true\n"
                    "- 判断是否继续提问：\n"
                    "  * 如果用户说\"不用了\"、\"够了\"、\"就这样\"、\"不用再问\"、\"可以了\"、\"没问题\"、\"行\"、\"OK\"、\"ok\"等表示结束的词，should_continue=false\n"
                    "  * 如果用户说\"开始推荐\"、\"推荐吧\"、\"给我推荐\"、\"随便推荐\"、\"随便给我推荐\"、\"随便给我推荐个吧\"等，should_continue=false\n"
                    "  * 如果用户说\"随便\"、\"都可以\"、\"你看着办\"、\"你决定\"等表示让系统决定，should_continue=false\n"
                    "  * 如果已收集信息中包含预算、用途、分辨率三个核心信息，可以考虑停止提问（should_continue=false）\n"
                    "  * 否则继续提问（should_continue=true）\n"
                    "- 提问策略：每次只问一个最关键的问题，不要一次问多个\n"
                    "- 语气要求：{follow_style}\n"
                    "- 直接输出JSON，不要markdown标记，不要包含任何解释性文字。\n\n"
                    "当 should_continue=false 时：\n"
                    "- 回复应该表示需求收集完成，系统将自动生成推荐配置\n"
                    "- 例如：\"好的，我已经了解了您的需求。系统将根据您的需求自动生成推荐配置。\" 或 \"明白了，需求收集完成，现在为您生成配置方案。\"\n"
                    "- 不要提及具体的硬件配置，只表示需求收集完成\n\n"
                    "当 should_continue=true 时（重要！必须提出问题）：\n"
                    "- 回复必须包含一个最关键的问题，不能只说\"收到\"或\"好的\"等\n"
                    "- 例如：\"好的，预算9000元很明确。请问这台电脑主要用于什么用途呢？比如游戏、办公、视频剪辑还是AI训练？\"\n"
                    "- 例如：\"明白了，办公用途已记录。请问您对显示器分辨率有什么要求吗？比如1080p、2K还是4K？\"\n"
                    "- 注意：如果已收集信息中已经包含某个字段，就不要再问这个问题\n\n"
                    "输出格式：\n"
                    "{{\n"
                    "  \"requirement_update\": {{\n"
                    "    \"use_case\": [\"office\"],\n"
                    "    \"use_case_set\": true,\n"
                    "    \"prefer_brands\": [\"Intel\"],\n"
                    "    \"prefer_brands_set\": true,\n"
                    "    ...\n"
                    "  }},\n"
                    "  \"reply\": \"你的回复内容（当 should_continue=true 时，必须包含问题！）\",\n"
                    "  \"should_continue\": true/false\n"
                    "}}",
                ),
                (
                    "human",
                    "已收集信息: {collected}\n上一轮助手提问: {last_question}\n用户本轮输入: {text}",
                ),
            ]
        )

        try:
            structured = llm.with_structured_output(RequirementUpdateWithReply)

            model_input = {
                "collected": collected_summary,
                "text": text,
                "follow_style": follow_style,
                "last_question": last_assistant_reply or "(这是第一轮对话)",
            }

            result = invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(
                    lambda: (prompt | structured).invoke(model_input)
                ),
                timeout_seconds=60.0
            )

            return result
        except Exception as e:
            update = self._extract_with_rules(text, current)
            reply = self._generate_fallback_reply(update, follow_up_questions, enthusiasm_level)
            return RequirementUpdateWithReply(requirement_update=update, reply=reply, should_continue=True)

    def _generate_fallback_reply(self, update: RequirementUpdate, follow_up_questions: List[str], enthusiasm_level: str) -> str:
        if follow_up_questions:
            return follow_up_questions[0]
        return "收到，我们继续。"

    def _enthusiasm_instructions(self, level: str) -> tuple[str, str]:
        if level == "high":
            return (
                "语气要热情、兴奋，多用感叹号和积极词汇！",
                "语气要热情、兴奋，多用感叹号和积极词汇！"
            )
        return (
            "语气要友好、专业，保持简洁。",
            "语气要友好、专业，保持简洁。"
        )

    def _extract_with_rules(self, text: str, current: UserRequirements) -> RequirementUpdate:
        lower = text.lower()
        update = RequirementUpdate()

        values: List[int] = []
        budget_context_tokens = [
            "预算",
            "价位",
            "价格",
            "花",
            "人民币",
            "rmb",
            "￥",
            "¥",
            "元",
            "块",
        ]
        has_budget_context = any(tok in lower for tok in budget_context_tokens)
        if has_budget_context:
            numbers = re.findall(r"(\d{4,6})", text)
            if numbers:
                values.extend(int(n) for n in numbers)

            wan_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*万", lower)
            if wan_numbers:
                values.extend(int(float(n) * 10000) for n in wan_numbers)

            k_values = []
            k_values.extend(
                re.findall(
                    r"(?:预算|价位|价格|预算在|控制在|大概|约|花|人民币|rmb|￥|¥)\s*([1-9]\d?(?:\.\d+)?)\s*k\b",
                    lower,
                )
            )
            k_values.extend(
                re.findall(
                    r"\b([1-9]\d?(?:\.\d+)?)\s*k\b\s*(?:预算|人民币|rmb|元|块)",
                    lower,
                )
            )
            if k_values:
                values.extend(int(float(n) * 1000) for n in k_values)

        if values:
            vals = sorted(values)
            if len(vals) >= 2:
                update.budget_min = vals[0]
                update.budget_max = vals[-1]
            else:
                update.budget_max = vals[0]
                update.budget_min = int(vals[0] * 0.85)
            update.budget_set = True

        use_case = []
        if any(k in lower for k in ["游戏", "gaming", "fps"]):
            use_case.append("gaming")
        if any(k in lower for k in ["剪辑", "pr", "davinci", "video"]):
            use_case.append("video_editing")
        if any(k in lower for k in ["ai", "深度学习", "模型"]):
            use_case.append("ai")
        if any(k in lower for k in ["办公", "office", "word", "excel"]):
            use_case.append("office")
        if use_case:
            update.use_case = sorted(set(use_case))
            update.use_case_set = True

        if "2k" in lower or "1440" in lower:
            update.resolution = "1440p"
            update.resolution_set = True
        elif "4k" in lower:
            update.resolution = "4k"
            update.resolution_set = True
        elif "1080" in lower:
            update.resolution = "1080p"
            update.resolution_set = True

        brand_blacklist = []
        if "不要" in text and "amd" in lower:
            brand_blacklist.append("AMD")
        if "不要" in text and "intel" in lower:
            brand_blacklist.append("Intel")
        if brand_blacklist:
            update.brand_blacklist = brand_blacklist

        prefer_brands = []
        if "prefer" in lower or "优先" in text:
            if "amd" in lower:
                prefer_brands.append("AMD")
            if "intel" in lower:
                prefer_brands.append("Intel")
            if "nvidia" in lower:
                prefer_brands.append("NVIDIA")
        if prefer_brands:
            update.prefer_brands = prefer_brands

        if "matx" in lower:
            update.case_size = "mATX"
        if "atx" in lower:
            update.case_size = "ATX"

        if "静音" in text:
            update.need_quiet = True
            update.noise_set = True
        if any(k in text for k in ["不在乎噪音", "噪音无所谓", "噪音不敏感"]):
            update.need_quiet = False
            update.noise_set = True
        if "wifi" in lower:
            update.need_wifi = True
        if any(k in lower for k in ["1t", "2t", "512g", "1tb", "2tb"]):
            if "2t" in lower or "2tb" in lower:
                update.storage_target_gb = 2000
            elif "1t" in lower or "1tb" in lower:
                update.storage_target_gb = 1000
            else:
                update.storage_target_gb = 512
            update.storage_set = True
        if any(k in lower for k in ["intel", "amd"]):
            if "intel" in lower:
                update.cpu_preference = "Intel"
            elif "amd" in lower:
                update.cpu_preference = "AMD"
        games = []
        # 固定游戏名称匹配
        fixed_games = re.findall(r"(cs2|valorant|lol|dota2|apex|pubg|原神|黑神话)", lower)
        games.extend(fixed_games)
        
        # 游戏相关关键词匹配
        game_keywords = ["英雄联盟", "守望先锋", "魔兽世界", "星际争霸", "炉石传说", 
                        "王者荣耀", "和平精英", "穿越火线", "使命召唤", "战地",
                        "赛博朋克", "艾尔登法环", "塞尔达", "马里奥", "动物森友会"]
        for game in game_keywords:
            if game in text:
                games.append(game)
        
        # 通用游戏表达匹配
        if any(keyword in lower for keyword in ["游戏", "玩", "电竞", "网游", "单机", "steam", "epic"]):
            # 如果没有具体游戏名称，添加通用游戏标签
            if not games:
                games.append("游戏")
        
        if games:
            update.game_titles = sorted(set(games))
        if any(k in text for k in ["便宜点", "省点", "省钱", "降点预算", "预算太高"]):
            update.priority = "budget"
            update.budget_max = max(6000, int(current.budget_max * 0.9))
            update.budget_min = max(5000, int(update.budget_max * 0.85))
            update.budget_set = True
        if any(k in text for k in ["性能高点", "性能优先", "拉满", "更强"]):
            update.priority = "performance"
        if any(k in text for k in ["均衡", "平衡"]):
            update.priority = "balanced"

        missing = []
        if not (update.budget_set or current.budget_set):
            missing.append("budget")
        if not (update.use_case_set or current.use_case_set):
            missing.append("use_case")
        if not (update.resolution_set or current.resolution_set):
            missing.append("resolution")
        if not (update.storage_set or current.storage_set):
            missing.append("storage")
        if not (update.noise_set or current.noise_set):
            missing.append("noise")
        update.missing_fields = missing
        return update


def merge_requirements(current: UserRequirements, update: RequirementUpdate) -> UserRequirements:
    payload = current.model_dump()
    for key, value in update.model_dump().items():
        if key == "missing_fields":
            continue
        if value is not None:
            payload[key] = value

    # 规范化CPU偏好
    if payload.get("cpu_preference"):
        payload["cpu_preference"] = _canon_brand(payload["cpu_preference"])

    payload["prefer_brands"] = _canon_brand_list(payload.get("prefer_brands")) or []
    payload["brand_blacklist"] = _canon_brand_list(payload.get("brand_blacklist")) or []

    result = UserRequirements.model_validate(payload)
    return result


def ensure_budget_fit(req: UserRequirements, build: BuildPlan) -> BuildPlan:
    # Keep parts realistic and let validation report budget overflow.
    return build


class RigForgeGraph:
    def __init__(self, toolset: Toolset):
        self.tool_map = toolset.register()
        self.build_data_source = toolset.build_data_source
        self.build_data_version = toolset.build_data_version
        self.build_data_mode = toolset.build_data_mode
        self.extractor = RequirementExtractor()
        self.llm = _AUTO_LLM
        self._llm_cache: Dict[tuple[str, float], object] = {}
        self.fallback_templates = self._load_fallback_templates()
        self.graph = self._build_graph()

    @staticmethod
    def _get_model_name(provider: str) -> str:
        """获取模型的具体名称"""
        if provider == "zhipu":
            return os.getenv("ZHIPU_MODEL", "glm-4.7-flash")
        elif provider == "openrouter":
            return os.getenv("OPENROUTER_MODEL", "openrouter/free")
        elif provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-4o")
        elif provider == "rules":
            return "规则模式"
        return provider

    @staticmethod
    def _load_fallback_templates() -> Dict[str, List[str]]:
        root = Path(__file__).resolve().parents[2]
        path = root / "config" / "fallback_templates.json"
        defaults: Dict[str, List[str]] = {
            "no_question": ["你给的信息已经很有帮助了，我先按这些条件给你出一版方案。"],
            "followup_prefix_high": ["太棒了，这个信息非常关键！我们继续："],
            "followup_prefix_standard": ["收到，这条信息很关键。接下来我想确认："],
            "opener_high": ["太棒了，这套配置已经非常接近你的目标："],
            "opener_standard": ["太好了，已经按你的方向整理出一版配置："],
            "closing_high": ["如果你愿意，我可以再给你一套更省钱的备选方案。"],
            "closing_standard": ["如果你愿意，我可以再给你出一套更省钱的备选方案。"],
        }
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            out: Dict[str, List[str]] = {}
            for k, v in defaults.items():
                values = data.get(k)
                if isinstance(values, list) and all(isinstance(x, str) for x in values) and values:
                    out[k] = values
                else:
                    out[k] = v
            return out
        except Exception:
            return defaults

    def _pick_template_from_category(
        self,
        category: str,
        level: Literal["standard", "high"],
        turn_number: int,
        user_input: str,
        template_history: Dict[str, List[int]],
    ) -> tuple[str, Dict[str, List[int]]]:
        options = self.fallback_templates.get(category, [])
        if not options:
            return "", template_history
        used = template_history.get(category, [])
        available = [i for i in range(len(options)) if i not in used]
        if not available:
            used = []
            available = list(range(len(options)))
        seed = f"{category}|{level}|{turn_number}|{user_input}".encode("utf-8")
        pick = available[int(hashlib.md5(seed).hexdigest(), 16) % len(available)]
        template_history.setdefault(category, []).append(pick)
        return options[pick], template_history

    def _get_cached_llm(self, provider: Literal["zhipu", "openrouter", "openai"], temperature: float):
        key = (provider, temperature)
        if key not in self._llm_cache:
            self._llm_cache[key] = build_llm(provider, temperature)
        return self._llm_cache[key]

    def _runtime_llm(
        self, provider: Literal["zhipu", "openrouter", "rules"], temperature: float
    ):
        if self.llm is _AUTO_LLM:
            if provider == "rules":
                return None
            if provider in ("zhipu", "openrouter"):
                return self._get_cached_llm(provider, temperature)
            return None
        if self.llm is None:
            return None
        return self.llm

    def _provider_probe(self, provider: Literal["zhipu", "openrouter"]) -> tuple[bool, str]:
        now = time.monotonic()
        fail_until = _PROVIDER_FAIL_UNTIL.get(provider, 0.0)
        if now < fail_until:
            left = int(max(1, fail_until - now))
            return False, f"冷却中({left}s)"

        llm = self._get_cached_llm(provider, 0)
        if llm is None:
            return False, "未配置密钥"

        probe_timeout = float(os.getenv("LLM_PROVIDER_PROBE_TIMEOUT_SECONDS", "1.5"))
        fail_cooldown = float(os.getenv("LLM_PROVIDER_FAIL_COOLDOWN_SECONDS", "120"))
        try:
            invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(lambda: llm.invoke("ping")),
                timeout_seconds=probe_timeout,
            )
            return True, "可用"
        except Exception as err:
            _PROVIDER_FAIL_UNTIL[provider] = time.monotonic() + fail_cooldown
            reason = self._classify_fallback_reason(err)
            reason_map = {
                "rate_limited": "限流",
                "timeout": "超时",
                "auth_error": "鉴权失败",
                "model_error": "调用失败",
            }
            return False, reason_map.get(reason, "调用失败")

    def select_provider_for_session(self) -> tuple[Literal["zhipu", "openrouter", "rules"], str]:
        # 优先检查配置，避免启动时不必要的 LLM 调用
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        # 如果智谱有配置密钥，直接使用智谱（延迟检测可用性）
        if zhipu_key:
            return "zhipu", "智谱已配置"
        
        # 如果只有 OpenRouter 有配置
        if openrouter_key:
            return "openrouter", "智谱未配置，使用 OpenRouter"
        
        # 都没配置，使用规则模式
        return "rules", "未配置 LLM 密钥，使用规则模式"

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("collect_requirements", self.collect_requirements)
        builder.add_node("generate_follow_up", self.generate_follow_up)
        builder.add_node("recommend_build", self.recommend_build)
        builder.add_node("validate_build", self.validate_build)
        builder.add_node("compose_reply", self.compose_reply)

        builder.set_entry_point("collect_requirements")
        builder.add_conditional_edges(
            "collect_requirements",
            self.route_after_collection,
            {"ask_more": "generate_follow_up", "recommend": "recommend_build"},
        )
        builder.add_conditional_edges(
            "generate_follow_up",
            lambda state: state.get("route", "ask_more"),
            {"recommend": "recommend_build", "ask_more": "compose_reply"},
        )
        builder.add_edge("recommend_build", "validate_build")
        builder.add_edge("validate_build", "compose_reply")
        builder.add_conditional_edges(
            "compose_reply",
            self.route_after_reply,
            {"end": END, "continue": END},
        )

        return builder.compile()

    def route_after_reply(self, state: GraphState):
        if state.get("response_text"):
            return "end"
        return "continue"

    @staticmethod
    def _should_use_llm(user_input: str, last_assistant_reply: str) -> bool:
        """判断是否需要 LLM：简单输入用规则模式即可"""
        text = user_input.strip()
        text_lower = text.lower()
        
        # 简单关键词回复，不需要 LLM
        simple_keywords = {
            "办公", "游戏", "设计", "剪辑", "开发", "ai", "渲染", "直播",
            "是", "否", "好", "好的", "ok", "可以", "行", "对", "是的",
            "继续", "下一步", "不用了", "不需要", "不用", "不用吧",
            "1080p", "2k", "4k", "1440p", "1k",
            "intel", "amd", "nvidia", "n卡", "a卡", "英伟达",
            "要", "要的", "要一个", "加一个", "加上",
            "不要", "没必要", "算了",
            "静音", "安静", "无所谓", "都行", "随便",
            "1t", "2t", "512g", "1tb", "2tb",
        }
        
        # 输入很短（<= 15 字符）且是简单关键词，跳过 LLM
        if len(text) <= 15 and text_lower in simple_keywords:
            return False
        
        # 输入很短且不包含复杂信息，跳过 LLM
        if len(text) <= 8:
            return False
        
        # 纯数字回复（预算），跳过 LLM
        normalized = text.replace("-", "").replace("~", "").replace("到", "").replace("k", "").replace("K", "").replace("万", "000")
        if normalized.isdigit():
            return False
        
        # 预算范围格式 "7000-9000", "7k到9k" 等
        budget_patterns = [
            r"^\d+[kKwW]?[-~到]\d+[kKwW]?$",  # 7k-9k, 7000-9000
            r"^\d+万?[-~到]\d+万?$",  # 1万-2万
        ]
        import re
        if any(re.match(p, text) for p in budget_patterns):
            return False
        
        # 上一个问题很简单，不需要 LLM 分析
        last = last_assistant_reply.lower()
        simple_questions = ["预算", "用途", "分辨率", "要不要", "需要", "静音", "噪音", "存储", "显示器", "品牌", "cpu"]
        if any(q in last for q in simple_questions) and len(text) <= 25:
            return False
        
        # 简单的品牌偏好回复
        brand_keywords = ["华硕", "微星", "技嘉", "七彩虹", "影驰", "索泰", "铭瑄", "耕升", 
                         "海盗船", "金士顿", "芝奇", "威刚", "英睿达", "光威",
                         "三星", "西部数据", "致态", "铠侠"]
        if text_lower in brand_keywords or any(brand in text for brand in brand_keywords) and len(text) <= 20:
            return False
        
        return True

    def collect_requirements(self, state: GraphState):
        tracker = PerformanceTracker("collect_requirements")
        
        current = state.get("requirements") or UserRequirements()
        user_input = state["user_input"]
        last_assistant_reply = state.get("last_assistant_reply", "")
        interaction_mode = state.get("interaction_mode", "chat")
        enthusiasm_level = state.get("enthusiasm_level", "standard")
        
        if interaction_mode == "chat":
            result = self._collect_requirements_chat_mode(state, current, user_input, last_assistant_reply, enthusiasm_level, tracker)
        else:
            result = self._collect_requirements_component_mode(state, current, user_input, last_assistant_reply, tracker)
        
        tracker.finish()
        return result

    def _collect_requirements_chat_mode(self, state: GraphState, current: UserRequirements, user_input: str, last_assistant_reply: str, enthusiasm_level: str, tracker: PerformanceTracker):
        tracker.start("LLM setup")
        llm = self._runtime_llm(state.get("model_provider", "rules"), 0)
        tracker.end()
        
        tracker.start("extract_and_reply")
        result = self.extractor.extract_and_reply(
            user_input, 
            current, 
            llm=llm, 
            enthusiasm_level=enthusiasm_level,
            last_assistant_reply=last_assistant_reply
        )
        tracker.end()
        
        # 应用上下文推断，处理简短回答（如 "10000-12000" 回答预算问题）
        result.requirement_update = self._apply_contextual_short_answer(
            result.requirement_update, user_input, last_assistant_reply
        )
        
        tracker.start("merge_requirements")
        merged = merge_requirements(current, result.requirement_update)
        tracker.end()
        
        tracker.start("route_decision")
        turn_number = state.get("turn_number", 1)
        max_turns = 10
        
        rule_based_route = self._rule_based_route_decision(merged, user_input, turn_number, max_turns)
        llm_based_route = "recommend" if not result.should_continue else "ask_more"
        
        # 安全检查：确保核心信息收集完整
        # 即使 LLM 认为应该结束，如果核心信息不足，也强制继续提问
        core_fields_collected = 0
        if merged.budget_set:
            core_fields_collected += 1
        if merged.use_case_set:
            core_fields_collected += 1
        if merged.resolution_set:
            core_fields_collected += 1
        
        # 核心信息不足时，强制继续提问
        if core_fields_collected < 2:
            route = "ask_more"
        else:
            # LLM 判断优先级更高，只有当 LLM 判断为 "recommend" 时才使用规则引擎的判断
            # 这样可以确保 LLM 的对话意图得到尊重
            route = llm_based_route
            if llm_based_route == "recommend" and rule_based_route == "recommend":
                route = "recommend"
            elif llm_based_route == "recommend" and rule_based_route == "ask_more":
                # LLM 认为应该结束，但规则引擎认为应该继续提问
                # 优先使用 LLM 的判断
                route = "recommend"
            elif llm_based_route == "ask_more" and rule_based_route == "recommend":
                # LLM 认为应该继续提问，但规则引擎认为应该结束
                # 优先使用 LLM 的判断，让对话继续
                route = "ask_more"
            else:
                route = "ask_more"
        
        tracker.end()
        
        return {
            "requirements": merged,
            "follow_up_questions": [],
            "route": route,
            "high_cooperation": True,
            "avoid_repeat_field": None,
            "response_text": result.reply if route == "ask_more" else "",
            "response_mode": "llm" if llm else "fallback",
            "fallback_reason": None,
        }

    def _rule_based_route_decision(self, merged: UserRequirements, user_input: str, turn_number: int, max_turns: int) -> str:
        """规则引擎判断是否应该结束对话"""

        # 1. 检查对话轮数是否超过限制
        if turn_number >= max_turns:
            return "recommend"

        # 2. 检查用户输入是否包含结束对话的关键词
        stop_keywords = [
            "不用了", "够了", "就这样", "不用再问", "可以了", "没问题",
            "行", "OK", "ok", "开始推荐", "推荐吧", "给我推荐",
            "随便推荐", "随便给我推荐", "随便给我推荐个吧", "随便",
            "都可以", "你看着办", "你决定"
        ]
        if any(keyword in user_input for keyword in stop_keywords):
            return "recommend"

        # 3. 检查是否收集到足够的关键信息（预算、用途、分辨率）
        key_fields_collected = 0
        if merged.budget_set:
            key_fields_collected += 1
        if merged.use_case_set:
            key_fields_collected += 1
        if merged.resolution_set:
            key_fields_collected += 1

        # 如果收集到3个关键信息，可以考虑结束对话
        if key_fields_collected >= 3:
            return "recommend"

        # 如果收集到2个关键信息，且对话轮数较多，也可以考虑结束对话
        if key_fields_collected >= 2 and turn_number >= 5:
            return "recommend"

        # 否则继续提问
        return "ask_more"

    def _collect_requirements_component_mode(self, state: GraphState, current: UserRequirements, user_input: str, last_assistant_reply: str, tracker: PerformanceTracker):
        tracker.start("LLM setup")
        llm = self._runtime_llm(state.get("model_provider", "rules"), 0)
        tracker.end()
        
        tracker.start("extract")
        try:
            update = self.extractor.extract(user_input, current, llm=llm)
        except TypeError:
            update = self.extractor.extract(user_input, current)
        tracker.end()
        
        tracker.start("guards_and_normalization")
        update = self._apply_keyword_guards(update, user_input)
        update = self._apply_contextual_short_answer(update, user_input, last_assistant_reply)
        update = self._normalize_update_flags(update)
        tracker.end()
        
        lower = user_input.lower()
        if update.budget_max is None and any(
            k in state["user_input"] for k in ["便宜点", "省点", "省钱", "再便宜", "降一点"]
        ):
            update.budget_max = max(6000, int(current.budget_max * 0.9))
            update.budget_min = max(5000, int(update.budget_max * 0.85))
            update.budget_set = True
            if update.priority is None:
                update.priority = "budget"
        if update.priority is None and any(k in lower for k in ["性能优先", "更强", "拉满"]):
            update.priority = "performance"
        
        tracker.start("merge_requirements")
        merged = merge_requirements(current, update)
        tracker.end()
        
        tracker.start("route_decision")
        stop_followup = self._should_stop_followup(user_input)
        high_cooperation = self._is_high_cooperation(update)

        missing = []
        if not merged.budget_set:
            missing.append("budget")
        if not merged.use_case_set:
            missing.append("use_case")
        if not merged.resolution_set:
            missing.append("resolution")
        if not merged.storage_set:
            missing.append("storage")
        if not merged.noise_set:
            missing.append("noise")

        if stop_followup:
            minimum_missing = self._minimum_missing_for_direct_recommend(merged)
            if minimum_missing:
                route = "ask_more"
                questions = [self._minimum_required_question(minimum_missing[0])]
            else:
                route = "recommend"
                questions = []
        else:
            route = "ask_more" if missing else "recommend"
            questions = missing

        # 安全机制：如果对话轮数超过限制，强制结束对话
        turn_number = state.get("turn_number", 1)
        max_turns = 10
        if route == "ask_more" and turn_number >= max_turns:
            route = "recommend"
            questions = []

        avoid_repeat_field = None
        if (
            route == "ask_more"
            and self._is_generic_continue(user_input)
            and not self._has_material_update(update)
        ):
            last_field = self._infer_last_question_field(last_assistant_reply)
            if last_field in questions:
                avoid_repeat_field = last_field
        
        tracker.end()
        
        return {
            "requirements": merged,
            "follow_up_questions": questions,
            "route": route,
            "high_cooperation": high_cooperation,
            "avoid_repeat_field": avoid_repeat_field,
        }

    def route_after_collection(self, state: GraphState):
        return state["route"]

    def generate_follow_up(self, state: GraphState):
        start = time.time()
        missing = state["follow_up_questions"]
        req = state["requirements"]
        high_cooperation = state.get("high_cooperation", False)
        avoid_repeat_field = state.get("avoid_repeat_field")
        existing_response = state.get("response_text", "")
        user_input = state.get("user_input", "")
        
        # 检查用户是否表示不知道
        dont_know_keywords = ["不知道", "不了解", "不清楚", "没什么", "随便", "都行"]
        user_doesnt_know = any(keyword in user_input for keyword in dont_know_keywords)
        
        # 检查用户输入是否包含游戏相关内容
        game_related_keywords = ["游戏", "玩", "电竞", "网游", "单机", "steam", "epic", 
                               "lol", "dota", "cs", "pubg", "apex", "valorant", 
                               "原神", "黑神话", "守望先锋", "英雄联盟", "使命召唤"]
        has_game_related_content = any(keyword in user_input for keyword in game_related_keywords)
        
        mapping = {
            "budget": "你的预算范围大概是多少呀？例如 7000-9000 或 10000-12000。",
            "use_case": "这台电脑主要做什么呢？游戏、办公、剪辑，还是 AI 开发？",
            "resolution": "你目标分辨率和刷新率是啥？比如 1080p 144Hz、2K 165Hz、4K 60Hz。",
            "cpu_preference": "CPU 你更偏向 Intel 还是 AMD？没有偏好我就按性价比来。",
            "storage": "你对存储有要求吗？比如至少 1TB，或者 2TB 更稳。",
            "noise": "你会在意静音吗？比如希望风扇噪音尽量小。",
        }
        optional_questions: List[str] = []
        ask_cpu_preference = not (req.cpu_preference or "").strip()
        known_missing_keys = [item for item in missing if item in mapping]
        candidate_keys = list(known_missing_keys)
        if ask_cpu_preference and known_missing_keys:
            candidate_keys.append("cpu_preference")
        if "gaming" in req.use_case and not req.game_titles and not user_doesnt_know and not has_game_related_content:
            optional_questions.append("你平时主要玩哪些游戏？我会按目标帧率给你分配显卡和 CPU 预算。")

        ask_limit = 2 if high_cooperation else 1
        ordered_keys = ["budget", "use_case", "resolution", "cpu_preference", "storage", "noise"]
        if avoid_repeat_field in ordered_keys:
            ordered_keys = [k for k in ordered_keys if k != avoid_repeat_field] + [avoid_repeat_field]
        questions: List[str] = []
        for key in ordered_keys:
            if key in candidate_keys:
                questions.append(mapping.get(key, key))
            if len(questions) >= ask_limit:
                break
        if not questions:
            for item in missing:
                if len(questions) >= ask_limit:
                    break
                if item not in mapping:
                    questions.append(item)
        if not questions and optional_questions:
            questions = optional_questions[:ask_limit]
        
        # 如果没有 missing 字段，根据当前需求推断缺失的关键信息
        # 注意：必须检查 *_set 标志，而不是值是否存在，因为有默认值
        # 核心字段：预算、用途、分辨率（必须询问）
        # 可选字段：CPU、显卡、存储、静音（用户可选是否提供）
        if not questions:
            if not req.budget_set:
                questions.append(mapping["budget"])
            elif not req.use_case_set:
                questions.append(mapping["use_case"])
            elif not req.resolution_set:
                questions.append(mapping["resolution"])
            elif not req.cpu_set:
                questions.append("你对 CPU 有偏好吗？比如 Intel 还是 AMD？")
            elif not req.gpu_set:
                questions.append("显卡方面有要求吗？比如 N卡还是 A卡？")
            elif not req.storage_set:
                questions.append("存储容量需要多大？比如 1TB 或 2TB？")
            elif not req.noise_set:
                questions.append("对机箱静音有要求吗？")
            else:
                # 已经有足够信息，应该直接推荐
                questions.append("看来信息已经足够了，我这就为你生成推荐配置！")
        
        # 更新 response_text：如果已有回复是无意义的 fallback，替换为真实问题
        is_placeholder_reply = existing_response in ["收到，我们继续。", "好的", "收到"]
        if is_placeholder_reply and questions:
            response_text = questions[0] if len(questions) == 1 else "、".join(questions)
        else:
            response_text = existing_response

        # 检查是否需要生成推荐配置
        route = "recommend" if "看来信息已经足够了，我这就为你生成推荐配置！" in response_text else "ask_more"

        return {"follow_up_questions": questions, "response_text": response_text, "route": route}

    def recommend_build(self, state: GraphState):
        tracker = PerformanceTracker("recommend_build")
        
        req = state["requirements"]
        
        tracker.start("recommendation_context")
        _context = self.tool_map["recommendation_context"].invoke(req.model_dump())
        tracker.end()
        
        tracker.start("pick_build_from_candidates")
        build = pick_build_from_candidates(req, self.tool_map["search_parts"])
        tracker.end()
        
        tracker.start("ensure_budget_fit")
        build = ensure_budget_fit(req, build)
        tracker.end()
        
        tracker.finish()
        return {"build": build}

    def validate_build(self, state: GraphState):
        tracker = PerformanceTracker("validate_build")
        build = state["build"]
        
        skus = []
        for key in ["cpu", "motherboard", "memory", "gpu", "psu", "case", "cooler"]:
            part = getattr(build, key)
            if part:
                skus.append(part.sku)

        if len(skus) < 7:
            tracker.finish()
            return {
                "compatibility_issues": ["build generation incomplete, please provide a wider budget"],
                "estimated_power": 0,
            }

        tracker.start("check_compatibility")
        issues = self.tool_map["check_compatibility"].invoke(
            {
                "cpu_sku": build.cpu.sku,
                "motherboard_sku": build.motherboard.sku,
                "memory_sku": build.memory.sku,
                "gpu_sku": build.gpu.sku,
                "psu_sku": build.psu.sku,
                "case_sku": build.case.sku,
                "cooler_sku": build.cooler.sku,
            }
        )
        tracker.end()
        
        tracker.start("estimate_power")
        estimated_power = self.tool_map["estimate_power"].invoke({"parts": skus})
        tracker.end()
        
        if build.total_price() > state["requirements"].budget_max:
            issue = f"Total price {build.total_price()} exceeds budget max {state['requirements'].budget_max}"
            issues.append(issue)
        
        tracker.finish()
        return {
            "compatibility_issues": issues,
            "estimated_power": estimated_power,
        }

    def _compose_recommendation_reply(self, state: GraphState, existing_reply: str):
        """生成推荐配置的回复"""
        tracker = PerformanceTracker("_compose_recommendation_reply")
        
        build = state["build"]
        req = state["requirements"]
        issues = state.get("compatibility_issues", [])
        perf = self._estimate_performance_label(req)
        
        level = state.get("enthusiasm_level", "standard")
        effective_level = self._effective_enthusiasm_level(level, state.get("turn_number", 1))
        follow_style, recommend_style = self._enthusiasm_instructions(effective_level)
        
        tracker.start("LLM setup")
        llm = self._runtime_llm(state.get("model_provider", "rules"), 0.2)
        tracker.end()
        
        if llm:
            try:
                tracker.start("prompt building")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "你是热情、靠谱的专业装机顾问。"
                            "用中文输出，简洁、清楚、避免空话。"
                            "{recommend_style}",
                        ),
                        (
                            "human",
                            "需求: {req}\n方案: {build}\n风险: {issues}\n总价: {price}\n功耗: {power}\n"
                            "请输出: 1) 推荐摘要 2) 关键配件理由 3) 风险与替代建议。",
                        ),
                    ]
                )
                tracker.end()
                
                model_input = {
                    "req": req.model_dump_json(),
                    "build": build.model_dump_json(),
                    "issues": issues,
                    "price": build.total_price(),
                    "power": state.get("estimated_power", 0),
                    "recommend_style": recommend_style,
                }

                tracker.start("LLM invoke")
                recommendation_reply = invoke_with_rate_limit(
                    lambda: invoke_with_turn_timeout(
                        lambda: (prompt | llm).invoke(model_input).content,
                        timeout_seconds=120.0
                    )
                )
                tracker.end()

                combined_reply = f"{existing_reply}\n\n{recommendation_reply}"

                tracker.finish()
                return {
                    "response_text": combined_reply,
                    "response_mode": "llm",
                    "fallback_reason": None,
                    "template_history": state.get("template_history", {}),
                }
            except Exception as err:
                fallback_reply, _ = self._fallback_reply(
                    build,
                    issues,
                    perf,
                    effective_level,
                    turn_number=state.get("turn_number", 1),
                    user_input=state.get("user_input", ""),
                    template_history=state.get("template_history", {}),
                )

                combined_reply = f"{existing_reply}\n\n{fallback_reply}"

                return {
                    "response_text": combined_reply,
                    "response_mode": "fallback",
                    "fallback_reason": self._classify_fallback_reason(err),
                    "template_history": state.get("template_history", {}),
                }
        else:
            fallback_reply, _ = self._fallback_reply(
                build,
                issues,
                perf,
                effective_level,
                turn_number=state.get("turn_number", 1),
                user_input=state.get("user_input", ""),
                template_history=state.get("template_history", {}),
            )

            combined_reply = f"{existing_reply}\n\n{fallback_reply}"

            return {
                "response_text": combined_reply,
                "response_mode": "fallback",
                "fallback_reason": "no_model_config",
                "template_history": state.get("template_history", {}),
            }

    def compose_reply(self, state: GraphState):
        interaction_mode = state.get("interaction_mode", "chat")
        route = state.get("route", "ask_more")

        if interaction_mode == "chat":
            existing_reply = state.get("response_text", "")
            if route == "recommend":
                return self._compose_recommendation_reply(state, existing_reply)
            elif existing_reply:
                return {
                    "response_text": existing_reply,
                    "response_mode": state.get("response_mode", "fallback"),
                    "fallback_reason": state.get("fallback_reason"),
                    "template_history": state.get("template_history", {}),
                }

        # 判断是否需要 LLM 生成追问回复
        user_input = state.get("user_input", "")
        questions = state.get("follow_up_questions", [])

        # 暂时屏蔽简单判断逻辑，所有回答都优先走大模型
        use_llm_for_reply = True

        llm = self._runtime_llm(state.get("model_provider", "rules"), 0.2) if use_llm_for_reply else None

        template_history = deepcopy(state.get("template_history", {}))
        level = state.get("enthusiasm_level", "standard")
        effective_level = self._effective_enthusiasm_level(level, state.get("turn_number", 1))
        follow_style, recommend_style = self._enthusiasm_instructions(effective_level)
        if state.get("follow_up_questions"):
            questions = state["follow_up_questions"]
            if llm:
                try:
                    follow_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "你是装机顾问。用简短自然的中文回复，语气积极。"
                                "先肯定用户输入，再提一个问题。{follow_style}",
                            ),
                            ("human", "用户输入: {user_input}\n待问: {questions}"),
                        ]
                    )

                    model_input = {
                        "user_input": state.get("user_input", ""),
                        "questions": "\n".join(f"- {q}" for q in questions),
                        "follow_style": follow_style,
                    }

                    reply = invoke_with_rate_limit(
                        lambda: invoke_with_turn_timeout(
                            lambda: (follow_prompt | llm).invoke(
                                model_input
                            ).content,
                            timeout_seconds=60.0
                        )
                    )

                    response_mode = "llm"
                    fallback_reason = None
                except Exception as err:
                    reply, template_history = self._fallback_followup_reply(
                        questions,
                        effective_level,
                        turn_number=state.get("turn_number", 1),
                        user_input=state.get("user_input", ""),
                        template_history=template_history,
                    )
                    response_mode = "fallback"
                    fallback_reason = self._classify_fallback_reason(err)
            else:
                reply, template_history = self._fallback_followup_reply(
                    questions,
                    effective_level,
                    turn_number=state.get("turn_number", 1),
                    user_input=state.get("user_input", ""),
                    template_history=template_history,
                )
                response_mode = "fallback"
                fallback_reason = "no_model_config"
            return {
                "response_text": reply,
                "response_mode": response_mode,
                "fallback_reason": fallback_reason,
                "template_history": template_history,
                "messages": [AIMessage(content=reply)],
            }

        build = state["build"]
        req = state["requirements"]
        issues = state.get("compatibility_issues", [])
        perf = self._estimate_performance_label(req)

        if llm:
            try:
                prompt_start = time.time()
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "你是热情、靠谱的专业装机顾问。"
                            "先用一句积极反馈提升用户参与感，再给出可执行建议。"
                            "用中文输出，简洁、清楚、避免空话。"
                            "{recommend_style}",
                        ),
                        (
                            "human",
                            "需求: {req}\n方案: {build}\n风险: {issues}\n总价: {price}\n功耗: {power}\n"
                            "请输出: 1) 推荐摘要 2) 关键配件理由 3) 风险与替代建议。",
                        ),
                    ]
                )

                model_input = {
                    "req": req.model_dump_json(),
                    "build": build.model_dump_json(),
                    "issues": issues,
                    "price": build.total_price(),
                    "power": state.get("estimated_power", 0),
                    "recommend_style": recommend_style,
                }

                reply = invoke_with_rate_limit(
                    lambda: invoke_with_turn_timeout(
                        lambda: (prompt | llm).invoke(
                            model_input
                        ).content,
                        timeout_seconds=60.0
                    )
                )

                response_mode = "llm"
                fallback_reason = None
            except Exception as err:
                reply, template_history = self._fallback_reply(
                    build,
                    issues,
                    perf,
                    effective_level,
                    turn_number=state.get("turn_number", 1),
                    user_input=state.get("user_input", ""),
                    template_history=template_history,
                )
                response_mode = "fallback"
                fallback_reason = self._classify_fallback_reason(err)
        else:
            reply, template_history = self._fallback_reply(
                build,
                issues,
                perf,
                effective_level,
                turn_number=state.get("turn_number", 1),
                user_input=state.get("user_input", ""),
                template_history=template_history,
            )
            response_mode = "fallback"
            fallback_reason = "no_model_config"

        return {
            "response_text": reply,
            "response_mode": response_mode,
            "fallback_reason": fallback_reason,
            "template_history": template_history,
            "messages": [AIMessage(content=reply)],
        }

    def invoke(
        self,
        message: str,
        requirements: Optional[UserRequirements] = None,
        enthusiasm_level: Literal["standard", "high"] = "standard",
        turn_number: int = 1,
        last_assistant_reply: str = "",
        model_provider: Literal["zhipu", "openrouter", "rules"] = "rules",
        template_history: Optional[Dict[str, List[int]]] = None,
        interaction_mode: Literal["chat", "component"] = "chat",
    ) -> ChatResponse:
        req = requirements or UserRequirements()
        history = deepcopy(template_history) if template_history is not None else {}
        initial: GraphState = {
            "messages": [
                SystemMessage(
                    content="你是RigForge（锐格锻造坊）装机顾问，风格热情专业，优先提高用户参与感并用简洁问题推进需求收集。"
                ),
                HumanMessage(content=message),
            ],
            "user_input": message,
            "requirements": req,
            "follow_up_questions": [],
            "build": BuildPlan(),
            "compatibility_issues": [],
            "estimated_power": 0,
            "route": "ask_more",
            "response_text": "",
            "enthusiasm_level": enthusiasm_level,
            "response_mode": "fallback",
            "fallback_reason": None,
            "high_cooperation": False,
            "turn_number": turn_number,
            "last_assistant_reply": last_assistant_reply,
            "model_provider": model_provider,
            "template_history": history,
            "avoid_repeat_field": None,
            "interaction_mode": interaction_mode,
        }
        out = self.graph.invoke(initial)
        turn_provider: Literal["zhipu", "openrouter", "rules"] = (
            model_provider if out.get("response_mode", "fallback") == "llm" else "rules"
        )
        return ChatResponse(
            reply=out.get("response_text", ""),
            requirements=out.get("requirements", req),
            build=out.get("build", BuildPlan()),
            compatibility_issues=out.get("compatibility_issues", []),
            estimated_power=out.get("estimated_power", 0),
            estimated_performance=self._estimate_performance_label(out.get("requirements", req)),
            enthusiasm_level=enthusiasm_level,
            response_mode=out.get("response_mode", "fallback"),
            fallback_reason=out.get("fallback_reason"),
            model_name=self._get_model_name(model_provider),
            session_model_provider=model_provider,
            turn_model_provider=turn_provider,
            build_data_source=self.build_data_source,
            build_data_version=self.build_data_version,
            build_data_mode=self.build_data_mode,
            template_history=out.get("template_history", history),
        )

    @staticmethod
    def _estimate_performance_label(req: UserRequirements) -> str:
        if req.resolution == "4k":
            return "4K中高画质可用，建议优先显卡预算"
        if req.resolution == "1440p":
            return "2K高画质主流游戏稳定"
        return "1080p高帧率体验"

    def _fallback_followup_reply(
        self,
        questions: List[str],
        level: Literal["standard", "high"],
        turn_number: int = 1,
        user_input: str = "",
        template_history: Optional[Dict[str, List[int]]] = None,
    ) -> tuple[str, Dict[str, List[int]]]:
        history = deepcopy(template_history or {})
        if not questions:
            text, history = self._pick_template_from_category(
                "no_question", level, turn_number, user_input, history
            )
            return text, history
        question_lines = "\n".join(f"- {q}" for q in questions[:2])
        category = "followup_prefix_high" if level == "high" else "followup_prefix_standard"
        prefix, history = self._pick_template_from_category(
            category, level, turn_number, user_input, history
        )
        return prefix + "\n" + question_lines, history

    @staticmethod
    def _enthusiasm_instructions(level: Literal["standard", "high"]) -> tuple[str, str]:
        if level == "high":
            return (
                "每轮先给一句明显积极反馈，语气更有感染力，尽量结合用户本轮信息做鼓励。",
                "开头先肯定用户目标与偏好，用更强互动语气提升参与感，再进入建议。",
            )
        return (
            "每轮先给一句简短积极反馈即可，不要过度夸张，语气自然。",
            "开头先简短肯定用户目标，再进入建议，语气专业稳重。",
        )

    @staticmethod
    def _effective_enthusiasm_level(
        level: Literal["standard", "high"], turn_number: int
    ) -> Literal["standard", "high"]:
        if level == "high" and turn_number > 2:
            return "standard"
        return level

    def _fallback_reply(
        self,
        build: BuildPlan,
        issues: List[str],
        perf: str,
        level: Literal["standard", "high"],
        turn_number: int = 1,
        user_input: str = "",
        template_history: Optional[Dict[str, List[int]]] = None,
    ) -> tuple[str, Dict[str, List[int]]]:
        history = deepcopy(template_history or {})
        opener_category = "opener_high" if level == "high" else "opener_standard"
        opener, history = self._pick_template_from_category(
            opener_category, level, turn_number, user_input, history
        )
        lines = [
            opener,
            "推荐方案已生成（新机）：",
            f"- CPU: {build.cpu.name if build.cpu else 'N/A'}",
            f"- 主板: {build.motherboard.name if build.motherboard else 'N/A'}",
            f"- 内存: {build.memory.name if build.memory else 'N/A'}",
            f"- 显卡: {build.gpu.name if build.gpu else 'N/A'}",
            f"- SSD: {build.storage.name if build.storage else 'N/A'}",
            f"- 电源: {build.psu.name if build.psu else 'N/A'}",
            f"- 机箱: {build.case.name if build.case else 'N/A'}",
            f"- 散热: {build.cooler.name if build.cooler else 'N/A'}",
            f"- 预估总价: {build.total_price()} 元",
            f"- 性能预估: {perf}",
        ]
        if issues:
            lines.append("- 兼容性风险: " + "; ".join(issues))
        else:
            lines.append("- 兼容性检查: 通过")
        closing_category = "closing_high" if level == "high" else "closing_standard"
        closing, history = self._pick_template_from_category(
            closing_category, level, turn_number, user_input, history
        )
        lines.append(closing)
        return "\n".join(lines), history

    @staticmethod
    def _classify_fallback_reason(err: Exception) -> str:
        name = type(err).__name__.lower()
        msg = str(err).lower()
        if "ratelimit" in name or "rate limit" in msg or "429" in msg:
            return "rate_limited"
        if "auth" in name or "api key" in msg or "unauthorized" in msg:
            return "auth_error"
        if "timeout" in name or "timeout" in msg:
            return "timeout"
        return "model_error"

    @staticmethod
    def _should_stop_followup(text: str) -> bool:
        lower = text.lower()
        keywords = [
            "其他我都不知道",
            "我都不知道",
            "我也不知道",
            "不清楚",
            "你看着配",
            "你来定",
            "你决定",
            "直接推荐",
            "先这样",
            "不用再问",
            "别再问了",
            "按默认",
        ]
        return any(k in lower or k in text for k in keywords)

    @staticmethod
    def _is_high_cooperation(update: RequirementUpdate) -> bool:
        signal_fields = [
            update.budget_set,
            update.use_case_set,
            update.resolution_set,
            update.storage_set,
            update.noise_set,
            update.cpu_set,
            update.gpu_set,
            update.memory_set,
            bool(update.cpu_preference),
            bool(update.cpu_model),
            bool(update.gpu_preference),
            bool(update.gpu_model),
            bool(update.game_titles),
            bool(update.prefer_brands),
            bool(update.brand_blacklist),
        ]
        return sum(1 for x in signal_fields if x) >= 2

    @staticmethod
    def _minimum_missing_for_direct_recommend(req: UserRequirements) -> List[str]:
        missing = []
        if not req.budget_set:
            missing.append("budget")
        if not req.use_case_set:
            missing.append("use_case")
        return missing

    @staticmethod
    def _minimum_required_question(field_name: str) -> str:
        mapping = {
            "budget": "可以直接给你方案。先告诉我一个预算范围，比如 7000-9000。",
            "use_case": "可以直接给你方案。再告诉我主要用途：游戏、办公、剪辑还是 AI？",
        }
        return mapping.get(field_name, "可以直接给你方案。请先补充一个最关键的需求。")

    @staticmethod
    def _normalize_update_flags(update: RequirementUpdate) -> RequirementUpdate:
        # 核心需求标志
        if update.budget_set is None and (update.budget_min is not None or update.budget_max is not None):
            update.budget_set = True
        if update.use_case_set is None and update.use_case:
            update.use_case_set = True
        if update.resolution_set is None and update.resolution is not None:
            update.resolution_set = True
        if update.storage_set is None and update.storage_target_gb is not None:
            update.storage_set = True
        if update.noise_set is None and update.need_quiet is not None:
            update.noise_set = True
        
        # 新增配件偏好标志
        if update.cpu_set is None and (update.cpu_preference or update.cpu_model):
            update.cpu_set = True
        if update.gpu_set is None and (update.gpu_preference or update.gpu_model):
            update.gpu_set = True
        if update.memory_set is None and update.memory_gb is not None:
            update.memory_set = True
        
        return update

    @staticmethod
    def _apply_keyword_guards(update: RequirementUpdate, text: str) -> RequirementUpdate:
        lower = text.lower()
        if "intel" in lower:
            update.cpu_preference = "Intel"
        elif "amd" in lower:
            update.cpu_preference = "AMD"
        if "静音" in text:
            update.need_quiet = True
        if any(k in text for k in ["不在乎噪音", "噪音无所谓", "噪音不敏感"]):
            update.need_quiet = False

        if update.storage_target_gb is None:
            if "2t" in lower or "2tb" in lower:
                update.storage_target_gb = 2000
            elif "1t" in lower or "1tb" in lower:
                update.storage_target_gb = 1000
            elif "512g" in lower:
                update.storage_target_gb = 512
        return update

    @staticmethod
    def _apply_contextual_short_answer(
        update: RequirementUpdate, user_input: str, last_assistant_reply: str
    ) -> RequirementUpdate:
        raw_text = user_input.strip().lower()
        if len(raw_text) > 24:
            return update

        text = re.sub(r"[^\w\u4e00-\u9fff]+", "", raw_text)
        yes_words = {
            "要",
            "是",
            "需要",
            "好",
            "好的",
            "可以",
            "可以的",
            "行",
            "行的",
            "要的",
            "对",
            "嗯",
            "嗯嗯",
            "ok",
            "okay",
        }
        no_words = {"不要", "不用", "不需要", "否", "不", "no", "没有", "没", "无", "没要求", "没有要求", "无要求"}
        
        # 检查是否为肯定/否定回答
        # 1. 完全匹配
        if text in yes_words:
            yes_no_matched = True
            is_yes = True
            is_no = False
        elif text in no_words:
            yes_no_matched = True
            is_yes = False
            is_no = True
        # 2. 包含否定词（处理"没有要求"、"没要求"等情况）
        elif any(word in text for word in no_words if len(word) >= 2):
            yes_no_matched = True
            is_yes = False
            is_no = True
        else:
            yes_no_matched = False
            is_yes = False
            is_no = False

        last = (last_assistant_reply or "").lower()
        asks_noise = ("静音" in last) or ("噪音" in last)
        if yes_no_matched and asks_noise:
            update.need_quiet = is_yes
            update.noise_set = True

        # CPU 偏好推断
        asks_cpu = "cpu" in last or "intel" in last or "amd" in last
        if asks_cpu:
            text_lower = text.lower()
            if "intel" in text_lower or "i5" in text_lower or "i7" in text_lower or "i9" in text_lower:
                update.cpu_preference = "Intel"
                update.cpu_set = True
            elif "amd" in text_lower or "r5" in text_lower or "r7" in text_lower or "r9" in text_lower or "ryzen" in text_lower:
                update.cpu_preference = "AMD"
                update.cpu_set = True
            elif text_lower in ["无所谓", "都行", "随便", "都可以", "没要求", "没有"]:
                update.cpu_preference = ""
                update.cpu_set = True  # 标记为已回答（无偏好）

        # 显卡偏好推断
        asks_gpu = "显卡" in last or "n卡" in last or "a卡" in last or "nvidia" in last or "gpu" in last
        if asks_gpu:
            text_lower = text.lower()
            if "n" in text_lower or "nvidia" in text_lower or "rtx" in text_lower or "gtx" in text_lower:
                update.gpu_preference = "NVIDIA"
                update.gpu_set = True
            elif "a" in text_lower and ("卡" in last or "显卡" in last or "amd" in text_lower or "rx" in text_lower or "radeon" in text_lower):
                update.gpu_preference = "AMD"
                update.gpu_set = True
            elif text_lower in ["无所谓", "都行", "随便", "都可以", "没要求", "没有"]:
                update.gpu_preference = ""
                update.gpu_set = True

        # 存储容量推断
        asks_storage = "存储" in last or "硬盘" in last or "容量" in last or "tb" in last
        if asks_storage:
            text_lower = text.lower()
            # 匹配数字+TB/T格式，如 "1TB", "2T", "1t"
            tb_match = re.search(r"(\d+)\s*[tT][bB]?", text_lower)
            if not tb_match:
                tb_match = re.search(r"(\d+)\s*t\b", text_lower)
            if tb_match:
                gb = int(tb_match.group(1)) * 1000
                update.storage_target_gb = gb
                update.storage_set = True
            elif text_lower in ["无所谓", "都行", "随便", "都可以", "没要求", "没有"]:
                update.storage_target_gb = 0
                update.storage_set = True

        asks_budget = "预算" in last or "价位" in last or "多少钱" in last
        if asks_budget:
            # Support concise budget replies: "9000", "9k", "9000-10000", "9000到10000".
            normalized = raw_text.replace(" ", "")
            values: List[int] = []

            range_match = re.findall(r"(\d{3,6})\s*[-~到]\s*(\d{3,6})", normalized)
            for lo, hi in range_match:
                values.extend([int(lo), int(hi)])

            if not values:
                k_match = re.findall(r"\b([1-9]\d?(?:\.\d+)?)k\b", normalized)
                if k_match:
                    values.extend(int(float(x) * 1000) for x in k_match)

            if not values:
                num_match = re.findall(r"\b(\d{3,6})\b", normalized)
                if num_match:
                    values.extend(int(x) for x in num_match)

            if values:
                vals = sorted(values)
                if len(vals) >= 2:
                    update.budget_min = vals[0]
                    update.budget_max = vals[-1]
                else:
                    update.budget_max = vals[0]
                    update.budget_min = int(vals[0] * 0.85)
                update.budget_set = True
        return update

    @staticmethod
    def _is_generic_continue(text: str) -> bool:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.strip().lower())
        if not normalized:
            return False
        tokens = {
            "好",
            "好的",
            "ok",
            "okay",
            "继续",
            "继续吧",
            "继续说",
            "继续问",
            "下一步",
            "往下",
        }
        return normalized in tokens

    @staticmethod
    def _has_material_update(update: RequirementUpdate) -> bool:
        return any(
            [
                update.budget_set is True,
                update.use_case_set is True,
                update.resolution_set is True,
                update.storage_set is True,
                update.noise_set is True,
                update.priority is not None,
                update.cpu_preference is not None,
                bool(update.game_titles),
                bool(update.prefer_brands),
                bool(update.brand_blacklist),
            ]
        )

    @staticmethod
    def _infer_last_question_field(last_assistant_reply: str) -> Optional[str]:
        last = (last_assistant_reply or "").lower()
        if not last:
            return None
        if "预算" in last:
            return "budget"
        if "主要做什么" in last or "游戏、办公、剪辑" in last or "ai 开发" in last:
            return "use_case"
        if "分辨率" in last or "刷新率" in last:
            return "resolution"
        if "intel" in last or "amd" in last or "cpu" in last:
            return "cpu_preference"
        if "显示器" in last:
            return "monitor"
        if "存储" in last or "1tb" in last or "2tb" in last:
            return "storage"
        if "静音" in last or "噪音" in last:
            return "noise"
        return None
