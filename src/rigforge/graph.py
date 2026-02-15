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

_LLM_CALL_LOCK = threading.Lock()
_LAST_LLM_CALL_AT = 0.0
_AUTO_LLM = object()
_PROVIDER_FAIL_UNTIL: Dict[str, float] = {"zhipu": 0.0, "openrouter": 0.0}


_BRAND_CANON = {
    "intel": "Intel",
    "amd": "AMD",
    "nvidia": "NVIDIA",
}


def _canon_brand(value: str) -> str:
    key = value.strip().lower()
    return _BRAND_CANON.get(key, value.strip())


def _canon_brand_list(values: List[str] | None) -> List[str] | None:
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
        print(f"[PERF] Rate limit wait: {wait:.3f}s")
        time.sleep(wait)
    print(f"[PERF] Rate limit overhead: {time.time() - start:.3f}s")
    return invoke_fn()


def invoke_with_turn_timeout(invoke_fn, timeout_seconds: Optional[float] = None):
    start = time.time()
    try:
        result = invoke_fn()
        elapsed = time.time() - start
        print(f"[PERF] LLM call completed in {elapsed:.3f}s")
        return result
    except Exception as err:
        elapsed = time.time() - start
        print(f"[PERF] LLM call failed after {elapsed:.3f}s: {err}")
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
            print(f"[CACHE] 命中缓存，跳过 LLM 调用")
            return self._cache[cache_key]

        start = time.time()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是PC装机需求提取器。从用户输入中提取以下信息：\n"
                    "- 预算范围 (budget_min, budget_max)\n"
                    "- 用途 (use_case): gaming/video_editing/ai/office\n"
                    "- 分辨率 (resolution): 1080p/1440p/4k\n"
                    "- CPU偏好 (cpu_preference): Intel/AMD\n"
                    "- 显卡偏好 (gpu_preference): NVIDIA/AMD/具体型号\n"
                    "- 内存容量 (memory_gb): 数字\n"
                    "- 存储容量 (storage_target_gb): 数字\n"
                    "- 静音需求 (need_quiet): true/false\n"
                    "- 品牌偏好 (prefer_brands)\n"
                    "- 禁用品牌 (brand_blacklist)\n"
                    "- 优先级 (priority): budget/balanced/performance\n"
                    "\n"
                    "重要：如果识别到某个字段，必须将对应的 *_set 字段设为 true！\n"
                    "例如：识别到用途=办公，必须设置 use_case=['office'] 且 use_case_set=true\n"
                    "直接输出JSON，不要markdown标记。",
                ),
                (
                    "human",
                    "已收集信息: {collected}\n用户本轮输入: {text}",
                ),
            ]
        )
        print(f"[PERF] Prompt built in {time.time() - start:.3f}s")
        
        try:
            structured_start = time.time()
            structured = llm.with_structured_output(RequirementUpdate)
            print(f"[PERF] Structured output setup in {time.time() - structured_start:.3f}s")
            
            # 构建增量信息，只发送已收集的信息摘要，而非完整 JSON
            collected_summary = self._build_collected_summary(current)
            
            model_input = {
                "collected": collected_summary,  # 简洁摘要，而非完整 JSON
                "text": text
            }
            print(f"\n{'='*60}")
            print(f"[LLM INPUT #1] 需求提取")
            print(f"  已收集: {collected_summary}")
            print(f"  用户输入: {text}")
            print(f"{'='*60}\n")
            
            invoke_start = time.time()
            result = invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(
                    lambda: (prompt | structured).invoke(
                        model_input
                    )
                )
            )
            
            # 显示模型输出
            print(f"\n{'='*60}")
            print(f"[LLM OUTPUT #1] 需求提取结果:")
            print(f"  {result.model_dump_json()[:500]}...")
            print(f"{'='*60}\n")
            
            # 存入缓存
            self._cache[cache_key] = result
            
            print(f"[PERF] Total extract() call: {time.time() - start:.3f}s")
            return result
        except Exception as e:
            print(f"[PERF] Extract failed after {time.time() - start:.3f}s, falling back to rules: {e}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            print(f"[DEBUG] Exception details: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return self._extract_with_rules(text, current)

    def extract_and_reply(self, text: str, current: UserRequirements, llm=None, follow_up_questions: List[str] = None, enthusiasm_level: str = "standard") -> RequirementUpdateWithReply:
        if llm is None:
            update = self._extract_with_rules(text, current)
            reply = self._generate_fallback_reply(update, follow_up_questions, enthusiasm_level)
            return RequirementUpdateWithReply(requirement_update=update, reply=reply)

        start = time.time()
        
        follow_style, recommend_style = self._enthusiasm_instructions(enthusiasm_level)
        
        follow_up_text = ""
        if follow_up_questions:
            follow_up_text = "\n".join([f"- {q}" for q in follow_up_questions])
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是PC装机需求提取器和回复生成器。"
                    "从用户话术里提取预算、用途、分辨率、品牌偏好、禁用品牌、机箱尺寸、推荐优先级、"
                    "存储容量需求、静音偏好、CPU偏好、游戏名称。"
                    "priority 取值: budget/balanced/performance。"
                    "如果识别到对应信息，请把 *_set 标记为 true。"
                    "missing_fields 仅包含还不明确的关键字段: budget/use_case/resolution/storage/noise。"
                    "重要：直接输出 JSON，不要使用 markdown 代码块，不要包含 ```json 或 ``` 标记。"
                    "只输出 JSON，不要包含任何解释性文字、注释或额外内容。"
                    "保持 JSON 简洁，未识别的字段留空或设为默认值。"
                    "回复要求："
                    "1. 用自然中文与用户互动，语气积极、口语化，不要太长，语言要灵活。"
                    "2. 先对用户本轮给出的配置信息做一句积极肯定。"
                    "3. 如果用户信息明显不妥，再温和指出并给建议。"
                    "4. 然后只提出一个最关键的下一个问题，不要一次问多个问题。"
                    "{follow_style}",
                ),
                (
                    "human",
                    "当前需求: {current}\n用户输入: {text}\n问题列表: {follow_up_text}",
                ),
            ]
        )
        print(f"[PERF] Prompt built in {time.time() - start:.3f}s")
        
        try:
            structured_start = time.time()
            structured = llm.with_structured_output(RequirementUpdateWithReply)
            print(f"[PERF] Structured output setup in {time.time() - structured_start:.3f}s")
            
            invoke_start = time.time()
            result = invoke_with_turn_timeout(
                lambda: invoke_with_rate_limit(
                    lambda: (prompt | structured).invoke(
                        {
                            "current": current.model_dump_json(),
                            "text": text,
                            "follow_up_text": follow_up_text,
                            "follow_style": follow_style,
                        }
                    )
                )
            )
            print(f"[PERF] Total extract_and_reply() call: {time.time() - start:.3f}s")
            return result
        except Exception as e:
            print(f"[PERF] Extract and reply failed after {time.time() - start:.3f}s, falling back to rules: {e}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            print(f"[DEBUG] Exception details: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            update = self._extract_with_rules(text, current)
            reply = self._generate_fallback_reply(update, follow_up_questions, enthusiasm_level)
            return RequirementUpdateWithReply(requirement_update=update, reply=reply)

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
        games = re.findall(r"(cs2|valorant|lol|dota2|apex|pubg|原神|黑神话)", lower)
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
    if payload.get("cpu_preference"):
        payload["cpu_preference"] = _canon_brand(payload["cpu_preference"])
    payload["prefer_brands"] = _canon_brand_list(payload.get("prefer_brands")) or []
    payload["brand_blacklist"] = _canon_brand_list(payload.get("brand_blacklist")) or []
    return UserRequirements.model_validate(payload)


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
        builder.add_edge("generate_follow_up", "compose_reply")
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
        start_time = time.time()
        current = state.get("requirements") or UserRequirements()
        user_input = state["user_input"]
        last_assistant_reply = state.get("last_assistant_reply", "")
        
        # 暂时屏蔽简单判断逻辑，所有回答都优先走大模型
        # use_llm = self._should_use_llm(user_input, last_assistant_reply)
        use_llm = True  # 强制使用 LLM
        
        llm_start = time.time()
        llm = self._runtime_llm(state.get("model_provider", "rules"), 0) if use_llm else None
        print(f"[PERF] LLM setup took {time.time() - llm_start:.3f}s (use_llm={use_llm})")
        
        extract_start = time.time()
        try:
            update = self.extractor.extract(user_input, current, llm=llm)
        except TypeError:
            update = self.extractor.extract(user_input, current)
        print(f"[PERF] Extract took {time.time() - extract_start:.3f}s")
        
        guards_start = time.time()
        update = self._apply_keyword_guards(update, user_input)
        update = self._apply_contextual_short_answer(update, user_input, last_assistant_reply)
        update = self._normalize_update_flags(update)
        print(f"[PERF] Guards and normalization took {time.time() - guards_start:.3f}s")
        
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
        
        merge_start = time.time()
        merged = merge_requirements(current, update)
        print(f"[PERF] Merge requirements took {time.time() - merge_start:.3f}s")
        
        # 诊断日志：显示需求提取和合并结果
        print(f"\n[DEBUG] 需求提取结果:")
        print(f"  update.use_case = {update.use_case}")
        print(f"  update.use_case_set = {update.use_case_set}")
        print(f"  update.budget_set = {update.budget_set}")
        print(f"[DEBUG] 合并后状态:")
        print(f"  merged.use_case = {merged.use_case}")
        print(f"  merged.use_case_set = {merged.use_case_set}")
        print(f"  merged.budget_set = {merged.budget_set}")
        
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

        avoid_repeat_field = None
        if (
            route == "ask_more"
            and self._is_generic_continue(user_input)
            and not self._has_material_update(update)
        ):
            last_field = self._infer_last_question_field(last_assistant_reply)
            if last_field in questions:
                avoid_repeat_field = last_field
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
        if "gaming" in req.use_case and not req.game_titles:
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
        print(f"[PERF] generate_follow_up took {time.time() - start:.3f}s")
        return {"follow_up_questions": questions}

    def recommend_build(self, state: GraphState):
        start = time.time()
        req = state["requirements"]
        context_start = time.time()
        _context = self.tool_map["recommendation_context"].invoke(req.model_dump())
        print(f"[PERF] recommendation_context took {time.time() - context_start:.3f}s")
        
        pick_start = time.time()
        build = pick_build_from_candidates(req, self.tool_map["search_parts"])
        print(f"[PERF] pick_build_from_candidates took {time.time() - pick_start:.3f}s")
        
        fit_start = time.time()
        build = ensure_budget_fit(req, build)
        print(f"[PERF] ensure_budget_fit took {time.time() - fit_start:.3f}s")
        
        print(f"[PERF] recommend_build total took {time.time() - start:.3f}s")
        return {"build": build}

    def validate_build(self, state: GraphState):
        start = time.time()
        build = state["build"]
        skus = []
        for key in ["cpu", "motherboard", "memory", "gpu", "psu", "case", "cooler"]:
            part = getattr(build, key)
            if part:
                skus.append(part.sku)

        if len(skus) < 7:
            print(f"[PERF] validate_build took {time.time() - start:.3f}s (incomplete build)")
            return {
                "compatibility_issues": ["build generation incomplete, please provide a wider budget"],
                "estimated_power": 0,
            }

        compat_start = time.time()
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
        print(f"[PERF] check_compatibility took {time.time() - compat_start:.3f}s")
        
        power_start = time.time()
        estimated_power = self.tool_map["estimate_power"].invoke({"parts": skus})
        print(f"[PERF] estimate_power took {time.time() - power_start:.3f}s")
        
        if build.total_price() > state["requirements"].budget_max:
            issues.append(
                f"Total price {build.total_price()} exceeds budget max {state['requirements'].budget_max}"
            )
        print(f"[PERF] validate_build total took {time.time() - start:.3f}s")
        return {
            "compatibility_issues": issues,
            "estimated_power": estimated_power,
        }

    def compose_reply(self, state: GraphState):
        start = time.time()
        
        # 判断是否需要 LLM 生成追问回复
        user_input = state.get("user_input", "")
        questions = state.get("follow_up_questions", [])
        
        # 暂时屏蔽简单判断逻辑，所有回答都优先走大模型
        use_llm_for_reply = True  # 强制使用 LLM
        
        llm_setup_start = time.time()
        llm = self._runtime_llm(state.get("model_provider", "rules"), 0.2) if use_llm_for_reply else None
        print(f"[PERF] compose_reply LLM setup took {time.time() - llm_setup_start:.3f}s (use_llm={use_llm_for_reply})")
        
        template_history = deepcopy(state.get("template_history", {}))
        print(f"[PERF] compose_reply started")
        level = state.get("enthusiasm_level", "standard")
        effective_level = self._effective_enthusiasm_level(level, state.get("turn_number", 1))
        follow_style, recommend_style = self._enthusiasm_instructions(effective_level)
        if state.get("follow_up_questions"):
            questions = state["follow_up_questions"]
            if llm:
                try:
                    prompt_start = time.time()
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
                    print(f"[PERF] follow_prompt built in {time.time() - prompt_start:.3f}s")
                    
                    # 显示模型输入
                    model_input = {
                        "user_input": state.get("user_input", ""),
                        "questions": "\n".join(f"- {q}" for q in questions),
                        "follow_style": follow_style,
                    }
                    print(f"\n{'='*60}")
                    print(f"[LLM INPUT #2] 追问回复生成")
                    print(f"  用户输入: {model_input['user_input']}")
                    print(f"  问题列表: {model_input['questions']}")
                    print(f"{'='*60}\n")
                    
                    invoke_start = time.time()
                    reply = invoke_with_rate_limit(
                        lambda: invoke_with_turn_timeout(
                            lambda: (follow_prompt | llm).invoke(
                                model_input
                            ).content
                        )
                    )
                    
                    # 显示模型输出
                    print(f"\n{'='*60}")
                    print(f"[LLM OUTPUT #2] 追问回复:")
                    print(f"  {reply[:300]}...")
                    print(f"{'='*60}\n")
                    
                    print(f"[PERF] follow_up LLM invoke took {time.time() - invoke_start:.3f}s")
                    
                    response_mode = "llm"
                    fallback_reason = None
                except Exception as err:
                    print(f"[PERF] follow_up LLM failed, falling back: {err}")
                    fallback_start = time.time()
                    reply, template_history = self._fallback_followup_reply(
                        questions,
                        effective_level,
                        turn_number=state.get("turn_number", 1),
                        user_input=state.get("user_input", ""),
                        template_history=template_history,
                    )
                    print(f"[PERF] follow_up fallback took {time.time() - fallback_start:.3f}s")
                    response_mode = "fallback"
                    fallback_reason = self._classify_fallback_reason(err)
            else:
                fallback_start = time.time()
                reply, template_history = self._fallback_followup_reply(
                    questions,
                    effective_level,
                    turn_number=state.get("turn_number", 1),
                    user_input=state.get("user_input", ""),
                    template_history=template_history,
                )
                print(f"[PERF] follow_up fallback (no LLM) took {time.time() - fallback_start:.3f}s")
                response_mode = "fallback"
                fallback_reason = "no_model_config"
            print(f"[PERF] compose_reply total took {time.time() - start:.3f}s")
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
                print(f"[PERF] recommend_prompt built in {time.time() - prompt_start:.3f}s")
                
                invoke_start = time.time()
                reply = invoke_with_rate_limit(
                    lambda: invoke_with_turn_timeout(
                        lambda: (prompt | llm).invoke(
                            {
                                "req": req.model_dump_json(),
                                "build": build.model_dump_json(),
                                "issues": issues,
                                "price": build.total_price(),
                                "power": state.get("estimated_power", 0),
                                "recommend_style": recommend_style,
                            }
                        ).content
                    )
                )
                print(f"[PERF] recommend LLM invoke took {time.time() - invoke_start:.3f}s")
                
                response_mode = "llm"
                fallback_reason = None
            except Exception as err:
                print(f"[PERF] recommend LLM failed, falling back: {err}")
                fallback_start = time.time()
                reply, template_history = self._fallback_reply(
                    build,
                    issues,
                    perf,
                    effective_level,
                    turn_number=state.get("turn_number", 1),
                    user_input=state.get("user_input", ""),
                    template_history=template_history,
                )
                print(f"[PERF] recommend fallback took {time.time() - fallback_start:.3f}s")
                response_mode = "fallback"
                fallback_reason = self._classify_fallback_reason(err)
        else:
            fallback_start = time.time()
            reply, template_history = self._fallback_reply(
                build,
                issues,
                perf,
                effective_level,
                turn_number=state.get("turn_number", 1),
                user_input=state.get("user_input", ""),
                template_history=template_history,
            )
            print(f"[PERF] recommend fallback (no LLM) took {time.time() - fallback_start:.3f}s")
            response_mode = "fallback"
            fallback_reason = "no_model_config"
        print(f"[PERF] compose_reply total took {time.time() - start:.3f}s")

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
        no_words = {"不要", "不用", "不需要", "否", "不", "no"}
        if text not in yes_words and text not in no_words:
            yes_no_matched = False
        else:
            yes_no_matched = True

        last = (last_assistant_reply or "").lower()
        asks_noise = ("静音" in last) or ("噪音" in last)
        if yes_no_matched:
            if asks_noise:
                update.need_quiet = text in yes_words
                update.noise_set = True

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
