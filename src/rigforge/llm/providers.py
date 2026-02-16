"""LLM 提供商构建与调用管理"""

from __future__ import annotations

import os
import time
import threading
from typing import Literal, Optional

from langchain_openai import ChatOpenAI

# 全局锁与状态
_LLM_CALL_LOCK = threading.Lock()
_LAST_LLM_CALL_AT = 0.0
_PROVIDER_FAIL_UNTIL: dict[str, float] = {"zhipu": 0.0, "openrouter": 0.0}


def build_llm(
    provider: Literal["zhipu", "openrouter", "openai"], 
    temperature: float
) -> Optional[ChatOpenAI]:
    """构建指定提供商的 LLM 实例
    
    Args:
        provider: LLM 提供商 (zhipu/openrouter/openai)
        temperature: 生成温度
        
    Returns:
        ChatOpenAI 实例，如果配置缺失则返回 None
    """
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
    """带速率限制的 LLM 调用
    
    根据 LLM_RATE_LIMIT_ENABLED 环境变量决定是否启用速率限制
    """
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


def invoke_with_turn_timeout(
    invoke_fn, 
    timeout_seconds: Optional[float] = None
):
    """带超时控制的 LLM 调用
    
    Args:
        invoke_fn: 要执行的函数
        timeout_seconds: 超时时间（秒），None 表示不限制
    """
    start = time.time()
    try:
        if timeout_seconds:
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
                elapsed = time.time() - start
                print(f"[PERF] LLM call timed out after {elapsed:.3f}s (timeout={timeout_seconds}s)")
                raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            elapsed = time.time() - start
            return result[0]
        else:
            result = invoke_fn()
            return result
    except Exception as err:
        elapsed = time.time() - start
        print(f"[PERF] LLM call failed after {elapsed:.3f}s: {err}")
        raise


def get_model_name(provider: str) -> str:
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
