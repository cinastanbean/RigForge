"""
LLM 提供商构建与调用管理 - LLM Provider Building and Invocation Management

管理 LLM 实例的构建、速率限制和超时控制。
Manage LLM instance building, rate limiting, and timeout control.
"""

from __future__ import annotations

import os
import time
import threading
from typing import Literal, Optional

from langchain_openai import ChatOpenAI

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
Record the timestamp of the last LLM call for rate limiting.
"""

_PROVIDER_FAIL_UNTIL: dict[str, float] = {"zhipu": 0.0, "openrouter": 0.0}
"""
提供商失败时间 - Provider Failure Time

记录每个提供商的失败时间，用于熔断机制。
Record failure time for each provider for circuit breaker mechanism.
"""


def build_llm(
    provider: Literal["zhipu", "openrouter", "openai"], 
    temperature: float
) -> Optional[ChatOpenAI]:
    """
    构建指定提供商的 LLM 实例 - Build LLM Instance for Specified Provider
    
    根据提供商类型和环境变量配置，创建 LLM 实例。
    Create LLM instance based on provider type and environment variable configuration.
    
    参数 Parameters:
        provider: LLM 提供商类型
                  LLM provider type
        temperature: 生成温度（0.0-1.0）
                    Generation temperature (0.0-1.0)
    
    返回 Returns:
        ChatOpenAI 实例，如果配置缺失则返回 None
        ChatOpenAI instance, or None if configuration is missing
    """
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "0"))
    """
    最大重试次数 - Maximum Retries
    
    从环境变量 LLM_MAX_RETRIES 读取，默认为 0。
    Read from environment variable LLM_MAX_RETRIES, default is 0.
    """

    # 构建 OpenRouter LLM 实例 - Build OpenRouter LLM instance
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

    # 构建 Zhipu AI LLM 实例 - Build Zhipu AI LLM instance
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

    # 构建 OpenAI LLM 实例 - Build OpenAI LLM instance
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
    """
    带速率限制的 LLM 调用 - LLM Invocation with Rate Limiting
    
    根据 LLM_RATE_LIMIT_ENABLED 环境变量决定是否启用速率限制。
    Decide whether to enable rate limiting based on LLM_RATE_LIMIT_ENABLED environment variable.
    
    参数 Parameters:
        invoke_fn: 要执行的 LLM 调用函数
                   LLM invocation function to execute
    
    返回 Returns:
        LLM 调用结果
        LLM invocation result
    """
    global _LAST_LLM_CALL_AT
    
    # 检查是否启用速率限制 - Check if rate limiting is enabled
    rate_limit_enabled = os.getenv("LLM_RATE_LIMIT_ENABLED", "false").lower() == "true"
    
    # 如果未启用速率限制，直接执行 - If rate limiting is disabled, execute directly
    if not rate_limit_enabled:
        return invoke_fn()
    
    # 获取最小调用间隔 - Get minimum call interval
    min_interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "1.0"))
    start = time.time()
    
    # 使用锁控制并发 - Use lock to control concurrency
    with _LLM_CALL_LOCK:
        now = time.monotonic()
        target_at = max(now, _LAST_LLM_CALL_AT + min_interval)
        _LAST_LLM_CALL_AT = target_at
    
    # 等待达到目标时间 - Wait until target time is reached
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
    """
    带超时控制的 LLM 调用 - LLM Invocation with Timeout Control
    
    在指定的超时时间内执行 LLM 调用，超时则抛出异常。
    Execute LLM call within specified timeout, raise exception if timeout.
    
    参数 Parameters:
        invoke_fn: 要执行的函数
                   Function to execute
        timeout_seconds: 超时时间（秒），None 表示不限制
                         Timeout in seconds, None means no limit
    
    返回 Returns:
        函数执行结果
        Function execution result
    
    异常 Raises:
        TimeoutError: 如果调用超时
                      If call times out
    """
    start = time.time()
    try:
        if timeout_seconds:
            # 使用线程实现超时控制 - Use thread for timeout control
            result = [None]
            exception = [None]
            
            def worker():
                """
                工作线程函数 - Worker Thread Function
                
                在单独的线程中执行 LLM 调用。
                Execute LLM call in a separate thread.
                """
                try:
                    result[0] = invoke_fn()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            # 检查是否超时 - Check if timeout
            if thread.is_alive():
                elapsed = time.time() - start
                print(f"[PERF] LLM call timed out after {elapsed:.3f}s (timeout={timeout_seconds}s)")
                raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")
            
            # 检查是否有异常 - Check if there is exception
            if exception[0]:
                raise exception[0]
            
            elapsed = time.time() - start
            return result[0]
        else:
            # 无超时限制，直接执行 - No timeout limit, execute directly
            result = invoke_fn()
            return result
    except Exception as err:
        elapsed = time.time() - start
        print(f"[PERF] LLM call failed after {elapsed:.3f}s: {err}")
        raise


def get_model_name(provider: str) -> str:
    """
    获取模型的具体名称 - Get Specific Model Name
    
    根据提供商类型获取模型的具体名称。
    Get specific model name based on provider type.
    
    参数 Parameters:
        provider: 提供商类型
                 Provider type
    
    返回 Returns:
        模型名称
        Model name
    """
    if provider == "zhipu":
        return os.getenv("ZHIPU_MODEL", "glm-4.7-flash")
    elif provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "openrouter/free")
    elif provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    elif provider == "rules":
        return "规则模式"
    return provider
