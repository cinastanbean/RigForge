import os
import time
from typing import Callable
from functools import wraps

_last_llm_call_at = 0.0
_llm_call_lock = None


def setup_test_rate_limit():
    global _llm_call_lock
    import threading
    _llm_call_lock = threading.Lock()
    
    os.environ["LLM_RATE_LIMIT_ENABLED"] = "true"
    os.environ["LLM_MIN_INTERVAL_SECONDS"] = "1.25"


def test_rate_limit(invoke_fn: Callable):
    global _last_llm_call_at, _llm_call_lock
    
    if _llm_call_lock is None:
        setup_test_rate_limit()
    
    min_interval = 1.25
    with _llm_call_lock:
        now = time.monotonic()
        target_at = max(now, _last_llm_call_at + min_interval)
        _last_llm_call_at = target_at
    
    wait = max(0.0, target_at - time.monotonic())
    if wait > 0:
        time.sleep(wait)
    
    return invoke_fn()
