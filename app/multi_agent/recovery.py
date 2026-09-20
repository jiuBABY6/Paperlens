"""外部依赖错误分类与有界指数退避。"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import httpx


T = TypeVar("T")


def classify_error(error: Exception) -> dict[str, object]:
    """把异常转换为 Trace 可持久化的统一错误契约。"""
    retryable = isinstance(error, (httpx.TimeoutException, httpx.TransportError))
    category = "transport" if retryable else "application"
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        retryable = status_code in {408, 409, 425, 429} or status_code >= 500
        category = "remote_http"
    elif isinstance(error, TimeoutError):
        retryable = True
        category = "timeout"
    return {
        "type": type(error).__name__,
        "category": category,
        "message": str(error)[:500],
        "retryable": retryable,
    }


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """只重试可恢复的网络/限流/服务端错误，次数严格有界。"""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            detail = classify_error(error)
            if not detail["retryable"] or attempt >= max_retries:
                raise
            if base_delay_seconds > 0:
                sleep(base_delay_seconds * (2 ** attempt))
            attempt += 1

