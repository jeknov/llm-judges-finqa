"""Retries with exponential backoff for external API calls, honouring a server's retry-after."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class TransientError(RuntimeError):
    """A failure worth retrying (rate limit, timeout, 5xx)."""


class RateLimited(TransientError):
    """HTTP 429; `retry_after` is the server's retry-after in seconds, if it sent one."""

    def __init__(self, retry_after: float | None = None):
        super().__init__(f"rate limited (retry-after={retry_after})")
        self.retry_after = retry_after


def with_retries(
    fn: Callable[[], T],
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError as e:
            if attempt == max_attempts - 1:
                raise
            delay = min(max_delay_s, base_delay_s * 2**attempt)
            retry_after = getattr(e, "retry_after", None)
            sleep(max(delay, retry_after) if retry_after is not None else delay)
    raise AssertionError("unreachable")
