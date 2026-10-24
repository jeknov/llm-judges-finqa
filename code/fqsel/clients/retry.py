"""Retries with exponential backoff for external API calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class TransientError(RuntimeError):
    """A failure worth retrying (rate limit, timeout, 5xx)."""


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
        except TransientError:
            if attempt == max_attempts - 1:
                raise
            sleep(min(max_delay_s, base_delay_s * 2**attempt))
    raise AssertionError("unreachable")
