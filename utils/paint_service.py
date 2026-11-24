"""Shared drawing entry point and request rate limiter."""

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from utils.providers.openai.images import ImageOptionError, ImageOptions, paint

__all__ = ["ImageOptionError", "ImageOptions", "PaintRateLimitResult", "PaintRateLimiter", "paint"]


@dataclass(frozen=True)
class PaintRateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


class PaintRateLimiter:
    def __init__(self):
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(
        self,
        user_id: str,
        *,
        now: float,
        max_requests: int,
        window_seconds: int,
    ) -> PaintRateLimitResult:
        if max_requests <= 0 or window_seconds <= 0:
            return PaintRateLimitResult(True, 0)

        requests = self._requests[user_id]
        cutoff = now - window_seconds
        while requests and requests[0] <= cutoff:
            requests.popleft()

        if len(requests) >= max_requests:
            retry_after = max(1, math.ceil(window_seconds - (now - requests[0])))
            return PaintRateLimitResult(False, retry_after)

        requests.append(now)
        return PaintRateLimitResult(True, 0)
