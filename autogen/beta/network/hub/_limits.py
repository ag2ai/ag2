# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Token-bucket rate limiter backing ``rule.limits.rate``.

Implementation is the standard leaky/refilling bucket: every time the
sender tries to post, we lazily refill the bucket based on the elapsed
wall-clock time since the last draw, clamp to ``burst`` capacity, and
decrement by 1. If the bucket is empty after refill, the draw fails.

The bucket is reset whenever the hub observes a new ``RateBlock``
configuration — Phase 5 rule edits via ``rule_changed`` will plug into
that reset path. Phase 2 keeps buckets purely in memory (they do not
survive a process restart); Phase 3's hydrate cycle leaves rate state
at rest because a restart implicitly drops all pending envelopes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..rule import RateBlock


@dataclass(slots=True)
class _TokenBucket:
    capacity: int
    refill_per_second: float
    tokens: float
    last_refill: float

    @classmethod
    def from_rate(cls, rate: RateBlock, *, clock: float | None = None) -> _TokenBucket:
        now = clock if clock is not None else time.monotonic()
        capacity = max(1, rate.effective_burst())
        return cls(
            capacity=capacity,
            refill_per_second=rate.per_minute / 60.0,
            tokens=float(capacity),
            last_refill=now,
        )

    def consume(self, *, clock: float | None = None) -> bool:
        now = clock if clock is not None else time.monotonic()
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(
            float(self.capacity),
            self.tokens + elapsed * self.refill_per_second,
        )
        self.last_refill = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class RateLimiter:
    """Per-actor token-bucket store.

    Buckets are created lazily on the first call to
    :meth:`check_and_consume` and rebuilt when the actor's
    ``RateBlock`` changes (new capacity or refill rate).
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _TokenBucket] = {}
        self._installed: dict[str, tuple[int, int]] = {}

    def check_and_consume(
        self,
        actor_id: str,
        rate: RateBlock,
        *,
        clock: float | None = None,
    ) -> bool:
        """Return True if the post is allowed, False if throttled."""

        if not rate.is_enabled():
            # No bucket, not rate-limited.
            self._buckets.pop(actor_id, None)
            self._installed.pop(actor_id, None)
            return True
        rate_key = (rate.per_minute, rate.effective_burst())
        installed_key = self._installed.get(actor_id)
        if installed_key != rate_key:
            self._buckets[actor_id] = _TokenBucket.from_rate(rate, clock=clock)
            self._installed[actor_id] = rate_key
        bucket = self._buckets[actor_id]
        return bucket.consume(clock=clock)

    def reset(self, actor_id: str) -> None:
        self._buckets.pop(actor_id, None)
        self._installed.pop(actor_id, None)
