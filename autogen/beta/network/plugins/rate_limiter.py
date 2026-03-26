# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""RateLimiter — routing plugin that rejects delegations exceeding rate limits."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from ..primitives.envelope import Envelope
from ..topology import BasePlugin, HubContext

if TYPE_CHECKING:
    pass


class RateLimiter(BasePlugin):
    """Rejects delegations that exceed per-sender rate limits.

    Routing plugin — sits in the delegation path. Returns None to reject
    envelopes from senders who have exceeded their rate limit.

    Example::

        hub = Hub(
            topology=Pipeline(RateLimiter(max_per_minute=10)),
        )
    """

    def __init__(self, max_per_minute: int = 60) -> None:
        self._max_per_minute = max_per_minute
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        now = time.monotonic()
        sender = envelope.sender
        window = now - 60.0

        # Clean old timestamps
        recent = [t for t in self._timestamps[sender] if t > window]
        if recent:
            self._timestamps[sender] = recent
        else:
            # Remove sender entry entirely to prevent unbounded dict growth
            self._timestamps.pop(sender, None)

        if len(self._timestamps.get(sender, [])) >= self._max_per_minute:
            return None  # Reject

        self._timestamps[sender].append(now)
        return envelope
