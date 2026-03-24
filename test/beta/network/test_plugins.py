# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.network.plugins.rate_limiter import RateLimiter
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.topology import HubContext


def _make_envelope(sender: str = "agent-a") -> Envelope:
    return Envelope(event=ModelMessage(content="test"), sender=sender)


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self) -> None:
        limiter = RateLimiter(max_per_minute=5)
        ctx = HubContext(hub=None)  # type: ignore

        for _ in range(5):
            result = await limiter.process(_make_envelope(), ctx)
            assert result is not None

    @pytest.mark.asyncio
    async def test_rejects_over_limit(self) -> None:
        limiter = RateLimiter(max_per_minute=3)
        ctx = HubContext(hub=None)  # type: ignore

        # First 3 should pass
        for _ in range(3):
            result = await limiter.process(_make_envelope(), ctx)
            assert result is not None

        # 4th should be rejected
        result = await limiter.process(_make_envelope(), ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_per_sender_limits(self) -> None:
        limiter = RateLimiter(max_per_minute=2)
        ctx = HubContext(hub=None)  # type: ignore

        # Agent A uses 2 slots
        for _ in range(2):
            result = await limiter.process(_make_envelope("agent-a"), ctx)
            assert result is not None

        # Agent A is now over limit
        result = await limiter.process(_make_envelope("agent-a"), ctx)
        assert result is None

        # Agent B still has quota
        result = await limiter.process(_make_envelope("agent-b"), ctx)
        assert result is not None
