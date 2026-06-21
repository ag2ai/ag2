# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import ConversationContext
from autogen.beta.events import Usage, UsageEvent
from autogen.beta.live import LiveAgent
from autogen.beta.stream import MemoryStream
from autogen.beta.usage import UsageReport


@pytest.mark.asyncio
class TestUsageReportSurface:
    async def test_aggregates_session_history(self) -> None:
        """``LiveAgent.usage_report`` sums the ``UsageEvent`` events emitted
        onto the session stream over the course of a live conversation."""
        ctx = ConversationContext(stream=MemoryStream())

        await ctx.send(
            UsageEvent(
                Usage(prompt_tokens=100, completion_tokens=40),
                kind="model_call",
                model="gpt-realtime",
                provider="openai",
            )
        )
        await ctx.send(
            UsageEvent(
                Usage(prompt_tokens=20, completion_tokens=8),
                kind="model_call",
                model="gpt-realtime",
                provider="openai",
            )
        )

        report = await LiveAgent.usage_report(ctx)

        assert report.total == Usage(prompt_tokens=120, completion_tokens=48)
        assert report.by_model == {"gpt-realtime": Usage(prompt_tokens=120, completion_tokens=48)}
        assert report.by_provider == {"openai": Usage(prompt_tokens=120, completion_tokens=48)}

    async def test_empty_session(self) -> None:
        ctx = ConversationContext(stream=MemoryStream())

        report = await LiveAgent.usage_report(ctx)

        assert report == UsageReport()
