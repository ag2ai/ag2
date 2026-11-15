# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``TokenMonitor(max_usd=...)``: a guard on the priced bill (the ``pricing`` extra)."""

from contextlib import ExitStack

import pytest

llm_prices = pytest.importorskip("llm_prices")

from ag2 import Context  # noqa: E402
from ag2.events import ObserverAlert, Severity, Usage, UsageEvent  # noqa: E402
from ag2.observers import TokenMonitor  # noqa: E402
from ag2.stream import MemoryStream  # noqa: E402
from ag2.usage import price_usage  # noqa: E402

_NO_THRESHOLD = 10**9


def _model() -> str:
    return next(p for p in llm_prices.Provider.ANTHROPIC.rows() if p.cache_read is not None).model


def _armed(monitor: TokenMonitor) -> tuple[Context, list[ObserverAlert]]:
    stream = MemoryStream()
    ctx = Context(stream=stream)
    alerts: list[ObserverAlert] = []
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))
    monitor.register(ExitStack(), ctx)
    return ctx, alerts


@pytest.mark.asyncio
class TestTokenMonitorMaxUsd:
    async def test_critical_once_the_priced_cost_reaches_max_usd(self) -> None:
        model = _model()
        usage = Usage(prompt_tokens=100_000, completion_tokens=1_000)
        usd = price_usage(usage, model=model, provider="anthropic")
        monitor = TokenMonitor(_NO_THRESHOLD, _NO_THRESHOLD, max_usd=usd * 1.5)
        ctx, alerts = _armed(monitor)

        await ctx.send(UsageEvent(usage, model=model, provider="anthropic"))
        assert alerts == []
        assert monitor.total_usd == pytest.approx(usd)

        await ctx.send(UsageEvent(usage, model=model, provider="anthropic"))
        assert [a.severity for a in alerts] == [Severity.CRITICAL]
        assert "Cost critical" in alerts[0].message
        assert monitor.total_usd == pytest.approx(2 * usd)

    async def test_cache_reads_move_the_dollar_guard_by_what_they_bill(self) -> None:
        """A cached turn is far over the token budget but well inside the dollar one."""
        model = _model()
        cached = Usage(prompt_tokens=100_000, completion_tokens=1_000, cache_read_input_tokens=95_000)
        fresh = Usage(prompt_tokens=100_000, completion_tokens=1_000)
        limit = price_usage(fresh, model=model, provider="anthropic")
        monitor = TokenMonitor(warn_threshold=50_000, alert_threshold=_NO_THRESHOLD, max_usd=limit)
        ctx, alerts = _armed(monitor)

        await ctx.send(UsageEvent(cached, model=model, provider="anthropic"))

        assert [a.severity for a in alerts] == [Severity.WARNING]  # the token warning only
        assert "Token usage warning" in alerts[0].message
        assert 0 < monitor.total_usd < limit

    async def test_unpriced_model_warns_once_and_is_not_counted(self) -> None:
        monitor = TokenMonitor(_NO_THRESHOLD, _NO_THRESHOLD, max_usd=1.0)
        ctx, alerts = _armed(monitor)
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)

        await ctx.send(UsageEvent(usage, model="llama3", provider="ollama"))
        await ctx.send(UsageEvent(usage, model="llama3", provider="ollama"))

        assert [a.severity for a in alerts] == [Severity.WARNING]
        assert "'llama3'" in alerts[0].message
        assert monitor.total_usd == 0.0

    async def test_reset_clears_the_cost(self) -> None:
        model = _model()
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)
        monitor = TokenMonitor(_NO_THRESHOLD, _NO_THRESHOLD, max_usd=10**-9)
        ctx, alerts = _armed(monitor)

        await ctx.send(UsageEvent(usage, model=model, provider="anthropic"))
        monitor.reset()
        assert monitor.total_usd == 0.0

        await ctx.send(UsageEvent(usage, model=model, provider="anthropic"))
        assert [a.severity for a in alerts] == [Severity.CRITICAL, Severity.CRITICAL]

    async def test_without_max_usd_nothing_is_priced(self) -> None:
        monitor = TokenMonitor(_NO_THRESHOLD, _NO_THRESHOLD)
        ctx, alerts = _armed(monitor)

        await ctx.send(UsageEvent(Usage(prompt_tokens=1_000), model="llama3", provider="ollama"))

        assert alerts == []
        assert monitor.total_usd == 0.0
