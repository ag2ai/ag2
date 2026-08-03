# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger governance middleware.

Uses real AG2 event types where possible. The test/__init__.py uses
pytest.importorskip("tealtiger") so these tests are skipped in CI
when tealtiger is not installed, and run locally when it is.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.extensions.tealtiger import (
    TealTigerMiddleware,
    GovernanceMode,
    GovernancePolicy,
)


def _make_context(agent_name: str = "test-agent"):
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent_mock = MagicMock()
    agent_mock.name = agent_name
    ctx.dependencies = {"agent": agent_mock}
    return ctx


def _make_tool_event(name: str = "search", arguments: dict | None = None):
    """Create a real ToolCallEvent."""
    return ToolCallEvent(
        id=f"call-{name}",
        name=name,
        arguments=arguments or {},
    )


class TestFactoryPattern:
    """Test that MiddlewareFactory pattern works correctly."""

    def test_factory_creates_per_turn_instance(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.tool_allowlist(["search"])],
        )
        ctx = _make_context()
        event = MagicMock()

        instance = gov(event, ctx)
        assert instance is not gov
        assert hasattr(instance, "on_tool_execution")

    def test_shared_state_persists_across_turns(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        gov.freeze("bad-agent")

        ctx = _make_context()
        event = MagicMock()

        inst1 = gov(event, ctx)
        inst2 = gov(event, ctx)

        # Both instances see the same frozen state
        assert inst1._state.frozen_agents == {"bad-agent"}
        assert inst2._state.frozen_agents == {"bad-agent"}


class TestToolAllowlist:
    """Test tool allowlist enforcement."""

    @pytest.mark.asyncio
    async def test_allowed_tool_passes(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.tool_allowlist(["search", "read_*"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)

        call_next.assert_called_once()
        assert not isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio
    async def test_denied_tool_returns_error_event(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.tool_allowlist(["search"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)

        # Should return ToolErrorEvent, NOT a string
        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_glob_pattern_matching(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.tool_allowlist(["read_*", "search"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        # read_file matches read_*
        event = _make_tool_event(name="read_file")
        call_next = AsyncMock(return_value=MagicMock())
        result = await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()


class TestPIIDetection:
    """Test PII scanning in tool arguments."""

    @pytest.mark.asyncio
    async def test_pii_in_args_blocks(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.pii_block(["email"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="send_email",
            arguments={"to": "john@example.com", "body": "hello"},
        )
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_args_pass(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.pii_block(["ssn"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search", arguments={"query": "weather"})
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()


class TestKillSwitch:
    """Test per-agent kill switch."""

    @pytest.mark.asyncio
    async def test_frozen_agent_blocked(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        gov.freeze("bad-agent")

        ctx = _make_context(agent_name="bad-agent")
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_unfrozen_agent_passes(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        gov.freeze("bad-agent")
        gov.unfreeze("bad-agent")

        ctx = _make_context(agent_name="bad-agent")
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_freeze_all_blocks_any_agent(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        gov.freeze("*")

        ctx = _make_context(agent_name="any-agent")
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)
        assert isinstance(result, ToolErrorEvent)


class TestMonitorMode:
    """Test MONITOR mode logs but allows."""

    @pytest.mark.asyncio
    async def test_monitor_allows_denied_tool(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.MONITOR,
            policies=[GovernancePolicy.tool_allowlist(["search"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)

        # MONITOR: allows through despite violation
        call_next.assert_called_once()
        # But decision is still recorded as DENY
        assert gov.decisions[-1].action == "DENY"


class TestAuditTrail:
    """Test audit trail and receipts."""

    @pytest.mark.asyncio
    async def test_decisions_recorded(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(gov.decisions) == 1
        assert gov.decisions[0].action == "ALLOW"
        assert gov.decisions[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_receipts_emitted(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(gov.receipts) == 1
        assert gov.receipts[0].tool_name == "search"
        assert gov.receipts[0].decision_id == gov.decisions[0].decision_id


class TestCostTracking:
    """Test cumulative cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_increments(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()

        for _ in range(5):
            instance = gov(MagicMock(), ctx)
            event = _make_tool_event(name="search")
            call_next = AsyncMock(return_value=MagicMock())
            await instance.on_tool_execution(call_next, event, ctx)

        assert gov._state.cumulative_cost > 0

    @pytest.mark.asyncio
    async def test_budget_exceeded_blocks(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.cost_limit(max_per_session=0.001)],
        )
        ctx = _make_context()

        # First call succeeds (cost starts at 0)
        instance = gov(MagicMock(), ctx)
        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())
        await instance.on_tool_execution(call_next, event, ctx)

        # Second call should be blocked (cost now > 0.001)
        instance2 = gov(MagicMock(), ctx)
        event2 = _make_tool_event(name="search")
        call_next2 = AsyncMock()
        result = await instance2.on_tool_execution(call_next2, event2, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next2.assert_not_called()
