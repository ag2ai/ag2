# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger governance middleware.

No external dependencies beyond AG2 and stdlib.
Uses mock Context/events to test governance logic without a running agent.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.extensions.tealtiger.types import TEECReceipt

# ─── Test fixtures ───────────────────────────────────────────────────────────


def _make_context(agent_name: str = "assistant") -> MagicMock:
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent = MagicMock()
    agent.name = agent_name
    ctx.dependencies = {AGENT_CONTEXT_DEPENDENCY_KEY: agent}
    return ctx


def _make_tool_event(name: str = "search", arguments: dict | None = None) -> MagicMock:
    """Create a mock ToolCallEvent with serialized_arguments."""
    event = MagicMock(spec=ToolCallEvent)
    event.name = name
    args = arguments or {}
    event.serialized_arguments = args
    event.arguments = json.dumps(args)
    event.call_id = "call-123"
    return event


# ─── Factory pattern tests ───────────────────────────────────────────────────


class TestFactoryPattern:
    def test_call_returns_base_middleware(self):
        mw = TealTigerMiddleware(policies=[GovernancePolicy.tool_allowlist(["search"])])
        ctx = _make_context()
        event = MagicMock()

        per_turn = mw(event, ctx)

        from ag2.middleware import BaseMiddleware

        assert isinstance(per_turn, BaseMiddleware)

    def test_state_persists_across_turns(self):
        mw = TealTigerMiddleware()
        ctx = _make_context()
        event = MagicMock()

        turn1 = mw(event, ctx)
        turn2 = mw(event, ctx)

        assert turn1._factory is turn2._factory
        assert turn1._factory._decisions is turn2._factory._decisions


# ─── Tool allowlist tests ────────────────────────────────────────────────────


class TestToolAllowlist:
    @pytest.mark.asyncio
    async def test_allowed_tool_passes(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search", "read_*"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio
    async def test_denied_tool_returns_error(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "GOVERNANCE DENIED" in str(result.error)
        assert "TOOL_NOT_ALLOWED" in str(result.error)
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_glob_pattern_matching(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["read_*", "search"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="read_file")
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()


# ─── PII detection tests ─────────────────────────────────────────────────────


class TestPIIDetection:
    @pytest.mark.asyncio
    async def test_ssn_blocked(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.pii_block(["ssn"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="send", arguments={"data": "SSN: 123-45-6789"})
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "PII_DETECTED:ssn" in str(result.error)
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clean_args_pass(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.pii_block(["ssn", "credit_card"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="search", arguments={"query": "weather"})
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_email_blocked(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.pii_block(["email"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="send", arguments={"to": "user@example.com"})
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_awaited()


# ─── Secret detection tests ──────────────────────────────────────────────────


class TestSecretDetection:
    @pytest.mark.asyncio
    async def test_openai_key_blocked(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.secret_detection()],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="run", arguments={"code": "key = 'sk-abcdefghij1234567890abcd'"})
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "SECRET_DETECTED" in str(result.error)

    @pytest.mark.asyncio
    async def test_aws_key_blocked(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.secret_detection()],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="deploy", arguments={"key": "AKIAIOSFODNN7EXAMPLE"})
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio
    async def test_clean_code_passes(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.secret_detection()],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="run", arguments={"code": "x = 1 + 2"})
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()


# ─── Kill switch tests ───────────────────────────────────────────────────────


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_frozen_agent_blocked(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        mw.freeze("assistant")

        ctx = _make_context("assistant")
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event()
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "AGENT_FROZEN" in str(result.error)
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unfreeze_restores_access(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        mw.freeze("assistant")
        mw.unfreeze("assistant")

        ctx = _make_context("assistant")
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event()
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()

    def test_freeze_one_doesnt_affect_another(self):
        mw = TealTigerMiddleware()
        mw.freeze("agent-a")

        assert mw.is_frozen("agent-a")
        assert not mw.is_frozen("agent-b")


# ─── Governance mode tests ───────────────────────────────────────────────────


class TestGovernanceModes:
    @pytest.mark.asyncio
    async def test_observe_allows_denied_tool(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.OBSERVE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        # OBSERVE mode allows everything through
        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio
    async def test_monitor_allows_denied_tool(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.MONITOR,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        # MONITOR mode allows through
        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio
    async def test_enforce_blocks_denied_tool(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_awaited()


# ─── Cost tracking tests ─────────────────────────────────────────────────────


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_cost_increments_on_allow(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE, cost_per_call=0.01)
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert mw.total_cost == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_cost_not_incremented_on_deny(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, _make_tool_event(name="bad_tool"), ctx)

        assert mw.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_budget_limit_enforced(self):
        mw = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            budget_limit=0.015,
            cost_per_call=0.01,
        )
        ctx = _make_context()
        call_next = AsyncMock(return_value=MagicMock())

        # First call: cost 0 < 0.015 → ALLOW
        per_turn = mw(MagicMock(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        # Second call: cost 0.01 < 0.015 → ALLOW
        per_turn = mw(MagicMock(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        # Third call: cost 0.02 >= 0.015 → DENY
        per_turn = mw(MagicMock(), ctx)
        call_next_deny = AsyncMock()
        result = await per_turn.on_tool_execution(call_next_deny, _make_tool_event(), ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "BUDGET_EXCEEDED" in str(result.error)
        call_next_deny.assert_not_awaited()


# ─── Audit trail tests ───────────────────────────────────────────────────────


class TestAudit:
    @pytest.mark.asyncio
    async def test_decisions_recorded(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(mw.decisions) == 1
        assert mw.decisions[0].action == "ALLOW"
        assert mw.decisions[0].agent_name == "assistant"

    @pytest.mark.asyncio
    async def test_unique_decision_ids(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        ids = [d.decision_id for d in mw.decisions]
        assert len(ids) == 2
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_receipt_emitted_for_allow(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(mw.receipts) == 1
        assert mw.receipts[0].execution_outcome == "executed"

    @pytest.mark.asyncio
    async def test_receipt_emitted_for_deny(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["search"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, _make_tool_event(name="bad"), ctx)

        assert len(mw.receipts) == 1
        assert mw.receipts[0].execution_outcome == "blocked"

    @pytest.mark.asyncio
    async def test_on_decision_callback(self):
        received = []
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE, on_decision=lambda d: received.append(d))
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(received) == 1
        assert received[0].agent_name == "assistant"

    @pytest.mark.asyncio
    async def test_on_receipt_callback(self):
        received = []
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE, on_receipt=lambda r: received.append(r))
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(received) == 1
        assert isinstance(received[0], TEECReceipt)


# ─── Reset tests ─────────────────────────────────────────────────────────────


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self):
        mw = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        call_next = AsyncMock(return_value=MagicMock())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        mw.freeze("assistant")

        mw.reset()

        assert len(mw.decisions) == 0
        assert len(mw.receipts) == 0
        assert mw.total_cost == 0.0
        assert not mw.is_frozen("assistant")
