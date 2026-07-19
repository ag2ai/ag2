# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger governance middleware.

Tests use mocked TealTiger engine to avoid requiring the tealtiger package
in the test environment. Integration test at the bottom uses a real Agent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.extensions.tealtiger.types import GovernanceDecision, TEECReceipt
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

# ─── Test fixtures ───────────────────────────────────────────────────────────


def _make_context(agent_name: str = "assistant") -> MagicMock:
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent = MagicMock()
    agent.name = agent_name
    ctx.dependencies = {AGENT_CONTEXT_DEPENDENCY_KEY: agent}
    return ctx


def _make_tool_event(name: str = "search", args: str = '{"query": "hello"}') -> MagicMock:
    """Create a mock ToolCallEvent."""
    event = MagicMock(spec=ToolCallEvent)
    event.name = name
    event.arguments = args
    event.call_id = "call-123"
    return event


def _make_tool_result() -> MagicMock:
    """Create a mock ToolResultEvent (successful execution)."""
    result = MagicMock(spec=ToolResultEvent)
    return result


@pytest.fixture
def governance():
    """Create a TealTigerMiddleware with mocked TealTiger engine."""
    with patch("ag2.extensions.tealtiger.middleware.tealtiger") as mock_tt:
        # Mock TealEngine
        mock_engine = MagicMock()
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }
        mock_tt.TealEngine.return_value = mock_engine

        mw = TealTigerMiddleware(
            policies=[
                GovernancePolicy.tool_allowlist(["search", "read_*"]),
                GovernancePolicy.pii_block(["ssn", "credit_card"]),
                GovernancePolicy.cost_limit(max_per_session=5.0),
            ],
            mode=GovernanceMode.ENFORCE,
            cost_per_call=0.01,
        )
        yield mw, mock_engine


# ─── Factory pattern tests ───────────────────────────────────────────────────


class TestFactoryPattern:
    def test_call_returns_base_middleware(self, governance):
        mw_factory, _ = governance
        ctx = _make_context()
        event = MagicMock()

        per_turn = mw_factory(event, ctx)

        # Should be a BaseMiddleware instance
        from ag2.middleware import BaseMiddleware

        assert isinstance(per_turn, BaseMiddleware)

    def test_state_persists_across_turns(self, governance):
        mw_factory, _ = governance
        ctx = _make_context()
        event = MagicMock()

        turn1 = mw_factory(event, ctx)
        turn2 = mw_factory(event, ctx)

        # Both turns share the same factory state
        assert turn1._factory is turn2._factory
        assert turn1._factory._decisions is turn2._factory._decisions


# ─── Tool execution governance tests ────────────────────────────────────────


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_allow_passes_through(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        tool_event = _make_tool_event()
        expected_result = _make_tool_result()
        call_next = AsyncMock(return_value=expected_result)

        result = await per_turn.on_tool_execution(call_next, tool_event, ctx)

        assert result is expected_result
        call_next.assert_awaited_once()
        assert len(mw_factory._decisions) == 1
        assert mw_factory._decisions[0].action == "ALLOW"

    @pytest.mark.asyncio
    async def test_deny_returns_tool_error_event(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "DENY",
            "reason_codes": ["PII_DETECTED:ssn"],
            "risk_score": 90,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        tool_event = _make_tool_event(name="send_email")
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, tool_event, ctx)

        # Must return ToolErrorEvent, not a string
        assert isinstance(result, ToolErrorEvent)
        assert "GOVERNANCE DENIED" in str(result.error)
        assert "send_email" in str(result.error)
        # call_next should NOT be called for denied tools
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deny_in_observe_mode_still_allows(self, governance):
        """In OBSERVE mode, even DENY decisions pass through."""
        with patch("ag2.extensions.tealtiger.middleware.tealtiger") as mock_tt:
            mock_engine = MagicMock()
            mock_engine.evaluate.return_value = {
                "action": "DENY",
                "reason_codes": ["PII"],
                "risk_score": 90,
            }
            mock_tt.TealEngine.return_value = mock_engine

            mw = TealTigerMiddleware(
                policies=[GovernancePolicy.pii_block()],
                mode=GovernanceMode.OBSERVE,
            )

        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        tool_event = _make_tool_event()
        expected_result = _make_tool_result()
        call_next = AsyncMock(return_value=expected_result)

        result = await per_turn.on_tool_execution(call_next, tool_event, ctx)

        # OBSERVE mode: passes through even on DENY
        assert result is expected_result
        call_next.assert_awaited_once()


# ─── Kill switch tests ───────────────────────────────────────────────────────


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_freeze_blocks_all_tools(self, governance):
        mw_factory, _ = governance
        mw_factory.freeze("assistant")

        ctx = _make_context("assistant")
        per_turn = mw_factory(MagicMock(), ctx)
        tool_event = _make_tool_event()
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, tool_event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "AGENT_FROZEN" in str(result.error)
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unfreeze_restores_access(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }
        mw_factory.freeze("assistant")
        mw_factory.unfreeze("assistant")

        ctx = _make_context("assistant")
        per_turn = mw_factory(MagicMock(), ctx)
        tool_event = _make_tool_event()
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, tool_event, ctx)

        # Should pass through after unfreeze
        call_next.assert_awaited_once()

    def test_freeze_one_agent_doesnt_affect_another(self, governance):
        mw_factory, _ = governance
        mw_factory.freeze("agent-a")

        assert mw_factory.is_frozen("agent-a")
        assert not mw_factory.is_frozen("agent-b")


# ─── Cost tracking tests ────────────────────────────────────────────────────


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_cost_increments_on_allow(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert mw_factory.total_cost == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_cost_does_not_increment_on_deny(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "DENY",
            "reason_codes": ["BUDGET_EXCEEDED"],
            "risk_score": 70,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert mw_factory.total_cost == 0.0


# ─── Decision and receipt audit tests ────────────────────────────────────────


class TestAudit:
    @pytest.mark.asyncio
    async def test_each_call_produces_unique_decision_id(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        ids = [d.decision_id for d in mw_factory.decisions]
        assert len(ids) == 2
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_receipt_emitted_for_allow(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(mw_factory.receipts) == 1
        assert mw_factory.receipts[0].execution_outcome == "executed"

    @pytest.mark.asyncio
    async def test_receipt_emitted_for_deny(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "DENY",
            "reason_codes": ["TOOL_NOT_ALLOWED"],
            "risk_score": 80,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(mw_factory.receipts) == 1
        assert mw_factory.receipts[0].execution_outcome == "blocked"
        assert mw_factory.receipts[0].policy_digest != ""

    @pytest.mark.asyncio
    async def test_on_decision_callback_invoked(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }
        received = []
        mw_factory.on_decision = lambda d: received.append(d)

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(received) == 1
        assert isinstance(received[0], GovernanceDecision)

    @pytest.mark.asyncio
    async def test_on_receipt_callback_invoked(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "DENY",
            "reason_codes": ["SECRET"],
            "risk_score": 95,
        }
        received = []
        mw_factory.on_receipt = lambda r: received.append(r)

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)

        assert len(received) == 1
        assert isinstance(received[0], TEECReceipt)


# ─── Reset tests ─────────────────────────────────────────────────────────────


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self, governance):
        mw_factory, mock_engine = governance
        mock_engine.evaluate.return_value = {
            "action": "ALLOW",
            "reason_codes": ["POLICY_ALLOW"],
            "risk_score": 0,
        }

        ctx = _make_context()
        per_turn = mw_factory(MagicMock(), ctx)
        call_next = AsyncMock(return_value=_make_tool_result())

        await per_turn.on_tool_execution(call_next, _make_tool_event(), ctx)
        mw_factory.freeze("assistant")

        mw_factory.reset()

        assert len(mw_factory.decisions) == 0
        assert len(mw_factory.receipts) == 0
        assert mw_factory.total_cost == 0.0
        assert not mw_factory.is_frozen("assistant")
