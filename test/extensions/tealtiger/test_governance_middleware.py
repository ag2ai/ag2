# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger governance middleware.

No external dependencies beyond AG2 and stdlib.
Uses real AG2 event types where possible, mocks for context.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.extensions.tealtiger import (
    TealTigerMiddleware,
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    TEECReceipt,
)


def _make_context(agent_name: str = "test-agent"):
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent_mock = MagicMock()
    agent_mock.name = agent_name
    ctx.dependencies = {"agent": agent_mock}
    return ctx


def _make_tool_event(name: str = "search", arguments: dict | None = None):
    """Create a real ToolCallEvent with JSON-serialized arguments."""
    args = arguments or {}
    return ToolCallEvent(
        id=f"call-{name}",
        name=name,
        arguments=json.dumps(args),
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
        assert hasattr(instance, "on_turn")

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

    @pytest.mark.asyncio
    async def test_ssn_detection(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.pii_block(["ssn"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="create_record",
            arguments={"data": "SSN is 123-45-6789"},
        )
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)
        assert isinstance(result, ToolErrorEvent)
        assert gov.decisions[-1].reason_codes == ["PII_DETECTED"]


class TestSecretDetection:
    """Test secret scanning in tool arguments."""

    @pytest.mark.asyncio
    async def test_api_key_blocked(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.secret_block()],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="write_code",
            arguments={"code": "api_key = 'sk-abcdefghijklmnopqrstuvwxyz1234567890'"},
        )
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)
        assert isinstance(result, ToolErrorEvent)
        assert gov.decisions[-1].reason_codes == ["SECRET_DETECTED"]

    @pytest.mark.asyncio
    async def test_aws_key_blocked(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.secret_block()],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="deploy",
            arguments={"config": "aws_key: AKIAIOSFODNN7EXAMPLE"},
        )
        call_next = AsyncMock()

        result = await instance.on_tool_execution(call_next, event, ctx)
        assert isinstance(result, ToolErrorEvent)
        assert gov.decisions[-1].reason_codes == ["SECRET_DETECTED"]

    @pytest.mark.asyncio
    async def test_clean_code_passes(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.secret_block()],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="write_code",
            arguments={"code": "x = 1 + 2"},
        )
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


class TestOnTurnKillSwitch:
    """Test on_turn kill switch enforcement."""

    @pytest.mark.asyncio
    async def test_frozen_agent_blocked_on_turn_enforce(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        gov.freeze("bad-agent")

        ctx = _make_context(agent_name="bad-agent")
        instance = gov(MagicMock(), ctx)

        call_next = AsyncMock()
        event = MagicMock()

        result = await instance.on_turn(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next.assert_not_called()
        assert gov.decisions[-1].reason_codes == ["KILL_SWITCH"]

    @pytest.mark.asyncio
    async def test_frozen_agent_allowed_on_turn_monitor(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.MONITOR)
        gov.freeze("bad-agent")

        ctx = _make_context(agent_name="bad-agent")
        instance = gov(MagicMock(), ctx)

        call_next = AsyncMock(return_value=MagicMock())
        event = MagicMock()

        result = await instance.on_turn(call_next, event, ctx)

        # MONITOR: records DENY but allows through
        call_next.assert_called_once()
        assert gov.decisions[-1].action == "DENY"

    @pytest.mark.asyncio
    async def test_unfrozen_agent_passes_on_turn(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)

        ctx = _make_context(agent_name="good-agent")
        instance = gov(MagicMock(), ctx)

        call_next = AsyncMock(return_value=MagicMock())
        event = MagicMock()

        result = await instance.on_turn(call_next, event, ctx)
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_observe_mode_skips_on_turn_check(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.OBSERVE)
        gov.freeze("agent")

        ctx = _make_context(agent_name="agent")
        instance = gov(MagicMock(), ctx)

        call_next = AsyncMock(return_value=MagicMock())
        event = MagicMock()

        result = await instance.on_turn(call_next, event, ctx)

        # OBSERVE: passes through without evaluation
        call_next.assert_called_once()
        # No decisions recorded in OBSERVE on_turn
        assert len(gov.decisions) == 0


class TestMonitorMode:
    """Test MONITOR mode evaluates policies but allows all through."""

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

    @pytest.mark.asyncio
    async def test_monitor_records_pii_but_allows(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.MONITOR,
            policies=[GovernancePolicy.pii_block(["email"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(
            name="send",
            arguments={"to": "user@test.com"},
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)

        call_next.assert_called_once()
        assert gov.decisions[-1].action == "DENY"
        assert gov.decisions[-1].reason_codes == ["PII_DETECTED"]


class TestObserveMode:
    """Test OBSERVE mode does not evaluate policies."""

    @pytest.mark.asyncio
    async def test_observe_skips_policy_evaluation(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.OBSERVE,
            policies=[GovernancePolicy.tool_allowlist(["search"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        # Tool NOT in allowlist, but OBSERVE mode skips evaluation
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)

        call_next.assert_called_once()
        # Decision recorded as ALLOW (observe passthrough)
        assert gov.decisions[-1].action == "ALLOW"
        assert "OBSERVE" in gov.decisions[-1].reason

    @pytest.mark.asyncio
    async def test_observe_still_emits_receipts(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.OBSERVE)
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(gov.receipts) == 1
        assert gov.receipts[0].execution_outcome == "executed"


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
    async def test_receipts_emitted_with_outcome(self):
        gov = TealTigerMiddleware(mode=GovernanceMode.ENFORCE)
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(gov.receipts) == 1
        assert gov.receipts[0].tool_name == "search"
        assert gov.receipts[0].decision_id == gov.decisions[0].decision_id
        assert gov.receipts[0].execution_outcome == "executed"

    @pytest.mark.asyncio
    async def test_denied_receipt_has_blocked_outcome(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.tool_allowlist(["search"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock()

        await instance.on_tool_execution(call_next, event, ctx)

        assert gov.receipts[-1].execution_outcome == "blocked"


class TestCostTracking:
    """Test cumulative cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_increments_by_cost_per_call(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            cost_per_call=0.01,
        )
        ctx = _make_context()

        for _ in range(5):
            instance = gov(MagicMock(), ctx)
            event = _make_tool_event(name="search")
            call_next = AsyncMock(return_value=MagicMock())
            await instance.on_tool_execution(call_next, event, ctx)

        assert abs(gov._state.cumulative_cost - 0.05) < 1e-9

    @pytest.mark.asyncio
    async def test_budget_exceeded_blocks_via_policy(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.cost_limit(max_per_session=0.001)],
            cost_per_call=0.002,
        )
        ctx = _make_context()

        # First call succeeds (cost starts at 0)
        instance = gov(MagicMock(), ctx)
        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())
        await instance.on_tool_execution(call_next, event, ctx)

        # Second call should be blocked (cost now 0.002 > 0.001)
        instance2 = gov(MagicMock(), ctx)
        event2 = _make_tool_event(name="search")
        call_next2 = AsyncMock()
        result = await instance2.on_tool_execution(call_next2, event2, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next2.assert_not_called()

    @pytest.mark.asyncio
    async def test_budget_limit_on_factory_enforced(self):
        """budget_limit on factory is enforced even without a cost_limit policy."""
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            budget_limit=0.003,
            cost_per_call=0.002,
        )
        ctx = _make_context()

        # First call: cost 0 < 0.003 → ALLOW, cost becomes 0.002
        instance = gov(MagicMock(), ctx)
        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())
        await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()

        # Second call: cost 0.002 < 0.003 → ALLOW, cost becomes 0.004
        instance2 = gov(MagicMock(), ctx)
        event2 = _make_tool_event(name="search")
        call_next2 = AsyncMock(return_value=MagicMock())
        await instance2.on_tool_execution(call_next2, event2, ctx)
        call_next2.assert_called_once()

        # Third call: cost 0.004 >= 0.003 → DENY
        instance3 = gov(MagicMock(), ctx)
        event3 = _make_tool_event(name="search")
        call_next3 = AsyncMock()
        result = await instance3.on_tool_execution(call_next3, event3, ctx)

        assert isinstance(result, ToolErrorEvent)
        call_next3.assert_not_called()
        assert gov.decisions[-1].reason_codes == ["BUDGET_EXCEEDED"]


class TestCallbacks:
    """Test on_decision and on_receipt callbacks."""

    @pytest.mark.asyncio
    async def test_on_decision_callback_invoked(self):
        decisions_seen = []
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            on_decision=lambda d: decisions_seen.append(d),
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(decisions_seen) == 1
        assert decisions_seen[0].agent_name == "test-agent"

    @pytest.mark.asyncio
    async def test_on_receipt_callback_invoked(self):
        receipts_seen = []
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            on_receipt=lambda r: receipts_seen.append(r),
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        await instance.on_tool_execution(call_next, event, ctx)

        assert len(receipts_seen) == 1
        assert receipts_seen[0].execution_outcome == "executed"


class TestArgumentParsing:
    """Test _parse_arguments handles various input formats."""

    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.pii_block(["ssn"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        # Empty string arguments
        event = ToolCallEvent(id="call-1", name="search", arguments="")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self):
        gov = TealTigerMiddleware(
            mode=GovernanceMode.ENFORCE,
            policies=[GovernancePolicy.pii_block(["ssn"])],
        )
        ctx = _make_context()
        instance = gov(MagicMock(), ctx)

        # Invalid JSON — should gracefully fall back to empty dict
        event = ToolCallEvent(id="call-1", name="search", arguments="not json{}")
        call_next = AsyncMock(return_value=MagicMock())

        result = await instance.on_tool_execution(call_next, event, ctx)
        call_next.assert_called_once()
