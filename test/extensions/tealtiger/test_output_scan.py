# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger output scanning (post-tool defense).

Where `pii_block` / `secret_detection` inspect a tool's *arguments* before it
runs, `output_scan` inspects what the tool *returns* — the data-leakage
direction. A tool that reads a database, file, or API can hand back an SSN or a
leaked credential that would otherwise land in the model's context on the next
turn.

The end-to-end cases script a real agent turn with ``TestConfig`` so the policy
runs where it does in production and the sensitive value comes back the way a
tool would actually return it. Because scanning happens *after* the tool runs,
"blocked" is observed as the turn failing with a GOVERNANCE DENIED error, and
"redacted" is observed on the recorded ``GovernanceDecision`` plus a direct
check that the result content was rewritten in place.
"""

import json
from typing import Any

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.events.input_events import TextInput
from ag2.events.tool_events import ToolResult
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.extensions.tealtiger.middleware import _TealTigerPerTurn
from ag2.testing import TestConfig

SSN = "123-45-6789"
AWS_KEY = "AKIA1234567890ABCDEF"


def _call(tool_name: str, **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


# ── Policy validation ────────────────────────────────────────────────────────


class TestOutputScanPolicyValidation:
    def test_scanning_nothing_is_rejected(self):
        with pytest.raises(ValueError, match="must scan something"):
            GovernancePolicy.output_scan(scan_pii=False, scan_secrets=False)

    def test_invalid_action_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid pii_action"):
            GovernancePolicy.output_scan(pii_action="NUKE")

    def test_unknown_category_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown PII category"):
            GovernancePolicy.output_scan(categories=["ssn", "passport"])

    def test_defaults(self):
        policy = GovernancePolicy.output_scan()
        assert policy.type == "output_scan"
        assert policy.config["pii_action"] == "REDACT"
        assert policy.config["secret_action"] == "BLOCK"
        assert policy.config["scan_pii"] and policy.config["scan_secrets"]


# ── End-to-end: block ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSecretInResultIsBlocked:
    async def test_secret_result_blocks_the_turn_in_enforce(self):
        def read_config(key: str) -> str:
            # The tool itself runs fine — the danger is in what it returns.
            return f"aws_key={AWS_KEY}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # secret_action defaults to BLOCK
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("read_config", key="aws"), "Done."),
            tools=[read_config],
            middleware=[governance],
        )
        with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*OUTPUT_SECRET_DETECTED"):
            await agent.ask("what is the aws key")

        decision = governance.decisions[-1]
        assert decision.action == "DENY"
        assert "OUTPUT_BLOCKED" in decision.reason_codes
        assert governance.receipts[-1].execution_outcome == "blocked"


# ── End-to-end: PII redaction (default) ────────────────────────────────────────


@pytest.mark.asyncio
class TestPiiInResultIsRedacted:
    async def test_pii_result_is_recorded_as_redacted(self):
        def lookup_customer(name: str) -> str:
            return f"{name} SSN {SSN}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # pii_action defaults to REDACT
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("lookup_customer", name="Ada"), "Done."),
            tools=[lookup_customer],
            middleware=[governance],
        )
        reply = await agent.ask("look up Ada")
        # Redaction does not fail the turn — the sanitized result flows through.
        assert reply.body == "Done."
        decision = governance.decisions[-1]
        assert decision.action == "ALLOW"
        assert any(rc.startswith("OUTPUT_PII_DETECTED") for rc in decision.reason_codes)
        assert "OUTPUT_REDACTED" in decision.reason_codes

    async def test_clean_result_passes_through_without_findings(self):
        def get_weather(city: str) -> str:
            return "Sunny, 22C"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("get_weather", city="Paris"), "Done."),
            tools=[get_weather],
            middleware=[governance],
        )
        reply = await agent.ask("weather in Paris")
        assert reply.body == "Done."
        # No output-scan decision recorded for a clean result (only the pre-exec ALLOW).
        assert all("OUTPUT_" not in rc for d in governance.decisions for rc in d.reason_codes)


# ── Direct: prove the result content is actually rewritten ─────────────────────


class _StubContext:
    """Minimal context: _get_agent_name reads context.dependencies.get(...)."""

    def __init__(self) -> None:
        self.dependencies: dict[Any, Any] = {}


def _result_event(text: str) -> ToolResultEvent:
    return ToolResultEvent(parent_id="call_1", name="tool_x", result=ToolResult(TextInput(text)))


def _perturn(factory: TealTigerMiddleware) -> _TealTigerPerTurn:
    # Build a per-turn instance without a real event; _scan_result only needs
    # the factory state and the (event, tool_name, result) it is given.
    inst = _TealTigerPerTurn.__new__(_TealTigerPerTurn)
    inst._factory = factory
    inst._agent_name = "assistant"
    return inst


class TestRedactionRewritesContent:
    def test_pii_value_removed_from_result_content(self):
        factory = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan(scan_secrets=False)],
            mode=GovernanceMode.ENFORCE,
        )
        inst = _perturn(factory)
        event = _call("lookup", name="x")
        result = _result_event(f"the ssn is {SSN}")

        scanned = inst._scan_result(event, "lookup", result)

        # Same event type (redact, not block), but the SSN is gone.
        assert isinstance(scanned, ToolResultEvent)
        content = scanned.result.parts[0].content
        assert SSN not in content
        assert "[REDACTED:ssn]" in content

    def test_secret_block_degrades_to_redaction_in_monitor(self):
        factory = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.MONITOR,  # BLOCK cannot terminate here
        )
        inst = _perturn(factory)
        event = _call("read_config", key="k")
        result = _result_event(f"key={AWS_KEY}")

        scanned = inst._scan_result(event, "read_config", result)

        # Not terminated (still a ToolResultEvent), but the secret never leaks.
        assert isinstance(scanned, ToolResultEvent)
        assert AWS_KEY not in scanned.result.parts[0].content
        assert "OUTPUT_REDACTED" in factory.decisions[-1].reason_codes

    def test_flag_passes_result_through_unchanged(self):
        factory = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan(pii_action="FLAG", scan_secrets=False)],
            mode=GovernanceMode.ENFORCE,
        )
        inst = _perturn(factory)
        event = _call("lookup", name="x")
        result = _result_event(f"ssn {SSN}")

        scanned = inst._scan_result(event, "lookup", result)

        # FLAG records but does not modify.
        assert scanned.result.parts[0].content == f"ssn {SSN}"
        assert any(rc.startswith("OUTPUT_PII_DETECTED") for rc in factory.decisions[-1].reason_codes)
        assert "OUTPUT_REDACTED" not in factory.decisions[-1].reason_codes

    def test_no_output_scan_policy_leaves_result_untouched(self):
        factory = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_allowlist(["*"])],
            mode=GovernanceMode.ENFORCE,
        )
        inst = _perturn(factory)
        event = _call("lookup", name="x")
        result = _result_event(f"ssn {SSN}")

        scanned = inst._scan_result(event, "lookup", result)
        assert scanned is result  # untouched, same object
