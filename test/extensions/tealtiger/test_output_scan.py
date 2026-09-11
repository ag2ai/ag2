# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger output scanning: PII/secrets in what a tool *returns*.

Each case scripts a real turn with ``TestConfig`` and reads what actually reached
the model through ``TrackingConfig``, so "redacted" is observed rather than claimed.
"""

import json
from typing import Any

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.events.input_events import TextInput
from ag2.events.tool_events import ToolResult
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, OutputAction, TealTigerMiddleware
from ag2.testing import TestConfig, TrackingConfig

SSN = "123-45-6789"
AWS_KEY = "AKIA1234567890ABCDEF"
EMAIL = "ada@example.com"


def _call(tool_name: str, **arguments: Any) -> ToolCallEvent:
    """The tool call a model would emit for these arguments."""
    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


def _results_seen_by_model(tracking: TrackingConfig) -> list[Any]:
    """The tool-result payloads the framework handed the model on its last call."""
    message = tracking.mock.call_args.args[0]
    assert isinstance(message, ToolResultsEvent)
    return [
        part.content if isinstance(part, TextInput) else part.data
        for result in message.results
        for part in result.result.parts
    ]


def _reason_codes(governance: TealTigerMiddleware) -> list[str]:
    """Every reason code recorded across the session, in order."""
    return [code for decision in governance.decisions for code in decision.reason_codes]


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

    def test_scan_pii_with_empty_categories_is_rejected(self):
        # scan_pii=True + [] would construct but scan no category — must fail loudly.
        with pytest.raises(ValueError, match="at least one PII category"):
            GovernancePolicy.output_scan(scan_pii=True, scan_secrets=False, categories=[])

    def test_empty_categories_is_allowed_when_only_secrets_are_scanned(self):
        # With scan_pii=False the empty PII category list is irrelevant, not an error.
        policy = GovernancePolicy.output_scan(scan_pii=False, scan_secrets=True, categories=[])
        assert policy.config["scan_pii"] is False

    def test_an_action_is_accepted_as_an_enum_member_or_its_string_name(self):
        from_enum = GovernancePolicy.output_scan(pii_action=OutputAction.BLOCK)
        from_string = GovernancePolicy.output_scan(pii_action="BLOCK")

        # Either way the policy stores the plain string, so receipts stay JSON-clean.
        assert from_enum.config == from_string.config
        assert from_enum.config["pii_action"] == "BLOCK"

    def test_defaults(self):
        policy = GovernancePolicy.output_scan()

        assert policy.config == {
            "scan_pii": True,
            "scan_secrets": True,
            "pii_action": "REDACT",
            "secret_action": "BLOCK",
            "categories": ["ssn", "credit_card", "email", "phone"],
        }


@pytest.mark.asyncio
async def test_a_secret_in_the_result_blocks_the_turn_in_enforce():
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

    assert governance.deny_count == 1
    assert "OUTPUT_BLOCKED" in _reason_codes(governance)
    assert governance.receipts[-1].execution_outcome == "blocked"


@pytest.mark.asyncio
class TestPiiInTheResultIsRedacted:
    async def test_the_pii_never_reaches_the_model(self):
        def lookup_customer(name: str) -> str:
            return f"{name} SSN {SSN}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # pii_action defaults to REDACT
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("lookup_customer", name="Ada"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        reply = await agent.ask("look up Ada")

        # Redaction does not fail the turn — the sanitized result flows through.
        assert reply.body == "Done."
        assert _results_seen_by_model(tracking) == ["Ada SSN [REDACTED:ssn]"]
        assert "OUTPUT_PII_DETECTED:ssn" in _reason_codes(governance)
        assert "OUTPUT_REDACTED" in _reason_codes(governance)

    async def test_a_clean_result_passes_through_without_findings(self):
        def get_weather(city: str) -> str:
            return "Sunny, 22C"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("get_weather", city="Paris"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[get_weather], middleware=[governance])

        reply = await agent.ask("weather in Paris")

        assert reply.body == "Done."
        assert _results_seen_by_model(tracking) == ["Sunny, 22C"]
        assert all(not code.startswith("OUTPUT_") for code in _reason_codes(governance))


@pytest.mark.asyncio
class TestOutputScanRespectsMode:
    async def test_observe_passes_the_result_through_unmodified(self):
        def lookup_customer(name: str) -> str:
            return f"{name} SSN {SSN}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.OBSERVE,
        )
        tracking = TrackingConfig(TestConfig(_call("lookup_customer", name="Ada"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        await agent.ask("look up Ada")

        # OBSERVE never alters a call, so it never alters a result either.
        assert _results_seen_by_model(tracking) == [f"Ada SSN {SSN}"]
        assert _reason_codes(governance) == ["OBSERVE_PASSTHROUGH"]

    async def test_monitor_degrades_a_block_to_redaction(self):
        def read_config(key: str) -> str:
            return f"aws_key={AWS_KEY}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # secret_action BLOCK...
            mode=GovernanceMode.MONITOR,  # ...which cannot terminate the turn here
        )
        tracking = TrackingConfig(TestConfig(_call("read_config", key="aws"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[read_config], middleware=[governance])

        reply = await agent.ask("what is the aws key")

        # The turn survives, but the secret still never reaches the model.
        assert reply.body == "Done."
        assert _results_seen_by_model(tracking) == ["aws_key=[REDACTED:secret]"]
        assert "OUTPUT_REDACTED" in _reason_codes(governance)
        assert "OUTPUT_BLOCKED" not in _reason_codes(governance)


@pytest.mark.asyncio
class TestEachDetectorActsIndependently:
    async def test_flag_records_the_finding_and_changes_nothing(self):
        def lookup_customer(name: str) -> str:
            return f"ssn {SSN}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan(pii_action="FLAG", scan_secrets=False)],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("lookup_customer", name="Ada"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        await agent.ask("look up Ada")

        assert _results_seen_by_model(tracking) == [f"ssn {SSN}"]
        assert "OUTPUT_PII_DETECTED:ssn" in _reason_codes(governance)
        assert "OUTPUT_REDACTED" not in _reason_codes(governance)

    async def test_scan_pii_false_leaves_pii_alone_while_redacting_a_secret(self):
        def read_config(key: str) -> str:
            return f"owner {EMAIL} key={AWS_KEY}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan(scan_pii=False, secret_action="REDACT")],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("read_config", key="aws"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[read_config], middleware=[governance])

        await agent.ask("read the config")

        # The email is PII the policy was told not to scan — it must survive untouched.
        assert _results_seen_by_model(tracking) == [f"owner {EMAIL} key=[REDACTED:secret]"]
        assert all(not code.startswith("OUTPUT_PII_DETECTED") for code in _reason_codes(governance))

    async def test_a_flagged_detector_does_not_redact_alongside_a_redacting_one(self):
        def read_record(key: str) -> str:
            return f"ssn {SSN} key={AWS_KEY}"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan(pii_action="FLAG", secret_action="REDACT")],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("read_record", key="ada"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[read_record], middleware=[governance])

        await agent.ask("read the record")

        assert _results_seen_by_model(tracking) == [f"ssn {SSN} key=[REDACTED:secret]"]
        assert "OUTPUT_PII_DETECTED:ssn" in _reason_codes(governance)


@pytest.mark.asyncio
async def test_the_most_restrictive_action_across_policies_wins():
    def lookup_customer(name: str) -> str:
        return f"ssn {SSN}"

    governance = TealTigerMiddleware(
        policies=[
            GovernancePolicy.output_scan(pii_action="BLOCK", scan_secrets=False),
            GovernancePolicy.output_scan(pii_action="FLAG", scan_secrets=False),
        ],
        mode=GovernanceMode.ENFORCE,
    )
    agent = Agent(
        "assistant",
        config=TestConfig(_call("lookup_customer", name="Ada"), "Done."),
        tools=[lookup_customer],
        middleware=[governance],
    )

    # The later FLAG policy must not soften the earlier BLOCK.
    with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*OUTPUT_PII_DETECTED:ssn"):
        await agent.ask("look up Ada")


@pytest.mark.asyncio
class TestStructuredResultsAreScannedToo:
    async def test_pii_in_a_dict_result_is_redacted(self):
        def lookup_customer(name: str) -> dict[str, str]:
            return {"name": name, "ssn": SSN}

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(TestConfig(_call("lookup_customer", name="Ada"), "Done."))
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        await agent.ask("look up Ada")

        # A dict result arrives as a DataInput, not a TextInput — it leaks just the same.
        assert _results_seen_by_model(tracking) == [{"name": "Ada", "ssn": "[REDACTED:ssn]"}]
        assert "OUTPUT_PII_DETECTED:ssn" in _reason_codes(governance)

    async def test_a_secret_nested_in_a_dict_result_blocks_the_turn(self):
        def read_config(key: str) -> dict[str, Any]:
            return {"env": "prod", "credentials": [{"aws": AWS_KEY}]}

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("read_config", key="aws"), "Done."),
            tools=[read_config],
            middleware=[governance],
        )

        with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*OUTPUT_SECRET_DETECTED"):
            await agent.ask("read the config")


@pytest.mark.asyncio
async def test_a_value_split_across_parts_is_withheld_rather_than_half_redacted():
    def read_card() -> ToolResult:
        # Neither part matches on its own; together they spell a card number.
        return ToolResult(TextInput("card 4111"), TextInput("1111 1111 1111 on file"))

    governance = TealTigerMiddleware(
        policies=[GovernancePolicy.output_scan()],
        mode=GovernanceMode.ENFORCE,
    )
    agent = Agent(
        "assistant",
        config=TestConfig(_call("read_card"), "Done."),
        tools=[read_card],
        middleware=[governance],
    )

    # Redaction works part by part and cannot cut a value that spans two parts,
    # so the result is withheld instead of passed on half-sanitized.
    with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*OUTPUT_REDACTION_INCOMPLETE"):
        await agent.ask("show the card on file")

    assert "OUTPUT_PII_DETECTED:credit_card" in _reason_codes(governance)


@pytest.mark.asyncio
class TestToolErrorsAreScannedToo:
    """A tool that raises can leak PII/secrets through its exception message.

    ``ToolErrorEvent`` subclasses ``ToolResultEvent`` and carries the traceback in
    its result parts plus the exception itself, both of which reach the model — so
    output scanning covers error results, not only successful ones.
    """

    async def test_pii_in_an_exception_is_redacted_not_leaked(self):
        def lookup_customer(name: str) -> str:
            raise RuntimeError(f"lookup failed for SSN {SSN}")

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # pii_action REDACT
            mode=GovernanceMode.ENFORCE,
        )
        # raise_tool_errors=False models a real provider, which is handed the failure.
        tracking = TrackingConfig(
            TestConfig(_call("lookup_customer", name="Ada"), "Done.", raise_tool_errors=False)
        )
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        await agent.ask("look up Ada")

        [error_text] = _results_seen_by_model(tracking)
        assert SSN not in error_text
        assert "[REDACTED:ssn]" in error_text
        assert "OUTPUT_PII_DETECTED:ssn" in _reason_codes(governance)
        assert "OUTPUT_REDACTED" in _reason_codes(governance)

    async def test_a_secret_in_an_exception_is_withheld_in_enforce(self):
        def read_config(key: str) -> str:
            raise RuntimeError(f"connection string aws_key={AWS_KEY}")

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],  # secret_action BLOCK
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(
            TestConfig(_call("read_config", key="aws"), "Done.", raise_tool_errors=False)
        )
        agent = Agent("assistant", config=tracking, tools=[read_config], middleware=[governance])

        await agent.ask("what is the aws key")

        # The credential must not reach the model on any error path.
        [error_text] = _results_seen_by_model(tracking)
        assert AWS_KEY not in error_text
        assert "OUTPUT_SECRET_DETECTED" in _reason_codes(governance)

    async def test_a_clean_exception_passes_through_untouched(self):
        def lookup_customer(name: str) -> str:
            raise RuntimeError("customer not found")

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.output_scan()],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(
            TestConfig(_call("lookup_customer", name="Ada"), "Done.", raise_tool_errors=False)
        )
        agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

        await agent.ask("look up Ada")

        [error_text] = _results_seen_by_model(tracking)
        assert "customer not found" in error_text
        assert all(not code.startswith("OUTPUT_") for code in _reason_codes(governance))


@pytest.mark.asyncio
async def test_the_result_is_untouched_without_an_output_scan_policy():
    def lookup_customer(name: str) -> str:
        return f"ssn {SSN}"

    governance = TealTigerMiddleware(
        policies=[GovernancePolicy.tool_allowlist(["*"])],
        mode=GovernanceMode.ENFORCE,
    )
    tracking = TrackingConfig(TestConfig(_call("lookup_customer", name="Ada"), "Done."))
    agent = Agent("assistant", config=tracking, tools=[lookup_customer], middleware=[governance])

    await agent.ask("look up Ada")

    assert _results_seen_by_model(tracking) == [f"ssn {SSN}"]
    assert all(not code.startswith("OUTPUT_") for code in _reason_codes(governance))
