# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.a2ui import A2UIAgent
from autogen.beta.testing import TestConfig

VALID_RESPONSE = (
    "Here is your UI.\n---a2ui_JSON---\n"
    '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
    '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}}]'
)

INVALID_RESPONSE_MISSING_CATALOG = (
    "Here.\n---a2ui_JSON---\n"
    '[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]'  # missing catalogId
)

INVALID_RESPONSE_BROKEN_JSON = "Text.\n---a2ui_JSON---\n{not valid json}"


@pytest.mark.asyncio()
class TestA2UIValidationMiddleware:
    async def test_valid_response_passes_through(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(VALID_RESPONSE),
            validate_responses=True,
            validation_retries=2,
        )
        reply = await agent.ask("Show UI")
        assert reply.body == VALID_RESPONSE

    async def test_plain_text_bypasses_validation(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig("Just plain text."),
            validate_responses=True,
            validation_retries=2,
        )
        reply = await agent.ask("Hi")
        assert reply.body == "Just plain text."

    async def test_retry_on_invalid_then_success(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(INVALID_RESPONSE_MISSING_CATALOG, VALID_RESPONSE),
            validate_responses=True,
            validation_retries=1,
        )
        reply = await agent.ask("Show UI")
        assert reply.body == VALID_RESPONSE

    async def test_retry_exhausted_returns_text_only(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(
                INVALID_RESPONSE_MISSING_CATALOG,
                INVALID_RESPONSE_MISSING_CATALOG,
                INVALID_RESPONSE_MISSING_CATALOG,
            ),
            validate_responses=True,
            validation_retries=2,
        )
        reply = await agent.ask("Show UI")
        # graceful degradation: text portion only, no A2UI delimiter
        assert reply.body == "Here."

    async def test_json_parse_error_triggers_retry(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(INVALID_RESPONSE_BROKEN_JSON, VALID_RESPONSE),
            validate_responses=True,
            validation_retries=1,
        )
        reply = await agent.ask("Show UI")
        assert reply.body == VALID_RESPONSE

    async def test_no_retry_when_validation_disabled(self) -> None:
        # Invalid response should pass through unchanged when validation is off.
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(INVALID_RESPONSE_MISSING_CATALOG),
            validate_responses=False,
        )
        reply = await agent.ask("Show UI")
        assert reply.body == INVALID_RESPONSE_MISSING_CATALOG
