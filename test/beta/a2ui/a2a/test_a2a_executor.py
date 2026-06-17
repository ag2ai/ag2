# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2UI ⇄ A2A behaviour, exercised end to end through a real client→server round-trip.

These tests drive the public surface only: a client :class:`~autogen.beta.Agent`
talks to an :class:`A2AServer` wired with an :class:`A2UIAgentExecutor` over an
in-process httpx ``ASGITransport`` (the same harness the ``test/beta/a2a`` E2E
suite uses). Incoming A2UI envelopes ride the outgoing message via the
``EXTRA_PARTS`` dependency hatch; outgoing A2UI DataParts and task-state
transitions are observed off the client's event stream.

The server LLM is mocked. Two doubles appear, both mirroring the sanctioned
pattern in ``test/beta/a2a/_helpers.py`` (``RecordingConfig``): ``TestConfig``
for canned responses, and a local capturing config for the cases where the
assertion is about *what the executor fed the model* — the resolved system
prompt (capability negotiation) or the synthesized user turn (incoming-envelope
rewrites) — neither of which ``TrackingConfig`` (last message only) exposes.
"""

import json

import pytest
from a2a.types import Part, TaskState

from autogen.beta import Agent
from autogen.beta.a2a.extension import EXTRA_PARTS_DEPENDENCY_KEY
from autogen.beta.a2ui import a2ui_action
from autogen.beta.a2ui.a2a import create_a2ui_parts
from autogen.beta.a2ui.a2a.executor import _extract_a2ui_envelopes
from autogen.beta.events import ToolCallEvent
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig

from ._helpers import (
    FUNCTION_RESULT_MARK,
    CallFunctionThenComplete,
    CapturingConfig,
    MetadataInterceptor,
    client_for,
    subscribe_task_stream,
    synthesized_text,
)

VERSION = "v0.9"
CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

DELETE_SURFACE_MSG = {"version": VERSION, "deleteSurface": {"surfaceId": "s1"}}
CREATE_SURFACE_MSG = {"version": VERSION, "createSurface": {"surfaceId": "s1", "catalogId": CATALOG}}

ACTION_ENVELOPE = {
    "version": VERSION,
    "action": {
        "name": "submit",
        "surfaceId": "s1",
        "sourceComponentId": "submit_btn",
        "timestamp": "2026-06-14T00:00:00Z",
        "context": {"email": "user@example.com"},
    },
}

ERROR_ENVELOPE = {
    "version": VERSION,
    "error": {
        "code": "VALIDATION_FAILED",
        "surfaceId": "s1",
        "message": "bad component",
        "path": "/components/0",
    },
}

FUNCTION_RESPONSE_ENVELOPE = {
    "version": "v1.0",
    "functionResponse": {"functionCallId": "fc-1", "call": "openUrl", "value": True},
}

# A server→client callFunction the LLM may emit inside <a2ui-json>. ``openUrl``
# is a basic-catalog function, so this validates under v1.0.
_CALL_FUNCTION_BLOCK = (
    '[{"version":"v1.0","functionCallId":"fc-1","wantResponse":true,'
    '"callFunction":{"call":"openUrl","args":{"url":"https://example.com"}}}]'
)


class TestExtractA2UIEnvelopes:
    """Envelope decode — the three on-the-wire shapes from get_a2ui_data."""

    def test_returns_empty_for_text_part(self) -> None:
        assert _extract_a2ui_envelopes(Part(text="hello")) == []

    def test_canonical_list_payload(self) -> None:
        [part] = create_a2ui_parts([ACTION_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_legacy_single_dict_payload(self) -> None:
        [part] = create_a2ui_parts(ACTION_ENVELOPE, legacy_split=True)
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_filters_entries_without_action_or_error(self) -> None:
        [part] = create_a2ui_parts([CREATE_SURFACE_MSG, ACTION_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_decodes_error_envelope(self) -> None:
        [part] = create_a2ui_parts([ERROR_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ERROR_ENVELOPE]

    def test_decodes_function_response_envelope(self) -> None:
        [part] = create_a2ui_parts([FUNCTION_RESPONSE_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [FUNCTION_RESPONSE_ENVELOPE]


@pytest.mark.asyncio
class TestFinalizationMessage:
    """The completed turn splits into conversational prose plus a canonical A2UI DataPart."""

    async def test_splits_prose_and_a2ui_datapart(self) -> None:
        response = f"Here is your UI.\n<a2ui-json>\n[{json.dumps(CREATE_SURFACE_MSG)}]\n</a2ui-json>"
        agent = Agent(name="ui_agent", config=TestConfig(response))
        client = client_for(agent, streaming=True)
        stream = MemoryStream()
        payloads, _states = subscribe_task_stream(stream)

        reply = await client.ask("show me a form", stream=stream)

        assert reply.response.content == "Here is your UI."
        assert payloads == [[CREATE_SURFACE_MSG]]

    async def test_a2ui_only_without_prose(self) -> None:
        response = f"<a2ui-json>\n[{json.dumps(DELETE_SURFACE_MSG)}]\n</a2ui-json>"
        agent = Agent(name="ui_agent", config=TestConfig(response))
        client = client_for(agent, streaming=True)
        stream = MemoryStream()
        payloads, _states = subscribe_task_stream(stream)

        await client.ask("clear it", stream=stream)

        assert payloads == [[DELETE_SURFACE_MSG]]

    async def test_plain_text_has_no_a2ui_datapart(self) -> None:
        agent = Agent(name="ui_agent", config=TestConfig("Just text."))
        client = client_for(agent, streaming=True, validate_responses=False)
        stream = MemoryStream()
        payloads, _states = subscribe_task_stream(stream)

        reply = await client.ask("hi", stream=stream)

        assert reply.response.content == "Just text."
        assert payloads == []


@pytest.mark.asyncio
class TestIncomingActionRewrite:
    """An incoming A2UI ``action`` envelope is rewritten into the agent's turn."""

    async def test_action_click_executes_registered_tool(self) -> None:
        clicked: list[str] = []

        @a2ui_action(description="Schedule all posts for the given time")
        def submit(email: str) -> str:
            clicked.append(email)
            return "done"

        agent = Agent(
            name="ui_agent",
            tools=[submit],
            config=TestConfig(ToolCallEvent(name="submit", arguments='{"email": "user@example.com"}'), "All set."),
        )
        client = client_for(agent, validate_responses=False)

        reply = await client.ask("act", dependencies={EXTRA_PARTS_DEPENDENCY_KEY: create_a2ui_parts([ACTION_ENVELOPE])})

        assert clicked == ["user@example.com"]
        assert reply.response.content == "All set."

    async def test_action_context_reaches_the_model(self) -> None:
        config = CapturingConfig()

        @a2ui_action(description="Submit the form")
        def submit(email: str) -> str:
            return email

        # The ACTION_ENVELOPE clicks the "submit" button, which maps to this tool.
        agent = Agent(name="ui_agent", config=config, tools=[submit])
        client = client_for(agent, validate_responses=False)

        await client.ask("act", dependencies={EXTRA_PARTS_DEPENDENCY_KEY: create_a2ui_parts([ACTION_ENVELOPE])})

        synthesized = synthesized_text(config.messages[-1])
        assert "submit" in synthesized
        assert "user@example.com" in synthesized

    async def test_unregistered_action_becomes_generic_prompt(self) -> None:
        config = CapturingConfig()
        agent = Agent(name="ui_agent", config=config)  # no tool-backed buttons
        client = client_for(agent, validate_responses=False)

        await client.ask("act", dependencies={EXTRA_PARTS_DEPENDENCY_KEY: create_a2ui_parts([ACTION_ENVELOPE])})

        # No matching tool → the click is rewritten generically (not dropped):
        # the button name and its context still reach the model so the LLM can
        # react to the button it itself rendered.
        synthesized = synthesized_text(config.messages[-1])
        assert "clicked" in synthesized
        assert "submit" in synthesized
        assert "user@example.com" in synthesized


@pytest.mark.asyncio
class TestIncomingErrorRewrite:
    async def test_error_envelope_becomes_corrective_prompt(self) -> None:
        config = CapturingConfig()
        agent = Agent(name="ui_agent", config=config)
        client = client_for(agent, validate_responses=False)

        await client.ask("retry", dependencies={EXTRA_PARTS_DEPENDENCY_KEY: create_a2ui_parts([ERROR_ENVELOPE])})

        synthesized = synthesized_text(config.messages[-1])
        assert "VALIDATION_FAILED" in synthesized
        assert "/components/0" in synthesized


@pytest.mark.asyncio
class TestIncomingFunctionResponse:
    async def test_function_response_becomes_continuation_prompt(self) -> None:
        config = CapturingConfig()
        agent = Agent(name="ui_agent", config=config)
        client = client_for(agent, protocol_version="v1.0", validate_responses=False)

        await client.ask(
            "continue", dependencies={EXTRA_PARTS_DEPENDENCY_KEY: create_a2ui_parts([FUNCTION_RESPONSE_ENVELOPE])}
        )

        synthesized = synthesized_text(config.messages[-1])
        assert "openUrl" in synthesized
        assert "fc-1" in synthesized


@pytest.mark.asyncio
class TestCallFunctionPause:
    """A v1.0 ``callFunction(wantResponse=true)`` pauses the task awaiting the
    client's ``functionResponse`` instead of completing it, delivering the
    ``callFunction`` DataPart on the input-required transition."""

    async def test_call_function_with_want_response_pauses(self) -> None:
        hitl_prompts: list[str] = []

        async def hitl_hook() -> str:
            hitl_prompts.append("called")
            return FUNCTION_RESULT_MARK

        agent = Agent(name="ui_agent", config=CallFunctionThenComplete(_CALL_FUNCTION_BLOCK))
        client = client_for(agent, streaming=True, hitl_hook=hitl_hook, protocol_version="v1.0")
        stream = MemoryStream()
        payloads, states = subscribe_task_stream(stream)

        await client.ask("open the link", stream=stream)

        # The task paused for input rather than completing on the first turn, and
        # that pause surfaced to the client as an input request.
        assert TaskState.TASK_STATE_INPUT_REQUIRED in states
        assert hitl_prompts == ["called"]
        # The callFunction DataPart rode the input-required transition.
        assert any(payload and "callFunction" in payload[0] for payload in payloads)

    async def test_no_call_function_completes_normally(self) -> None:
        agent = Agent(name="ui_agent", config=TestConfig("Just a plain reply."))
        client = client_for(agent, streaming=True, protocol_version="v1.0")
        stream = MemoryStream()
        _payloads, states = subscribe_task_stream(stream)

        reply = await client.ask("hi", stream=stream)

        assert reply.response.content == "Just a plain reply."
        assert TaskState.TASK_STATE_COMPLETED in states
        assert TaskState.TASK_STATE_INPUT_REQUIRED not in states


@pytest.mark.asyncio
class TestCapabilitiesNegotiation:
    """Client capabilities advertised in message metadata fold into the turn's system prompt."""

    async def test_client_caps_injected_into_prompt(self) -> None:
        config = CapturingConfig()
        agent = Agent(name="ui_agent", config=config)
        caps = {"a2uiClientCapabilities": {VERSION: {"supportedCatalogIds": ["https://other.example/c.json"]}}}
        client = client_for(agent, interceptors=[MetadataInterceptor(caps)], validate_responses=False)

        await client.ask("hi")

        prompt = "\n".join(config.prompts[-1])
        assert "## A2UI Client Capabilities" in prompt
        assert "did NOT list" in prompt  # agent catalog absent from client's advertised list

    async def test_no_caps_no_negotiation_prompt(self) -> None:
        config = CapturingConfig()
        agent = Agent(name="ui_agent", config=config)
        client = client_for(agent, validate_responses=False)

        await client.ask("hi")

        assert "## A2UI Client Capabilities" not in "\n".join(config.prompts[-1])
