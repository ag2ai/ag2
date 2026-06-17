# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import pytest

from autogen.beta import Agent, Context
from autogen.beta.a2ui import A2UIMessageEvent, A2UIValidationFailedEvent
from autogen.beta.a2ui._runtime import _A2UIRuntime
from autogen.beta.a2ui.middleware import _to_prose_message
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


class _RecordingClient(LLMClient):
    """Wraps another client, snapshotting the full events list of each call."""

    def __init__(self, inner: LLMClient, calls: list[list[BaseEvent]]) -> None:
        self._inner = inner
        self._calls = calls

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self._calls.append(list(messages))
        return await self._inner(messages, context=context, **kwargs)


class _RecordingConfig(ModelConfig):
    """A ``ModelConfig`` that records every events list sent to the LLM."""

    def __init__(self, inner: ModelConfig, calls: list[list[BaseEvent]]) -> None:
        self._inner = inner
        self._calls = calls

    def copy(self) -> "_RecordingConfig":
        return self

    def create(self) -> _RecordingClient:
        return _RecordingClient(self._inner.create(), self._calls)

    def create_files_client(self) -> None:
        return None


VALID_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
    '"catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"}}]\n'
    "</a2ui-json>"
)

# Canonical A2UI wire (JSONL) inside the tag — one message per line.
VALID_RESPONSE_JSONL = (
    "Here is your UI.\n<a2ui-json>\n"
    '{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
    '"catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"}}\n'
    '{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}\n'
    "</a2ui-json>"
)

INVALID_RESPONSE_MISSING_CATALOG = (
    "Here.\n<a2ui-json>\n"
    '[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]\n'  # missing catalogId
    "</a2ui-json>"
)

INVALID_RESPONSE_BROKEN_JSON = "Text.\n<a2ui-json>\n{not valid json}\n</a2ui-json>"


def _build(
    config: Any,
    *,
    validate_responses: bool = True,
    validation_retries: int = 1,
) -> "tuple[Agent, _A2UIRuntime]":
    """Build a plain Agent plus its A2UI runtime (the new home of validation)."""
    agent = Agent(name="test_agent", config=config)
    rt = _A2UIRuntime(agent, validate_responses=validate_responses, validation_retries=validation_retries)
    return agent, rt


def _make_collecting_stream(events: list[A2UIMessageEvent]) -> MemoryStream:
    """Build a stream that records every A2UIMessageEvent into ``events``."""
    stream = MemoryStream()

    @stream.subscribe
    async def _record(event: BaseEvent) -> None:
        if isinstance(event, A2UIMessageEvent):
            events.append(event)

    return stream


@pytest.mark.asyncio()
class TestA2UIValidationMiddleware:
    async def test_valid_response_emits_events_and_strips_prose(self) -> None:
        agent, rt = _build(TestConfig(VALID_RESPONSE), validate_responses=True, validation_retries=2)
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        # Durable response keeps only the conversational prose.
        assert reply.body == "Here is your UI."
        # The A2UI message is carried out-of-band as a typed event.
        assert [e.message for e in events] == [
            {
                "version": "v0.9",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                },
            }
        ]

    async def test_jsonl_inside_tag_emits_one_event_per_message(self) -> None:
        agent, rt = _build(TestConfig(VALID_RESPONSE_JSONL), validate_responses=True, validation_retries=2)
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Here is your UI."
        assert [e.message for e in events] == [
            {
                "version": "v0.9",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                },
            },
            {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}},
        ]

    async def test_plain_text_bypasses_validation(self) -> None:
        agent, rt = _build(TestConfig("Just plain text."), validate_responses=True, validation_retries=2)
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Hi", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Just plain text."
        assert events == []

    async def test_retry_on_invalid_then_success(self) -> None:
        agent, rt = _build(
            TestConfig(INVALID_RESPONSE_MISSING_CATALOG, VALID_RESPONSE),
            validate_responses=True,
            validation_retries=1,
        )
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Here is your UI."
        # Event emitted only on the final successful attempt — never on retries.
        assert len(events) == 1

    async def test_retry_feeds_bad_assistant_turn_back_to_llm(self) -> None:
        # Regression: the retry must include the prior (invalid) assistant turn as
        # a ModelResponse so the LLM can see what to fix. A bare ModelMessage is
        # silently dropped by provider mappers, defeating the correction.
        calls: list[list[BaseEvent]] = []
        agent, rt = _build(
            _RecordingConfig(TestConfig(INVALID_RESPONSE_MISSING_CATALOG, VALID_RESPONSE), calls),
            validate_responses=True,
            validation_retries=1,
        )
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Here is your UI."
        assert len(calls) == 2  # the turn was retried
        retry_events = calls[1]
        assert any(
            isinstance(e, ModelResponse) and e.message is not None and "createSurface" in (e.message.content or "")
            for e in retry_events
        )

    async def test_retry_exhausted_returns_text_only(self) -> None:
        agent, rt = _build(
            TestConfig(
                INVALID_RESPONSE_MISSING_CATALOG,
                INVALID_RESPONSE_MISSING_CATALOG,
                INVALID_RESPONSE_MISSING_CATALOG,
            ),
            validate_responses=True,
            validation_retries=2,
        )
        events: list[A2UIMessageEvent] = []
        stream = _make_collecting_stream(events)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        # graceful degradation: prose only, no A2UI tag, no events.
        assert reply.body == "Here."
        assert events == []

    async def test_retry_exhausted_emits_validation_failed_event(self) -> None:
        # The wire degrades to prose, but an internal observability event is
        # emitted so monitoring can tell a failed UI from an intentional text reply.
        agent, rt = _build(
            TestConfig(
                INVALID_RESPONSE_MISSING_CATALOG,
                INVALID_RESPONSE_MISSING_CATALOG,
            ),
            validate_responses=True,
            validation_retries=1,
        )
        failures: list[A2UIValidationFailedEvent] = []
        stream = MemoryStream()

        @stream.subscribe
        async def _record(event: BaseEvent) -> None:
            if isinstance(event, A2UIValidationFailedEvent):
                failures.append(event)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Here."
        assert len(failures) == 1
        # validation_retries=1 → 2 total attempts, and the errors are surfaced.
        assert failures[0].attempts == 2
        assert failures[0].errors

    async def test_no_validation_failed_event_on_success(self) -> None:
        agent, rt = _build(
            TestConfig(INVALID_RESPONSE_MISSING_CATALOG, VALID_RESPONSE),
            validate_responses=True,
            validation_retries=1,
        )
        failures: list[A2UIValidationFailedEvent] = []
        stream = MemoryStream()

        @stream.subscribe
        async def _record(event: BaseEvent) -> None:
            if isinstance(event, A2UIValidationFailedEvent):
                failures.append(event)

        reply = await agent.ask(
            "Show UI", stream=stream, middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section]
        )

        assert reply.body == "Here is your UI."
        assert failures == []

    async def test_json_parse_error_triggers_retry(self) -> None:
        agent, rt = _build(
            TestConfig(INVALID_RESPONSE_BROKEN_JSON, VALID_RESPONSE),
            validate_responses=True,
            validation_retries=1,
        )
        reply = await agent.ask("Show UI", middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section])
        assert reply.body == "Here is your UI."

    async def test_no_retry_when_validation_disabled(self) -> None:
        # Invalid response should pass through unchanged when validation is off.
        agent, rt = _build(TestConfig(INVALID_RESPONSE_MISSING_CATALOG), validate_responses=False)
        reply = await agent.ask("Show UI", middleware=rt.middleware_factories(), prompt=[rt.system_prompt_section])
        assert reply.body == INVALID_RESPONSE_MISSING_CATALOG


class TestToProseMessage:
    def test_preserves_original_metadata(self) -> None:
        original = ModelMessage("full text", metadata={"trace_id": "abc", "provider": "x"})
        prose = _to_prose_message(original, "text only")
        assert prose.content == "text only"
        assert prose.metadata == {"trace_id": "abc", "provider": "x"}

    def test_handles_none_message(self) -> None:
        prose = _to_prose_message(None, "text only")
        assert prose.content == "text only"
        assert prose.metadata == {}
