# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from fast_depends.pydantic import PydanticSerializer
from google.genai import types

from ag2 import ToolResult
from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.config.gemini.mappers import SKIP_THOUGHT_SIGNATURE, convert_messages
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)

REAL_SIGNATURE = b"real-signature"
SERIALIZER = PydanticSerializer()


def _step(*calls: ToolCallEvent) -> list[BaseEvent]:
    response = ModelResponse(message=None, tool_calls=ToolCallsEvent(calls=list(calls)))
    results = ToolResultsEvent(results=[ToolResultEvent.from_call(call, ToolResult("ok")) for call in calls])
    return [response, results]


def _call(call_id: str, signature: bytes | None = None) -> GeminiToolCallEvent:
    return GeminiToolCallEvent(id=call_id, name="search", arguments="{}", thought_signature=signature)


def _signatures(contents: list[types.Content]) -> dict[str, bytes | None]:
    return {
        part.function_call.id or "": part.thought_signature
        for content in contents
        for part in content.parts or ()
        if part.function_call is not None
    }


def test_current_turn_call_without_signature_gets_placeholder(caplog: pytest.LogCaptureFixture) -> None:
    messages = [ModelRequest([TextInput("Find it.")]), *_step(_call("call-1"))]

    with caplog.at_level(logging.WARNING, logger="ag2.config.gemini.mappers"):
        contents = convert_messages(messages, SERIALIZER)

    assert _signatures(contents) == {"call-1": SKIP_THOUGHT_SIGNATURE}
    [record] = caplog.records
    assert "'search'" in record.getMessage()
    assert "call-1" in record.getMessage()


def test_non_gemini_call_in_current_turn_gets_placeholder() -> None:
    plain = ToolCallEvent(id="call-1", name="search", arguments="{}")
    messages = [ModelRequest([TextInput("Find it.")]), *_step(plain)]

    assert _signatures(convert_messages(messages, SERIALIZER)) == {"call-1": SKIP_THOUGHT_SIGNATURE}


def test_existing_signature_is_kept(caplog: pytest.LogCaptureFixture) -> None:
    messages = [ModelRequest([TextInput("Find it.")]), *_step(_call("call-1", REAL_SIGNATURE))]

    with caplog.at_level(logging.WARNING, logger="ag2.config.gemini.mappers"):
        contents = convert_messages(messages, SERIALIZER)

    assert _signatures(contents) == {"call-1": REAL_SIGNATURE}
    assert caplog.records == []


def test_older_turn_call_is_left_unsigned() -> None:
    messages = [
        ModelRequest([TextInput("First question.")]),
        *_step(_call("old-call")),
        ModelResponse(message=ModelMessage("Nothing found."), tool_calls=ToolCallsEvent(calls=[])),
        ModelRequest([TextInput("Second question.")]),
        *_step(_call("new-call", REAL_SIGNATURE)),
    ]

    assert _signatures(convert_messages(messages, SERIALIZER)) == {
        "old-call": None,
        "new-call": REAL_SIGNATURE,
    }


def test_each_current_turn_step_is_signed_on_its_first_call_only() -> None:
    messages = [
        ModelRequest([TextInput("Find both.")]),
        *_step(_call("step1-a"), _call("step1-b")),
        *_step(_call("step2-a", REAL_SIGNATURE), _call("step2-b")),
        *_step(_call("step3-a")),
    ]

    assert _signatures(convert_messages(messages, SERIALIZER)) == {
        "step1-a": SKIP_THOUGHT_SIGNATURE,
        "step1-b": None,
        "step2-a": REAL_SIGNATURE,
        "step2-b": None,
        "step3-a": SKIP_THOUGHT_SIGNATURE,
    }
