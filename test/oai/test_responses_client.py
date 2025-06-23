# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


"""Unit-tests for the OpenAIResponsesClient abstraction.

These tests are self-contained—they DO NOT call the real OpenAI
endpoint. Instead we mock the `openai.OpenAI` instance and capture the
kwargs passed to `client.responses.create` / `client.responses.parse`.

We follow the style of existing tests in *test/oai/test_client.py*.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autogen.oai.client import OpenAIResponsesClient

# -----------------------------------------------------------------------------
# Helper fakes
# -----------------------------------------------------------------------------


class _FakeUsage:
    """Mimics the `.usage` member on an OpenAI Response object."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):  # type: ignore[override]
        return self._fields


class _FakeResponse:
    """Minimal object returned by mocked `.responses.create`"""

    def __init__(self, *, output=None, usage=None):
        self.output = output or []
        self.usage = usage or {}
        self.cost = 1.23  # arbitrary
        self.model = "gpt-4o"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def mocked_openai_client():
    """Return a fake `OpenAI` instance with stubbed `.responses` interface."""

    mock_client = MagicMock()
    mock_responses = MagicMock()
    mock_client.responses = mock_responses  # attach

    # By default `.create` returns an empty fake response; tests can overwrite.
    mock_responses.create.return_value = _FakeResponse()
    mock_responses.parse.return_value = _FakeResponse()

    return mock_client


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_messages_are_transformed_into_input(mocked_openai_client):
    """`messages=[…]` should be converted into `input=[{{type:'message',…}}]`."""

    client = OpenAIResponsesClient(mocked_openai_client)

    messages_param = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    client.create({"messages": messages_param})

    # capture the kwargs actually sent to mocked .responses.create
    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    assert "messages" not in kwargs, "messages should have been popped"
    assert "input" in kwargs, "input should be present after conversion"

    # the first converted item should reflect original content
    first_item = kwargs["input"][0]
    assert first_item["role"] == "user"
    assert first_item["content"][0]["text"] == "Hello"


def test_structured_output_path_uses_parse(mocked_openai_client):
    """When `response_format` / `text_format` is supplied the client should call
    `.responses.parse` instead of `.responses.create` and inject the correct
    `text_format` payload."""

    response_format_schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": response_format_schema,
    })

    # The parse method should have been invoked
    assert mocked_openai_client.responses.parse.called, "parse() must be used for structured output"

    # verify `text_format` kwarg exists
    kwargs = mocked_openai_client.responses.parse.call_args.kwargs
    assert "text_format" in kwargs


def test_usage_dict_parses_pydantic_like_object():
    usage_obj = _FakeUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    resp = _FakeResponse(usage=usage_obj)
    client = OpenAIResponsesClient(MagicMock())

    usage = client._usage_dict(resp)

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["cost"] == 1.23
    assert usage["model"] == "gpt-4o"


def test_message_retrieval_handles_various_item_types():
    # fake pydantic-like blocks
    class _Block:
        def __init__(self, d):
            self._d = d

        def model_dump(self):
            return self._d

    output = [
        _FakeResponse(output=[{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]).output[0],
        {"type": "function_call", "name": "foo", "arguments": "{}"},
        {"type": "web_search_call", "id": "abc", "arguments": {}},
    ]

    # Wrap dicts into objects providing model_dump to test conversion path
    output_wrapped = [_Block(o) if isinstance(o, dict) else o for o in output]

    resp = _FakeResponse(output=output_wrapped)
    client = OpenAIResponsesClient(MagicMock())

    msgs = client.message_retrieval(resp)

    # Should have produced three message entries
    assert len(msgs) == 3
    # first is plain string
    assert msgs[0] == "Hi"
    # second is a function_call dict
    assert msgs[1]["function_call"]["name"] == "foo"
    # third contains tool_calls list
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "web_search"
