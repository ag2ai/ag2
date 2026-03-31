# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.ollama.mappers import convert_messages
from autogen.beta.events import ModelResponse
from autogen.beta.events.tool_events import ToolCallEvent, ToolCallsEvent


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


class TestConvertMessagesEmptyArguments:
    """json.loads must not crash on empty or None tool call arguments."""

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_produce_empty_dict(self, arguments: str | None) -> None:
        result = convert_messages([], [_model_response_with_tool_call(arguments)])

        assistant_msg = result[0]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["tool_calls"][0]["function"]["arguments"] == {}

    def test_valid_arguments_are_preserved(self) -> None:
        result = convert_messages([], [_model_response_with_tool_call('{"category": "books"}')])

        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"category": "books"}
