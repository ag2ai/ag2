# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.gemini.mappers import convert_messages
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
        result = convert_messages([_model_response_with_tool_call(arguments)])

        assert len(result) == 1
        part = result[0].parts[0]
        assert part.function_call is not None
        assert part.function_call.args == {}

    def test_valid_arguments_are_preserved(self) -> None:
        result = convert_messages([_model_response_with_tool_call('{"category": "books"}')])

        part = result[0].parts[0]
        assert part.function_call.args == {"category": "books"}
