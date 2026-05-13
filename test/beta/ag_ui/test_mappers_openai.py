# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

pytest.importorskip("openai")

from openai.types.responses import ResponseCodeInterpreterToolCall, ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import (
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    ActionSearchSource,
)
from openai.types.responses.response_output_item import ImageGenerationCall

from autogen.beta.ag_ui.mappers import call_from_agui
from autogen.beta.ag_ui.mappers.openai import openai_call_from_agui
from autogen.beta.config.openai.events import OpenAIServerToolCallEvent
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


class TestWebSearch:
    def test_search_action_round_trips(self) -> None:
        item = ResponseFunctionWebSearch(
            id="ws_1",
            action=ActionSearch(
                type="search",
                query="bitcoin price",
                sources=[
                    ActionSearchSource(type="url", url="https://a.example"),
                    ActionSearchSource(type="url", url="https://b.example"),
                ],
            ),
            status="completed",
            type="web_search_call",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward

    def test_open_page_action_round_trips(self) -> None:
        item = ResponseFunctionWebSearch(
            id="ws_2",
            action=ActionOpenPage(type="open_page", url="https://example.com"),
            status="completed",
            type="web_search_call",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward

    def test_find_action_round_trips(self) -> None:
        item = ResponseFunctionWebSearch(
            id="ws_3",
            action=ActionFind(type="find", url="https://example.com", pattern="needle"),
            status="completed",
            type="web_search_call",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestCodeInterpreter:
    def test_with_code_round_trips(self) -> None:
        item = ResponseCodeInterpreterToolCall(
            id="ci_1",
            code="print('hi')",
            container_id="c_42",
            outputs=None,
            status="completed",
            type="code_interpreter_call",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward

    def test_without_code_round_trips(self) -> None:
        # `code` is optional in the SDK; container_id alone must be enough.
        item = ResponseCodeInterpreterToolCall(
            id="ci_2",
            code=None,
            container_id="c_99",
            outputs=None,
            status="completed",
            type="code_interpreter_call",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestImageGeneration:
    def test_round_trips_minimal(self) -> None:
        # image_generation result lives in a separate AG-UI ToolMessage and is
        # not serialized into arguments, so the round-trip restores the minimal
        # SDK shape (id + type + completed status).
        item = ImageGenerationCall(
            id="ig_1",
            status="completed",
            type="image_generation_call",
            result="YWJj",
            revised_prompt=None,
            output_format="png",
        )

        forward = OpenAIServerToolCallEvent.from_item(item)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == OpenAIServerToolCallEvent(
            id="ig_1",
            name=IMAGE_GENERATION_TOOL_NAME,
            arguments="",
            item=ImageGenerationCall(
                id="ig_1",
                status="completed",
                type="image_generation_call",
            ),
        )


class TestNonMatching:
    def test_unknown_name_returns_none(self) -> None:
        assert openai_call_from_agui("my_calculator", "call_1", {"x": 1}) is None

    def test_web_search_with_alien_action_type_returns_none(self) -> None:
        # Forward-compat: unknown action type → mapper returns None, caller
        # falls back to plain ToolCallEvent.
        assert openai_call_from_agui(WEB_SEARCH_TOOL_NAME, "ws_x", {"type": "telepathy"}) is None

    def test_code_interpreter_missing_container_id_returns_none(self) -> None:
        # Without container_id the SDK pydantic cannot validate.
        assert openai_call_from_agui(CODE_EXECUTION_TOOL_NAME, "ci_x", {"code": "x = 1"}) is None

    def test_call_from_agui_garbage_arguments_returns_none(self) -> None:
        assert call_from_agui(WEB_SEARCH_TOOL_NAME, "ws_x", "not-json") is None
        assert call_from_agui(WEB_SEARCH_TOOL_NAME, "ws_x", json.dumps([1, 2])) is None
