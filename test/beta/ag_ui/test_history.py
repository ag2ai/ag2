# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from ag_ui.core import AssistantMessage, FunctionCall, ToolCall, ToolMessage, UserMessage
from anthropic.types import (
    ServerToolUseBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
)
from google.genai import types
from openai.types.responses import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionSearch

from autogen.beta.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from autogen.beta.config.anthropic.events import (
    AnthropicServerToolCallEvent,
    AnthropicServerToolResultEvent,
)
from autogen.beta.config.gemini.events import GeminiServerToolCallEvent
from autogen.beta.config.openai.events import (
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
)
from autogen.beta.events import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

from .utils import create_run_input


def _map(messages: list) -> list:
    command = AGStreamInput(incoming=create_run_input(*messages), variables={})
    _, history = map_agui_messages_to_events(command)
    return history


class TestBuiltinRestoredFromHistory:
    def test_openai_web_search_restored(self) -> None:
        item = ResponseFunctionWebSearch(
            id="ws_1",
            action=ActionSearch(type="search", query="bitcoin"),
            status="completed",
            type="web_search_call",
        )
        wire_args = item.action.model_dump_json(warnings=False)

        history = _map([
            UserMessage(id="u1", content="search for bitcoin"),
            AssistantMessage(
                id="a1",
                content="searched",
                tool_calls=[
                    ToolCall(
                        id="ws_1",
                        type="function",
                        function=FunctionCall(name=WEB_SEARCH_TOOL_NAME, arguments=wire_args),
                    ),
                ],
            ),
        ])

        assert history == [
            ModelRequest([TextInput("search for bitcoin")]),
            ModelResponse(
                message=ModelMessage("searched"),
                tool_calls=ToolCallsEvent([
                    OpenAIServerToolCallEvent(
                        id="ws_1",
                        name=WEB_SEARCH_TOOL_NAME,
                        arguments=wire_args,
                        item=item,
                    ),
                ]),
            ),
        ]

    def test_anthropic_code_execution_restored_with_kind(self) -> None:
        wire_args = json.dumps({"_kind": "bash_code_execution", "cmd": "ls"})

        history = _map([
            UserMessage(id="u1", content="run ls"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="b1",
                        type="function",
                        function=FunctionCall(name=CODE_EXECUTION_TOOL_NAME, arguments=wire_args),
                    ),
                ],
            ),
        ])

        expected_block = ServerToolUseBlock(
            id="b1",
            name="bash_code_execution",
            input={"cmd": "ls"},
            type="server_tool_use",
        )
        assert history == [
            ModelRequest([TextInput("run ls")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    AnthropicServerToolCallEvent(
                        id="b1",
                        name=CODE_EXECUTION_TOOL_NAME,
                        arguments=wire_args,
                        block=expected_block,
                    ),
                ]),
            ),
        ]

    def test_gemini_executable_code_restored(self) -> None:
        wire_args = json.dumps({"code": "print(1)", "language": "PYTHON"})

        history = _map([
            UserMessage(id="u1", content="run code"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="g1",
                        type="function",
                        function=FunctionCall(name=CODE_EXECUTION_TOOL_NAME, arguments=wire_args),
                    ),
                ],
            ),
        ])

        expected_part = types.Part(
            executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON)
        )
        assert history == [
            ModelRequest([TextInput("run code")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    GeminiServerToolCallEvent(
                        id="g1",
                        name=CODE_EXECUTION_TOOL_NAME,
                        arguments=wire_args,
                        part=expected_part,
                    ),
                ]),
            ),
        ]


class TestFallback:
    def test_unknown_function_tool_stays_plain_silently(self) -> None:
        history = _map([
            UserMessage(id="u1", content="multiply"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        type="function",
                        function=FunctionCall(name="multiply", arguments='{"a": 2}'),
                    ),
                ],
            ),
        ])

        assert history == [
            ModelRequest([TextInput("multiply")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="tc_1", name="multiply", arguments='{"a": 2}'),
                ]),
            ),
        ]

    def test_known_builtin_with_broken_args_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # web_search with payload that does not match any known SDK shape
        # (no `type`, no `query`, no `queries`) → mapper returns None → fallback.
        wire_args = json.dumps({"random": "stuff"})

        with caplog.at_level(logging.WARNING, logger="autogen.beta.ag_ui.stream"):
            history = _map([
                UserMessage(id="u1", content="search"),
                AssistantMessage(
                    id="a1",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="ws_x",
                            type="function",
                            function=FunctionCall(name=WEB_SEARCH_TOOL_NAME, arguments=wire_args),
                        ),
                    ],
                ),
            ])

        assert history == [
            ModelRequest([TextInput("search")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="ws_x", name=WEB_SEARCH_TOOL_NAME, arguments=wire_args),
                ]),
            ),
        ]
        assert "builtin mapper failed" in caplog.text


class TestBuiltinResultsRestoredFromHistory:
    def test_openai_web_search_result_paired_with_call(self) -> None:
        item = ResponseFunctionWebSearch(
            id="ws_1",
            action=ActionSearch(type="search", query="bitcoin"),
            status="completed",
            type="web_search_call",
        )
        wire_args = item.action.model_dump_json(warnings=False)

        history = _map([
            UserMessage(id="u1", content="search"),
            AssistantMessage(
                id="a1",
                content="searched",
                tool_calls=[
                    ToolCall(
                        id="ws_1",
                        type="function",
                        function=FunctionCall(name=WEB_SEARCH_TOOL_NAME, arguments=wire_args),
                    ),
                ],
            ),
            ToolMessage(id="t1", tool_call_id="ws_1", content="https://example.com"),
        ])

        assert history == [
            ModelRequest([TextInput("search")]),
            ModelResponse(
                message=ModelMessage("searched"),
                tool_calls=ToolCallsEvent([
                    OpenAIServerToolCallEvent(
                        id="ws_1",
                        name=WEB_SEARCH_TOOL_NAME,
                        arguments=wire_args,
                        item=item,
                    ),
                ]),
            ),
            ToolResultsEvent([
                OpenAIServerToolResultEvent(
                    parent_id="ws_1",
                    name=WEB_SEARCH_TOOL_NAME,
                    result=ToolResult(TextInput("https://example.com")),
                ),
            ]),
        ]

    def test_anthropic_web_search_result_synthesizes_block(self) -> None:
        wire_args = json.dumps({"query": "bitcoin"})

        history = _map([
            UserMessage(id="u1", content="search"),
            AssistantMessage(
                id="a1",
                content="searched",
                tool_calls=[
                    ToolCall(
                        id="ws_1",
                        type="function",
                        function=FunctionCall(name=WEB_SEARCH_TOOL_NAME, arguments=wire_args),
                    ),
                ],
            ),
            ToolMessage(id="t1", tool_call_id="ws_1", content="https://a\nhttps://b"),
        ])

        expected_call_block = ServerToolUseBlock(
            id="ws_1",
            name="web_search",
            input={"query": "bitcoin"},
            type="server_tool_use",
        )
        expected_result_block = WebSearchToolResultBlock(
            tool_use_id="ws_1",
            type="web_search_tool_result",
            content=[
                WebSearchResultBlock(
                    url="https://a",
                    title="",
                    encrypted_content="",
                    page_age=None,
                    type="web_search_result",
                ),
                WebSearchResultBlock(
                    url="https://b",
                    title="",
                    encrypted_content="",
                    page_age=None,
                    type="web_search_result",
                ),
            ],
        )
        assert history == [
            ModelRequest([TextInput("search")]),
            ModelResponse(
                message=ModelMessage("searched"),
                tool_calls=ToolCallsEvent([
                    AnthropicServerToolCallEvent(
                        id="ws_1",
                        name=WEB_SEARCH_TOOL_NAME,
                        arguments=wire_args,
                        block=expected_call_block,
                    ),
                ]),
            ),
            ToolResultsEvent([
                AnthropicServerToolResultEvent(
                    parent_id="ws_1",
                    name=WEB_SEARCH_TOOL_NAME,
                    result=ToolResult(TextInput("https://a\nhttps://b")),
                    block=expected_result_block,
                ),
            ]),
        ]


class TestPlainFunctionToolResultUnchanged:
    def test_non_builtin_result_stays_plain(self) -> None:
        history = _map([
            UserMessage(id="u1", content="run"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="my_func", arguments="{}"),
                    ),
                ],
            ),
            ToolMessage(id="t1", tool_call_id="call_1", content="42"),
        ])

        assert history == [
            ModelRequest([TextInput("run")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="call_1", name="my_func", arguments="{}"),
                ]),
            ),
            ToolResultsEvent([
                ToolResultEvent(parent_id="call_1", result=ToolResult(TextInput("42"))),
            ]),
        ]


class TestInputBufferAndToolResult:
    """Regression tests for bugs 2 (input_buffer aliasing) and 3 (ToolResult([str]))."""

    def test_input_buffer_preserves_earlier_user_messages(self) -> None:
        history = _map([
            UserMessage(id="u1", content="FIRST"),
            AssistantMessage(id="a1", content="ok"),
            UserMessage(id="u2", content="SECOND"),
            AssistantMessage(id="a2", content="hello"),
            UserMessage(id="u3", content="THIRD"),
        ])

        assert history == [
            ModelRequest([TextInput("FIRST")]),
            ModelResponse(message=ModelMessage("ok"), tool_calls=ToolCallsEvent([])),
            ModelRequest([TextInput("SECOND")]),
            ModelResponse(message=ModelMessage("hello"), tool_calls=ToolCallsEvent([])),
            ModelRequest([TextInput("THIRD")]),
        ]

    def test_tool_result_text_content_is_textinput(self) -> None:
        history = _map([
            UserMessage(id="u1", content="run"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="my_func", arguments="{}"),
                    ),
                ],
            ),
            ToolMessage(id="t1", tool_call_id="call_1", content="bare string"),
        ])

        assert history == [
            ModelRequest([TextInput("run")]),
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent([
                    ToolCallEvent(id="call_1", name="my_func", arguments="{}"),
                ]),
            ),
            ToolResultsEvent([
                ToolResultEvent(parent_id="call_1", result=ToolResult(TextInput("bare string"))),
            ]),
        ]
