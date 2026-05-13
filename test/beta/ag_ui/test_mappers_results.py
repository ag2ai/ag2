# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from anthropic.types import (
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    CodeExecutionResultBlock,
    CodeExecutionToolResultBlock,
    DocumentBlock,
    PlainTextSource,
    ServerToolUseBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionViewResultBlock,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
)
from google.genai import types
from openai.types.responses import ResponseCodeInterpreterToolCall, ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionSearch

from autogen.beta.ag_ui.mappers import result_from_agui
from autogen.beta.config.anthropic.events import AnthropicServerToolCallEvent, AnthropicServerToolResultEvent
from autogen.beta.config.gemini.events import GeminiServerToolCallEvent, GeminiServerToolResultEvent
from autogen.beta.config.openai.events import OpenAIServerToolCallEvent, OpenAIServerToolResultEvent
from autogen.beta.events import TextInput, ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


class TestOpenAIResults:
    def test_web_search_result(self) -> None:
        call = OpenAIServerToolCallEvent.from_item(
            ResponseFunctionWebSearch(
                id="ws_1",
                action=ActionSearch(type="search", query="bitcoin"),
                status="completed",
                type="web_search_call",
            )
        )

        result = result_from_agui(call, "https://example.com\nhttps://other.example")

        assert result == OpenAIServerToolResultEvent(
            parent_id="ws_1",
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(TextInput("https://example.com\nhttps://other.example")),
        )

    def test_code_interpreter_result(self) -> None:
        call = OpenAIServerToolCallEvent.from_item(
            ResponseCodeInterpreterToolCall(
                id="ci_1",
                code="print(1)",
                container_id="c_1",
                outputs=None,
                status="completed",
                type="code_interpreter_call",
            )
        )

        result = result_from_agui(call, "1\n")

        assert result == OpenAIServerToolResultEvent(
            parent_id="ci_1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(TextInput("1\n")),
        )


class TestAnthropicResults:
    def test_web_search_synthesizes_url_blocks(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(id="w1", name="web_search", input={"query": "x"}, type="server_tool_use")
        )

        result = result_from_agui(call, "https://a.example\nhttps://b.example\nnot-a-url")

        assert result == AnthropicServerToolResultEvent(
            parent_id="w1",
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(TextInput("https://a.example\nhttps://b.example\nnot-a-url")),
            block=WebSearchToolResultBlock(
                tool_use_id="w1",
                type="web_search_tool_result",
                content=[
                    WebSearchResultBlock(
                        url="https://a.example",
                        title="",
                        encrypted_content="",
                        page_age=None,
                        type="web_search_result",
                    ),
                    WebSearchResultBlock(
                        url="https://b.example",
                        title="",
                        encrypted_content="",
                        page_age=None,
                        type="web_search_result",
                    ),
                ],
            ),
        )

    def test_web_search_empty_content(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(id="w2", name="web_search", input={"query": "x"}, type="server_tool_use")
        )

        result = result_from_agui(call, "")

        assert result == AnthropicServerToolResultEvent(
            parent_id="w2",
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(),
            block=WebSearchToolResultBlock(
                tool_use_id="w2",
                type="web_search_tool_result",
                content=[],
            ),
        )

    def test_web_fetch_wraps_text_in_document(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(
                id="f1",
                name="web_fetch",
                input={"url": "https://x"},
                type="server_tool_use",
            )
        )

        result = result_from_agui(call, "page body")

        assert result == AnthropicServerToolResultEvent(
            parent_id="f1",
            name=WEB_FETCH_TOOL_NAME,
            result=ToolResult(TextInput("page body")),
            block=WebFetchToolResultBlock(
                tool_use_id="f1",
                type="web_fetch_tool_result",
                content=WebFetchBlock(
                    url="",
                    retrieved_at=None,
                    type="web_fetch_result",
                    content=DocumentBlock(
                        source=PlainTextSource(data="page body", media_type="text/plain", type="text"),
                        title=None,
                        type="document",
                        citations=None,
                    ),
                ),
            ),
        )

    def test_code_execution_synthesizes_result_block(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(
                id="c1",
                name="code_execution",
                input={"code": "print(1)"},
                type="server_tool_use",
            )
        )

        result = result_from_agui(call, "1\n")

        assert result == AnthropicServerToolResultEvent(
            parent_id="c1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(TextInput("1\n")),
            block=CodeExecutionToolResultBlock(
                tool_use_id="c1",
                type="code_execution_tool_result",
                content=CodeExecutionResultBlock(
                    content=[],
                    return_code=0,
                    stderr="",
                    stdout="1\n",
                    type="code_execution_result",
                ),
            ),
        )

    def test_bash_code_execution_synthesizes_result_block(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(
                id="b1",
                name="bash_code_execution",
                input={"cmd": "ls"},
                type="server_tool_use",
            )
        )

        result = result_from_agui(call, "file.txt\n")

        assert result == AnthropicServerToolResultEvent(
            parent_id="b1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(TextInput("file.txt\n")),
            block=BashCodeExecutionToolResultBlock(
                tool_use_id="b1",
                type="bash_code_execution_tool_result",
                content=BashCodeExecutionResultBlock(
                    content=[],
                    return_code=0,
                    stderr="",
                    stdout="file.txt\n",
                    type="bash_code_execution_result",
                ),
            ),
        )

    def test_text_editor_code_execution_synthesizes_view_block(self) -> None:
        call = AnthropicServerToolCallEvent.from_block(
            ServerToolUseBlock(
                id="t1",
                name="text_editor_code_execution",
                input={"command": "view"},
                type="server_tool_use",
            )
        )

        result = result_from_agui(call, "line1\nline2")

        assert result == AnthropicServerToolResultEvent(
            parent_id="t1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(TextInput("line1\nline2")),
            block=TextEditorCodeExecutionToolResultBlock(
                tool_use_id="t1",
                type="text_editor_code_execution_tool_result",
                content=TextEditorCodeExecutionViewResultBlock(
                    content="line1\nline2",
                    file_type="text",
                    num_lines=2,
                    start_line=1,
                    total_lines=2,
                    type="text_editor_code_execution_view_result",
                ),
            ),
        )


class TestGeminiResults:
    def test_executable_code_result(self) -> None:
        part = types.Part(executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON))
        call = GeminiServerToolCallEvent.from_executable_code(part)

        result = result_from_agui(call, "1\n")

        assert result == GeminiServerToolResultEvent(
            parent_id=call.id,
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(TextInput("1\n")),
            part=types.Part(
                code_execution_result=types.CodeExecutionResult(
                    output="1\n",
                    outcome=types.Outcome.OUTCOME_OK,
                )
            ),
        )

    def test_grounding_result_reuses_call_metadata(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["bitcoin"])
        call = GeminiServerToolCallEvent.from_grounding(gm, name=WEB_SEARCH_TOOL_NAME)

        result = result_from_agui(call, "https://example.com")

        assert result == GeminiServerToolResultEvent(
            parent_id=call.id,
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(TextInput("https://example.com")),
            grounding_metadata=gm,
        )
