# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

import pytest
from anthropic.types import (
    Base64PDFSource,
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    CodeExecutionResultBlock,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultError,
    EncryptedCodeExecutionResultBlock,
    Message,
    PlainTextSource,
    ServerToolUseBlock,
    TextBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultError,
    TextEditorCodeExecutionViewResultBlock,
    ToolUseBlock,
    Usage,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)
from anthropic.types.bash_code_execution_output_block import BashCodeExecutionOutputBlock
from anthropic.types.bash_code_execution_tool_result_error import BashCodeExecutionToolResultError
from anthropic.types.code_execution_output_block import CodeExecutionOutputBlock
from anthropic.types.document_block import DocumentBlock

from autogen.beta import MemoryStream
from autogen.beta.config.anthropic import AnthropicClient
from autogen.beta.config.anthropic.events import AnthropicServerToolCallEvent, AnthropicServerToolResultEvent
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    FileIdInput,
    ModelMessage,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    UrlInput,
)
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


async def _process(content: Iterable[Any]) -> tuple[ModelResponse, list[BaseEvent]]:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=list(content),
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    response = await client._process_response(message, context)
    return response, list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_process_response_routes_all_block_types() -> None:
    web_call = ServerToolUseBlock(id="w1", name="web_search", input={"q": "x"}, type="server_tool_use")
    web_result = WebSearchToolResultBlock(tool_use_id="w1", type="web_search_tool_result", content=[])
    bash_call = ServerToolUseBlock(id="b1", name="bash_code_execution", input={"cmd": "ls"}, type="server_tool_use")
    bash_result = BashCodeExecutionToolResultBlock(
        tool_use_id="b1",
        type="bash_code_execution_tool_result",
        content=BashCodeExecutionToolResultError(
            error_code="unavailable",
            type="bash_code_execution_tool_result_error",
        ),
    )
    user_tool = ToolUseBlock(id="tc_1", name="my_func", input={"x": 1}, type="tool_use")

    response, events = await _process([
        TextBlock(text="Searching...", type="text"),
        web_call,
        web_result,
        bash_call,
        bash_result,
        user_tool,
    ])

    assert response.message == ModelMessage("Searching...")
    assert response.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="tc_1", name="my_func", arguments='{"x": 1}'),
    ])
    assert events == [
        AnthropicServerToolCallEvent(id="w1", name=WEB_SEARCH_TOOL_NAME, arguments='{"q": "x"}', block=web_call),
        AnthropicServerToolResultEvent(
            parent_id="w1",
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(metadata={"count": 0}),
            block=web_result,
        ),
        AnthropicServerToolCallEvent(
            id="b1", name=CODE_EXECUTION_TOOL_NAME, arguments='{"cmd": "ls"}', block=bash_call
        ),
        AnthropicServerToolResultEvent(
            parent_id="b1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(
                TextInput("bash_code_execution_tool_result_error: unavailable"),
                metadata={
                    "error": True,
                    "error_code": "unavailable",
                    "type": "bash_code_execution_tool_result_error",
                },
            ),
            block=bash_result,
        ),
    ]


@pytest.mark.asyncio
class TestResultParts:
    async def test_web_search_success_url_inputs(self) -> None:
        call = ServerToolUseBlock(id="w1", name="web_search", input={}, type="server_tool_use")
        hits = [
            WebSearchResultBlock(
                url="https://a", title="A", encrypted_content="", page_age="1d", type="web_search_result"
            ),
            WebSearchResultBlock(
                url="https://b", title="B", encrypted_content="", page_age=None, type="web_search_result"
            ),
        ]
        result = WebSearchToolResultBlock(tool_use_id="w1", type="web_search_tool_result", content=hits)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="w1", name=WEB_SEARCH_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="w1",
                name=WEB_SEARCH_TOOL_NAME,
                result=ToolResult(
                    UrlInput("https://a", kind=BinaryType.BINARY, metadata={"title": "A", "page_age": "1d"}),
                    UrlInput("https://b", kind=BinaryType.BINARY, metadata={"title": "B", "page_age": None}),
                    metadata={"count": 2},
                ),
                block=result,
            ),
        ]

    async def test_web_search_error_text_input(self) -> None:
        call = ServerToolUseBlock(id="w1", name="web_search", input={}, type="server_tool_use")
        err = WebSearchToolResultError(error_code="too_many_requests", type="web_search_tool_result_error")
        result = WebSearchToolResultBlock(tool_use_id="w1", type="web_search_tool_result", content=err)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="w1", name=WEB_SEARCH_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="w1",
                name=WEB_SEARCH_TOOL_NAME,
                result=ToolResult(
                    TextInput("web_search_tool_result_error: too_many_requests"),
                    metadata={
                        "error": True,
                        "error_code": "too_many_requests",
                        "type": "web_search_tool_result_error",
                    },
                ),
                block=result,
            ),
        ]

    async def test_web_fetch_pdf_binary_input(self) -> None:
        call = ServerToolUseBlock(id="f1", name="web_fetch", input={}, type="server_tool_use")
        src = Base64PDFSource(data="YWJj", media_type="application/pdf", type="base64")
        document = DocumentBlock(source=src, title="Doc", type="document", citations=None)
        fetch = WebFetchBlock(content=document, retrieved_at="2026-01-01", type="web_fetch_result", url="https://x")
        result = WebFetchToolResultBlock(tool_use_id="f1", type="web_fetch_tool_result", content=fetch)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="f1", name=WEB_FETCH_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="f1",
                name=WEB_FETCH_TOOL_NAME,
                result=ToolResult(
                    UrlInput("https://x", kind=BinaryType.BINARY),
                    BinaryInput(b"abc", media_type="application/pdf", kind=BinaryType.DOCUMENT),
                    metadata={"retrieved_at": "2026-01-01", "title": "Doc"},
                ),
                block=result,
            ),
        ]

    async def test_web_fetch_plain_text_input(self) -> None:
        call = ServerToolUseBlock(id="f2", name="web_fetch", input={}, type="server_tool_use")
        src = PlainTextSource(data="hello world", media_type="text/plain", type="text")
        document = DocumentBlock(source=src, title="T", type="document", citations=None)
        fetch = WebFetchBlock(content=document, retrieved_at="2026-01-02", type="web_fetch_result", url="https://y")
        result = WebFetchToolResultBlock(tool_use_id="f2", type="web_fetch_tool_result", content=fetch)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="f2", name=WEB_FETCH_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="f2",
                name=WEB_FETCH_TOOL_NAME,
                result=ToolResult(
                    UrlInput("https://y", kind=BinaryType.BINARY),
                    TextInput("hello world"),
                    metadata={"retrieved_at": "2026-01-02", "title": "T"},
                ),
                block=result,
            ),
        ]

    async def test_web_fetch_error_text_input(self) -> None:
        call = ServerToolUseBlock(id="f3", name="web_fetch", input={}, type="server_tool_use")
        err = WebFetchToolResultErrorBlock(error_code="url_not_accessible", type="web_fetch_tool_result_error")
        result = WebFetchToolResultBlock(tool_use_id="f3", type="web_fetch_tool_result", content=err)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="f3", name=WEB_FETCH_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="f3",
                name=WEB_FETCH_TOOL_NAME,
                result=ToolResult(
                    TextInput("web_fetch_tool_result_error: url_not_accessible"),
                    metadata={
                        "error": True,
                        "error_code": "url_not_accessible",
                        "type": "web_fetch_tool_result_error",
                    },
                ),
                block=result,
            ),
        ]

    async def test_code_execution_success_emits_stdout_and_files(self) -> None:
        call = ServerToolUseBlock(id="c1", name="code_execution", input={}, type="server_tool_use")
        out = CodeExecutionOutputBlock(file_id="file-abc", type="code_execution_output")
        body = CodeExecutionResultBlock(
            content=[out], return_code=0, stderr="", stdout="42\n", type="code_execution_result"
        )
        result = CodeExecutionToolResultBlock(tool_use_id="c1", type="code_execution_tool_result", content=body)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="c1", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="c1",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("42\n"),
                    FileIdInput("file-abc"),
                    metadata={"return_code": 0},
                ),
                block=result,
            ),
        ]

    async def test_code_execution_encrypted_emits_files_only(self) -> None:
        call = ServerToolUseBlock(id="c2", name="code_execution", input={}, type="server_tool_use")
        out = CodeExecutionOutputBlock(file_id="file-enc", type="code_execution_output")
        encrypted = EncryptedCodeExecutionResultBlock(
            content=[out],
            encrypted_stdout="opaque",
            return_code=0,
            stderr="",
            type="encrypted_code_execution_result",
        )
        result = CodeExecutionToolResultBlock(tool_use_id="c2", type="code_execution_tool_result", content=encrypted)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="c2", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="c2",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    FileIdInput("file-enc"),
                    metadata={"return_code": 0, "encrypted": True},
                ),
                block=result,
            ),
        ]

    async def test_code_execution_error_text_input(self) -> None:
        call = ServerToolUseBlock(id="c3", name="code_execution", input={}, type="server_tool_use")
        err = CodeExecutionToolResultError(error_code="unavailable", type="code_execution_tool_result_error")
        result = CodeExecutionToolResultBlock(tool_use_id="c3", type="code_execution_tool_result", content=err)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="c3", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="c3",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("code_execution_tool_result_error: unavailable"),
                    metadata={
                        "error": True,
                        "error_code": "unavailable",
                        "type": "code_execution_tool_result_error",
                    },
                ),
                block=result,
            ),
        ]

    async def test_bash_code_execution_success(self) -> None:
        call = ServerToolUseBlock(id="b1", name="bash_code_execution", input={}, type="server_tool_use")
        out = BashCodeExecutionOutputBlock(file_id="file-bash", type="bash_code_execution_output")
        body = BashCodeExecutionResultBlock(
            content=[out], return_code=0, stderr="warn", stdout="ok\n", type="bash_code_execution_result"
        )
        result = BashCodeExecutionToolResultBlock(
            tool_use_id="b1", type="bash_code_execution_tool_result", content=body
        )

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="b1", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="b1",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("ok\n"),
                    TextInput("warn"),
                    FileIdInput("file-bash"),
                    metadata={"return_code": 0},
                ),
                block=result,
            ),
        ]

    async def test_bash_code_execution_error(self) -> None:
        call = ServerToolUseBlock(id="b2", name="bash_code_execution", input={}, type="server_tool_use")
        err = BashCodeExecutionToolResultError(error_code="unavailable", type="bash_code_execution_tool_result_error")
        result = BashCodeExecutionToolResultBlock(tool_use_id="b2", type="bash_code_execution_tool_result", content=err)

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="b2", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="b2",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("bash_code_execution_tool_result_error: unavailable"),
                    metadata={
                        "error": True,
                        "error_code": "unavailable",
                        "type": "bash_code_execution_tool_result_error",
                    },
                ),
                block=result,
            ),
        ]

    async def test_text_editor_view_returns_content_text(self) -> None:
        call = ServerToolUseBlock(id="t1", name="text_editor_code_execution", input={}, type="server_tool_use")
        view = TextEditorCodeExecutionViewResultBlock(
            content="line1\nline2",
            file_type="text",
            num_lines=2,
            start_line=1,
            total_lines=2,
            type="text_editor_code_execution_view_result",
        )
        result = TextEditorCodeExecutionToolResultBlock(
            tool_use_id="t1", type="text_editor_code_execution_tool_result", content=view
        )

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="t1", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="t1",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("line1\nline2"),
                    metadata={"file_type": "text", "num_lines": 2, "start_line": 1, "total_lines": 2},
                ),
                block=result,
            ),
        ]

    async def test_text_editor_create_empty_parts(self) -> None:
        call = ServerToolUseBlock(id="t2", name="text_editor_code_execution", input={}, type="server_tool_use")
        create = TextEditorCodeExecutionCreateResultBlock(
            is_file_update=False, type="text_editor_code_execution_create_result"
        )
        result = TextEditorCodeExecutionToolResultBlock(
            tool_use_id="t2", type="text_editor_code_execution_tool_result", content=create
        )

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="t2", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="t2",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(metadata={"is_file_update": False}),
                block=result,
            ),
        ]

    async def test_text_editor_str_replace_lines_text(self) -> None:
        call = ServerToolUseBlock(id="t3", name="text_editor_code_execution", input={}, type="server_tool_use")
        replace = TextEditorCodeExecutionStrReplaceResultBlock(
            lines=["a", "b"],
            new_lines=2,
            new_start=1,
            old_lines=2,
            old_start=1,
            type="text_editor_code_execution_str_replace_result",
        )
        result = TextEditorCodeExecutionToolResultBlock(
            tool_use_id="t3", type="text_editor_code_execution_tool_result", content=replace
        )

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="t3", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="t3",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("a\nb"),
                    metadata={"new_lines": 2, "new_start": 1, "old_lines": 2, "old_start": 1},
                ),
                block=result,
            ),
        ]

    async def test_text_editor_error_text_input(self) -> None:
        call = ServerToolUseBlock(id="t4", name="text_editor_code_execution", input={}, type="server_tool_use")
        err = TextEditorCodeExecutionToolResultError(
            error_code="file_not_found",
            error_message="missing",
            type="text_editor_code_execution_tool_result_error",
        )
        result = TextEditorCodeExecutionToolResultBlock(
            tool_use_id="t4", type="text_editor_code_execution_tool_result", content=err
        )

        _, events = await _process([call, result])

        assert events == [
            AnthropicServerToolCallEvent(id="t4", name=CODE_EXECUTION_TOOL_NAME, arguments="{}", block=call),
            AnthropicServerToolResultEvent(
                parent_id="t4",
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(
                    TextInput("text_editor_code_execution_tool_result_error: file_not_found: missing"),
                    metadata={
                        "error": True,
                        "error_code": "file_not_found",
                        "error_message": "missing",
                        "type": "text_editor_code_execution_tool_result_error",
                    },
                ),
                block=result,
            ),
        ]
