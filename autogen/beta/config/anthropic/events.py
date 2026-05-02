# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from base64 import b64decode
from typing import Any, TypeAlias

from anthropic.types import (
    Base64PDFSource,
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    BashCodeExecutionToolResultError,
    CodeExecutionResultBlock,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultError,
    EncryptedCodeExecutionResultBlock,
    PlainTextSource,
    ServerToolUseBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultError,
    TextEditorCodeExecutionViewResultBlock,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)

from autogen.beta.events import (
    BinaryInput,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    FileIdInput,
    Input,
    TextInput,
    UrlInput,
)
from autogen.beta.events.base import Field
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

AnthropicServerToolResultBlockType: TypeAlias = (
    WebSearchToolResultBlock
    | WebFetchToolResultBlock
    | CodeExecutionToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
)


class AnthropicServerToolCallEvent(BuiltinToolCallEvent):
    block: ServerToolUseBlock = Field(repr=False)

    @classmethod
    def from_block(cls, block: ServerToolUseBlock) -> "AnthropicServerToolCallEvent | None":
        match block.name:
            case "web_search":
                name = WEB_SEARCH_TOOL_NAME
            case "web_fetch":
                name = WEB_FETCH_TOOL_NAME
            case "code_execution" | "bash_code_execution" | "text_editor_code_execution":
                name = CODE_EXECUTION_TOOL_NAME
            case _:
                return None
        return cls(
            id=block.id,
            name=name,
            arguments=json.dumps(block.input),
            block=block,
        )


class AnthropicServerToolResultEvent(BuiltinToolResultEvent):
    block: AnthropicServerToolResultBlockType = Field(repr=False)

    @classmethod
    def from_block(cls, block: object) -> "AnthropicServerToolResultEvent | None":
        name: str
        parts: list[Input] = []
        metadata: dict[str, Any] = {}

        if isinstance(block, WebSearchToolResultBlock):
            name = WEB_SEARCH_TOOL_NAME
            content = block.content
            if isinstance(content, WebSearchToolResultError):
                parts = [TextInput(f"{content.type}: {content.error_code}")]
                metadata = {"error": True, "error_code": content.error_code, "type": content.type}
            else:
                parts = [
                    UrlInput(
                        r.url,
                        kind=BinaryType.BINARY,
                        metadata={"title": r.title, "page_age": r.page_age},
                    )
                    for r in content
                    if isinstance(r, WebSearchResultBlock)
                ]
                metadata = {"count": len(content)}

        elif isinstance(block, WebFetchToolResultBlock):
            name = WEB_FETCH_TOOL_NAME
            content = block.content
            if isinstance(content, WebFetchToolResultErrorBlock):
                parts = [TextInput(f"{content.type}: {content.error_code}")]
                metadata = {"error": True, "error_code": content.error_code, "type": content.type}
            elif isinstance(content, WebFetchBlock):
                document = content.content
                source = document.source
                parts = [UrlInput(content.url, kind=BinaryType.BINARY)]
                if isinstance(source, Base64PDFSource):
                    parts.append(
                        BinaryInput(b64decode(source.data), media_type="application/pdf", kind=BinaryType.DOCUMENT)
                    )
                elif isinstance(source, PlainTextSource):
                    parts.append(TextInput(source.data))
                metadata = {"retrieved_at": content.retrieved_at, "title": document.title}

        elif isinstance(block, (CodeExecutionToolResultBlock, BashCodeExecutionToolResultBlock)):
            name = CODE_EXECUTION_TOOL_NAME
            content = block.content
            if isinstance(content, (CodeExecutionToolResultError, BashCodeExecutionToolResultError)):
                parts = [TextInput(f"{content.type}: {content.error_code}")]
                metadata = {"error": True, "error_code": content.error_code, "type": content.type}
            elif isinstance(content, EncryptedCodeExecutionResultBlock):
                parts = [FileIdInput(o.file_id) for o in content.content]
                metadata = {"return_code": content.return_code, "encrypted": True}
            elif isinstance(content, (CodeExecutionResultBlock, BashCodeExecutionResultBlock)):
                if content.stdout:
                    parts.append(TextInput(content.stdout))
                if content.stderr:
                    parts.append(TextInput(content.stderr))
                parts.extend(FileIdInput(o.file_id) for o in content.content)
                metadata = {"return_code": content.return_code}

        elif isinstance(block, TextEditorCodeExecutionToolResultBlock):
            name = CODE_EXECUTION_TOOL_NAME
            content = block.content
            if isinstance(content, TextEditorCodeExecutionToolResultError):
                text = f"{content.type}: {content.error_code}"
                if content.error_message:
                    text = f"{text}: {content.error_message}"
                parts = [TextInput(text)]
                metadata = {
                    "error": True,
                    "error_code": content.error_code,
                    "error_message": content.error_message,
                    "type": content.type,
                }
            elif isinstance(content, TextEditorCodeExecutionViewResultBlock):
                parts = [TextInput(content.content)]
                metadata = {
                    "file_type": content.file_type,
                    "num_lines": content.num_lines,
                    "start_line": content.start_line,
                    "total_lines": content.total_lines,
                }
            elif isinstance(content, TextEditorCodeExecutionCreateResultBlock):
                metadata = {"is_file_update": content.is_file_update}
            elif isinstance(content, TextEditorCodeExecutionStrReplaceResultBlock):
                if content.lines is not None:
                    parts = [TextInput("\n".join(content.lines))]
                metadata = {
                    "new_lines": content.new_lines,
                    "new_start": content.new_start,
                    "old_lines": content.old_lines,
                    "old_start": content.old_start,
                }

        else:
            return None

        return cls(
            parent_id=block.tool_use_id,
            name=name,
            result=ToolResult(parts=parts, metadata=metadata),
            block=block,
        )
