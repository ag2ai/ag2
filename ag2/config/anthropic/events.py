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
    ContainerUploadBlock,
    EncryptedCodeExecutionResultBlock,
    PlainTextSource,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionToolResultError,
    TextEditorCodeExecutionViewResultBlock,
    ToolSearchToolResultBlock,
    ToolSearchToolResultError,
    ToolSearchToolSearchResultBlock,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)

from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    Field,
    FileIdInput,
    Input,
    ProviderReplay,
    TextInput,
    ToolResult,
    UrlInput,
)
from ag2.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from ag2.tools.builtin.tool_search import TOOL_SEARCH_TOOL_NAME
from ag2.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from ag2.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

AnthropicServerToolResultBlockType: TypeAlias = (
    WebSearchToolResultBlock
    | WebFetchToolResultBlock
    | CodeExecutionToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
    | ToolSearchToolResultBlock
)


class AnthropicRedactedThinkingEvent(BaseEvent, ProviderReplay):
    """Safety-redacted thinking, opaque and encrypted.

    Anthropic requires the block echoed back unchanged on the next turn, so it is
    a replay anchor rather than a transient reasoning event.
    """

    __replay_role__ = "anchor"

    block: RedactedThinkingBlock = Field(repr=False)


class AnthropicContainerUploadEvent(BaseEvent):
    """A file the code-execution container produced.

    Recorded so the ``file_id`` survives, never replayed: the API answers 400
    "'container_upload' blocks are not permitted within assistant turns".
    """

    block: ContainerUploadBlock = Field(repr=False)

    @property
    def file_id(self) -> str:
        return self.block.file_id


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
            case "tool_search_tool_regex" | "tool_search_tool_bm25":
                name = TOOL_SEARCH_TOOL_NAME
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

        # Each branch names its own content: every hosted tool answers with a different
        # union, so one shared local would read as whichever branch bound it first.
        if isinstance(block, WebSearchToolResultBlock):
            name = WEB_SEARCH_TOOL_NAME
            search_result = block.content
            if isinstance(search_result, WebSearchToolResultError):
                parts = [TextInput(f"{search_result.type}: {search_result.error_code}")]
                metadata = {"error": True, "error_code": search_result.error_code, "type": search_result.type}
            else:
                parts = [
                    UrlInput(
                        r.url,
                        kind=BinaryType.BINARY,
                        metadata={"title": r.title, "page_age": r.page_age},
                    )
                    for r in search_result
                    if isinstance(r, WebSearchResultBlock)
                ]
                metadata = {"count": len(search_result)}

        elif isinstance(block, WebFetchToolResultBlock):
            name = WEB_FETCH_TOOL_NAME
            fetch_result = block.content
            if isinstance(fetch_result, WebFetchToolResultErrorBlock):
                parts = [TextInput(f"{fetch_result.type}: {fetch_result.error_code}")]
                metadata = {"error": True, "error_code": fetch_result.error_code, "type": fetch_result.type}
            elif isinstance(fetch_result, WebFetchBlock):
                document = fetch_result.content
                source = document.source
                parts = [UrlInput(fetch_result.url, kind=BinaryType.BINARY)]
                if isinstance(source, Base64PDFSource):
                    parts.append(
                        BinaryInput(b64decode(source.data), media_type="application/pdf", kind=BinaryType.DOCUMENT)
                    )
                elif isinstance(source, PlainTextSource):
                    parts.append(TextInput(source.data))
                metadata = {"retrieved_at": fetch_result.retrieved_at, "title": document.title}

        elif isinstance(block, (CodeExecutionToolResultBlock, BashCodeExecutionToolResultBlock)):
            name = CODE_EXECUTION_TOOL_NAME
            exec_result = block.content
            if isinstance(exec_result, (CodeExecutionToolResultError, BashCodeExecutionToolResultError)):
                parts = [TextInput(f"{exec_result.type}: {exec_result.error_code}")]
                metadata = {"error": True, "error_code": exec_result.error_code, "type": exec_result.type}
            elif isinstance(exec_result, EncryptedCodeExecutionResultBlock):
                parts = [FileIdInput(o.file_id) for o in exec_result.content]
                metadata = {"return_code": exec_result.return_code, "encrypted": True}
            elif isinstance(exec_result, (CodeExecutionResultBlock, BashCodeExecutionResultBlock)):
                if exec_result.stdout:
                    parts.append(TextInput(exec_result.stdout))
                if exec_result.stderr:
                    parts.append(TextInput(exec_result.stderr))
                parts.extend(FileIdInput(o.file_id) for o in exec_result.content)
                metadata = {"return_code": exec_result.return_code}

        elif isinstance(block, TextEditorCodeExecutionToolResultBlock):
            name = CODE_EXECUTION_TOOL_NAME
            edit_result = block.content
            if isinstance(edit_result, TextEditorCodeExecutionToolResultError):
                text = f"{edit_result.type}: {edit_result.error_code}"
                if edit_result.error_message:
                    text = f"{text}: {edit_result.error_message}"
                parts = [TextInput(text)]
                metadata = {
                    "error": True,
                    "error_code": edit_result.error_code,
                    "error_message": edit_result.error_message,
                    "type": edit_result.type,
                }
            elif isinstance(edit_result, TextEditorCodeExecutionViewResultBlock):
                parts = [TextInput(edit_result.content)]
                metadata = {
                    "file_type": edit_result.file_type,
                    "num_lines": edit_result.num_lines,
                    "start_line": edit_result.start_line,
                    "total_lines": edit_result.total_lines,
                }
            elif isinstance(edit_result, TextEditorCodeExecutionCreateResultBlock):
                metadata = {"is_file_update": edit_result.is_file_update}
            elif isinstance(edit_result, TextEditorCodeExecutionStrReplaceResultBlock):
                if edit_result.lines is not None:
                    parts = [TextInput("\n".join(edit_result.lines))]
                metadata = {
                    "new_lines": edit_result.new_lines,
                    "new_start": edit_result.new_start,
                    "old_lines": edit_result.old_lines,
                    "old_start": edit_result.old_start,
                }

        elif isinstance(block, ToolSearchToolResultBlock):
            name = TOOL_SEARCH_TOOL_NAME
            tool_search_result = block.content
            if isinstance(tool_search_result, ToolSearchToolResultError):
                parts = [TextInput(f"{tool_search_result.type}: {tool_search_result.error_code}")]
                metadata = {"error": True, "error_code": tool_search_result.error_code, "type": tool_search_result.type}
            elif isinstance(tool_search_result, ToolSearchToolSearchResultBlock):
                references = [ref.tool_name for ref in tool_search_result.tool_references]
                parts = [TextInput(", ".join(references))] if references else []
                metadata = {"tool_references": references}

        else:
            return None

        return cls(
            parent_id=block.tool_use_id,
            name=name,
            result=ToolResult(parts=parts, metadata=metadata),
            block=block,
        )
