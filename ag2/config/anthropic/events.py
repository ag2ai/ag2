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
            searched = block.content
            if isinstance(searched, WebSearchToolResultError):
                parts = [TextInput(f"{searched.type}: {searched.error_code}")]
                metadata = {"error": True, "error_code": searched.error_code, "type": searched.type}
            else:
                parts = [
                    UrlInput(
                        r.url,
                        kind=BinaryType.BINARY,
                        metadata={"title": r.title, "page_age": r.page_age},
                    )
                    for r in searched
                    if isinstance(r, WebSearchResultBlock)
                ]
                metadata = {"count": len(searched)}

        elif isinstance(block, WebFetchToolResultBlock):
            name = WEB_FETCH_TOOL_NAME
            fetched = block.content
            if isinstance(fetched, WebFetchToolResultErrorBlock):
                parts = [TextInput(f"{fetched.type}: {fetched.error_code}")]
                metadata = {"error": True, "error_code": fetched.error_code, "type": fetched.type}
            elif isinstance(fetched, WebFetchBlock):
                document = fetched.content
                source = document.source
                parts = [UrlInput(fetched.url, kind=BinaryType.BINARY)]
                if isinstance(source, Base64PDFSource):
                    parts.append(
                        BinaryInput(b64decode(source.data), media_type="application/pdf", kind=BinaryType.DOCUMENT)
                    )
                elif isinstance(source, PlainTextSource):
                    parts.append(TextInput(source.data))
                metadata = {"retrieved_at": fetched.retrieved_at, "title": document.title}

        elif isinstance(block, (CodeExecutionToolResultBlock, BashCodeExecutionToolResultBlock)):
            name = CODE_EXECUTION_TOOL_NAME
            executed = block.content
            if isinstance(executed, (CodeExecutionToolResultError, BashCodeExecutionToolResultError)):
                parts = [TextInput(f"{executed.type}: {executed.error_code}")]
                metadata = {"error": True, "error_code": executed.error_code, "type": executed.type}
            elif isinstance(executed, EncryptedCodeExecutionResultBlock):
                parts = [FileIdInput(o.file_id) for o in executed.content]
                metadata = {"return_code": executed.return_code, "encrypted": True}
            elif isinstance(executed, (CodeExecutionResultBlock, BashCodeExecutionResultBlock)):
                if executed.stdout:
                    parts.append(TextInput(executed.stdout))
                if executed.stderr:
                    parts.append(TextInput(executed.stderr))
                parts.extend(FileIdInput(o.file_id) for o in executed.content)
                metadata = {"return_code": executed.return_code}

        elif isinstance(block, TextEditorCodeExecutionToolResultBlock):
            name = CODE_EXECUTION_TOOL_NAME
            edited = block.content
            if isinstance(edited, TextEditorCodeExecutionToolResultError):
                text = f"{edited.type}: {edited.error_code}"
                if edited.error_message:
                    text = f"{text}: {edited.error_message}"
                parts = [TextInput(text)]
                metadata = {
                    "error": True,
                    "error_code": edited.error_code,
                    "error_message": edited.error_message,
                    "type": edited.type,
                }
            elif isinstance(edited, TextEditorCodeExecutionViewResultBlock):
                parts = [TextInput(edited.content)]
                metadata = {
                    "file_type": edited.file_type,
                    "num_lines": edited.num_lines,
                    "start_line": edited.start_line,
                    "total_lines": edited.total_lines,
                }
            elif isinstance(edited, TextEditorCodeExecutionCreateResultBlock):
                metadata = {"is_file_update": edited.is_file_update}
            elif isinstance(edited, TextEditorCodeExecutionStrReplaceResultBlock):
                if edited.lines is not None:
                    parts = [TextInput("\n".join(edited.lines))]
                metadata = {
                    "new_lines": edited.new_lines,
                    "new_start": edited.new_start,
                    "old_lines": edited.old_lines,
                    "old_start": edited.old_start,
                }

        elif isinstance(block, ToolSearchToolResultBlock):
            name = TOOL_SEARCH_TOOL_NAME
            tools_found = block.content
            if isinstance(tools_found, ToolSearchToolResultError):
                parts = [TextInput(f"{tools_found.type}: {tools_found.error_code}")]
                metadata = {"error": True, "error_code": tools_found.error_code, "type": tools_found.type}
            elif isinstance(tools_found, ToolSearchToolSearchResultBlock):
                references = [ref.tool_name for ref in tools_found.tool_references]
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
