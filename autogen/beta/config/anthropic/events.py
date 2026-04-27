# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import TypeAlias

from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    ServerToolUseBlock,
    TextEditorCodeExecutionToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchToolResultBlock,
)

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent
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
        if isinstance(block, WebSearchToolResultBlock):
            name = WEB_SEARCH_TOOL_NAME
        elif isinstance(block, WebFetchToolResultBlock):
            name = WEB_FETCH_TOOL_NAME
        elif isinstance(
            block,
            (
                CodeExecutionToolResultBlock,
                BashCodeExecutionToolResultBlock,
                TextEditorCodeExecutionToolResultBlock,
            ),
        ):
            name = CODE_EXECUTION_TOOL_NAME
        else:
            return None
        return cls(
            parent_id=block.tool_use_id,
            name=name,
            result=ToolResult(),
            block=block,
        )
