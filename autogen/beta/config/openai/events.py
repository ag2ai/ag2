# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseFunctionWebSearch,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import ImageGenerationCall

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent, ModelReasoning
from autogen.beta.events.base import Field
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

OpenAIServerToolItem: TypeAlias = ResponseFunctionWebSearch | ResponseCodeInterpreterToolCall | ImageGenerationCall


class OpenAIServerToolCallEvent(BuiltinToolCallEvent):
    item: OpenAIServerToolItem = Field(repr=False)

    @classmethod
    def from_item(cls, item: object) -> "OpenAIServerToolCallEvent | None":
        if isinstance(item, ResponseFunctionWebSearch):
            return cls(
                id=item.id,
                name=WEB_SEARCH_TOOL_NAME,
                arguments=item.action.model_dump_json(),
                item=item,
            )
        if isinstance(item, ResponseCodeInterpreterToolCall):
            return cls(
                id=item.id,
                name=CODE_EXECUTION_TOOL_NAME,
                arguments=json.dumps({"code": item.code}) if item.code is not None else "{}",
                item=item,
            )
        if isinstance(item, ImageGenerationCall) and item.result:
            return cls(
                id=item.id,
                name=IMAGE_GENERATION_TOOL_NAME,
                arguments="",
                item=item,
            )
        return None


class OpenAIServerToolResultEvent(BuiltinToolResultEvent):
    @classmethod
    def from_item(cls, item: object, *, parent_id: str) -> "OpenAIServerToolResultEvent | None":
        if isinstance(item, ResponseFunctionWebSearch):
            return cls(parent_id=parent_id, name=WEB_SEARCH_TOOL_NAME, result=ToolResult())
        if isinstance(item, ResponseCodeInterpreterToolCall):
            return cls(parent_id=parent_id, name=CODE_EXECUTION_TOOL_NAME, result=ToolResult())
        if isinstance(item, ImageGenerationCall) and item.result:
            return cls(parent_id=parent_id, name=IMAGE_GENERATION_TOOL_NAME, result=ToolResult())
        return None


class OpenAIReasoningEvent(ModelReasoning):
    __transient__ = False

    item: ResponseReasoningItem = Field(repr=False)
