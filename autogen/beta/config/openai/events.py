# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from base64 import b64decode
from typing import Any, TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseFunctionWebSearch,
    ResponseReasoningItem,
)
from openai.types.responses.response_code_interpreter_tool_call import OutputImage, OutputLogs
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall

from autogen.beta.events import (
    BinaryInput,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    Input,
    ModelReasoning,
    TextInput,
    UrlInput,
)
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
                # warnings=False: pydantic 2.x warns on Action discriminated-union
                # serialization and on action.sources[].type values that the SDK
                # has not caught up to (e.g. "api"). The warning is informational —
                # the dump still produces correct JSON for round-trip.
                arguments=item.action.model_dump_json(warnings=False),
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
        name: str
        parts: list[Input] = []
        metadata: dict[str, Any] = {}

        if isinstance(item, ResponseFunctionWebSearch):
            name = WEB_SEARCH_TOOL_NAME
            action = item.action
            metadata = {"action_type": action.type, "status": item.status}
            if isinstance(action, ActionSearch):
                # `sources` is populated only when the request asks for it via
                # include=["web_search_call.action.sources"]. The SDK declares
                # source.url as `str`, but the API has been observed to return
                # entries with empty url for synthesised/internal sources —
                # skip them rather than emit UrlInput(None).
                for source in action.sources or []:
                    if source.url:
                        parts.append(UrlInput(source.url, kind=BinaryType.BINARY))
                if action.queries:
                    metadata["queries"] = list(action.queries)
            elif isinstance(action, ActionOpenPage):
                if action.url:
                    parts.append(UrlInput(action.url, kind=BinaryType.BINARY))
            elif isinstance(action, ActionFind):
                parts.append(UrlInput(action.url, kind=BinaryType.BINARY))
                metadata["pattern"] = action.pattern

        elif isinstance(item, ResponseCodeInterpreterToolCall):
            name = CODE_EXECUTION_TOOL_NAME
            for output in item.outputs or []:
                if isinstance(output, OutputLogs):
                    parts.append(TextInput(output.logs))
                elif isinstance(output, OutputImage):
                    parts.append(UrlInput(output.url, kind=BinaryType.IMAGE))
            metadata = {"container_id": item.container_id, "status": item.status}

        elif isinstance(item, ImageGenerationCall) and item.result:
            name = IMAGE_GENERATION_TOOL_NAME
            parts = [BinaryInput(b64decode(item.result), media_type="image/png", kind=BinaryType.IMAGE)]
            metadata = item.model_dump(exclude={"result", "status", "type"})

        else:
            return None

        return cls(parent_id=parent_id, name=name, result=ToolResult(parts=parts, metadata=metadata))


class OpenAIReasoningEvent(ModelReasoning):
    __transient__ = False

    item: ResponseReasoningItem = Field(repr=False)
