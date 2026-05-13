# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any, get_args

from openai.types.responses import ResponseCodeInterpreterToolCall, ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall

from autogen.beta.config.openai.events import OpenAIServerToolCallEvent, OpenAIServerToolResultEvent
from autogen.beta.events import TextInput, ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)

_WEB_SEARCH_ACTION_TYPES: frozenset[str] = frozenset(
    literal
    for cls in (ActionSearch, ActionOpenPage, ActionFind)
    for literal in get_args(cls.model_fields["type"].annotation)
)


def openai_call_from_agui(
    name: str,
    call_id: str,
    payload: dict[str, Any],
) -> OpenAIServerToolCallEvent | None:
    if name == WEB_SEARCH_TOOL_NAME and payload.get("type") in _WEB_SEARCH_ACTION_TYPES:
        return _build_web_search(call_id, payload)

    if name == CODE_EXECUTION_TOOL_NAME and "container_id" in payload:
        return _build_code_interpreter(call_id, payload)

    if name == IMAGE_GENERATION_TOOL_NAME:
        return _build_image_generation(call_id, payload)

    return None


def _build_web_search(call_id: str, action_payload: dict[str, Any]) -> OpenAIServerToolCallEvent | None:
    try:
        item = ResponseFunctionWebSearch.model_validate({
            "id": call_id,
            "type": "web_search_call",
            "status": "completed",
            "action": action_payload,
        })
    except Exception as exc:
        logger.warning("ag_ui openai mapper: web_search restore failed for id=%s: %s", call_id, exc)
        return None

    return OpenAIServerToolCallEvent(
        id=call_id,
        name=WEB_SEARCH_TOOL_NAME,
        # Match `from_item` byte-for-byte so round-trip equality holds —
        # `json.dumps(payload)` would re-introduce whitespace.
        arguments=item.action.model_dump_json(warnings=False),
        item=item,
    )


def _build_code_interpreter(call_id: str, payload: dict[str, Any]) -> OpenAIServerToolCallEvent | None:
    try:
        item = ResponseCodeInterpreterToolCall.model_validate({
            "id": call_id,
            "type": "code_interpreter_call",
            "status": "completed",
            "container_id": payload["container_id"],
            "code": payload.get("code"),
        })
    except Exception as exc:
        logger.warning("ag_ui openai mapper: code_interpreter restore failed for id=%s: %s", call_id, exc)
        return None

    return OpenAIServerToolCallEvent(
        id=call_id,
        name=CODE_EXECUTION_TOOL_NAME,
        arguments=json.dumps(payload),
        item=item,
    )


def openai_result_from_agui(
    call: object,
    content: str,
) -> OpenAIServerToolResultEvent | None:
    if not isinstance(call, OpenAIServerToolCallEvent):
        return None
    parts = [TextInput(content)] if content else []
    return OpenAIServerToolResultEvent(
        parent_id=call.id,
        name=call.name,
        result=ToolResult(parts=parts),
    )


def _build_image_generation(call_id: str, payload: dict[str, Any]) -> OpenAIServerToolCallEvent | None:
    try:
        item = ImageGenerationCall.model_validate({
            "id": call_id,
            "type": "image_generation_call",
            "status": "completed",
            **payload,
        })
    except Exception as exc:
        logger.warning("ag_ui openai mapper: image_generation restore failed for id=%s: %s", call_id, exc)
        return None

    return OpenAIServerToolCallEvent(
        id=call_id,
        name=IMAGE_GENERATION_TOOL_NAME,
        arguments=json.dumps(payload) if payload else "",
        item=item,
    )
