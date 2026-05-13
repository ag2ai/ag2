# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any
from uuid import uuid4

from google.genai import types

from autogen.beta.config.gemini.events import GeminiServerToolCallEvent, GeminiServerToolResultEvent
from autogen.beta.events import TextInput, ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)


def gemini_call_from_agui(
    name: str,
    call_id: str,
    payload: dict[str, Any],
) -> GeminiServerToolCallEvent | None:
    if name == CODE_EXECUTION_TOOL_NAME and "language" in payload:
        return _build_executable_code(call_id, payload)

    if name in (WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME) and "queries" in payload:
        return _build_grounding(call_id, name, payload)

    return None


def _build_executable_code(call_id: str, payload: dict[str, Any]) -> GeminiServerToolCallEvent | None:
    try:
        part = types.Part.model_validate({
            "executable_code": {
                "code": payload.get("code", ""),
                "language": payload["language"],
            }
        })
    except Exception as exc:
        logger.warning("ag_ui gemini mapper: executable_code restore failed for id=%s: %s", call_id, exc)
        return None

    return GeminiServerToolCallEvent(
        id=call_id,
        name=CODE_EXECUTION_TOOL_NAME,
        arguments=json.dumps(payload),
        part=part,
    )


def _build_grounding(call_id: str, name: str, payload: dict[str, Any]) -> GeminiServerToolCallEvent | None:
    try:
        gm = types.GroundingMetadata.model_validate({
            "web_search_queries": list(payload.get("queries") or []),
        })
    except Exception as exc:
        logger.warning("ag_ui gemini mapper: grounding restore failed for id=%s: %s", call_id, exc)
        return None

    return GeminiServerToolCallEvent(
        # Gemini from_grounding generates ids on the fly (no SDK id for the
        # synthetic grounding call); preserve the AG-UI provided id, or assign
        # a fresh uuid when caller did not supply one.
        id=call_id or str(uuid4()),
        name=name,
        arguments=json.dumps(payload),
        grounding_metadata=gm,
    )


def gemini_result_from_agui(
    call: object,
    content: str,
) -> GeminiServerToolResultEvent | None:
    if not isinstance(call, GeminiServerToolCallEvent):
        return None
    parts = [TextInput(content)] if content else []

    if call.part is not None and call.part.executable_code is not None:
        result_part = types.Part(
            code_execution_result=types.CodeExecutionResult(
                output=content,
                outcome=types.Outcome.OUTCOME_OK,
            )
        )
        return GeminiServerToolResultEvent(
            parent_id=call.id,
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(parts=parts),
            part=result_part,
        )

    if call.grounding_metadata is not None:
        return GeminiServerToolResultEvent(
            parent_id=call.id,
            name=call.name,
            result=ToolResult(parts=parts),
            grounding_metadata=call.grounding_metadata,
        )

    return None
