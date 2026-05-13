# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any

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

from autogen.beta.config.anthropic.events import (
    CODE_EXECUTION_SDK_NAMES,
    AnthropicServerToolCallEvent,
    AnthropicServerToolResultEvent,
)
from autogen.beta.events import TextInput, ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)


def anthropic_call_from_agui(
    name: str,
    call_id: str,
    payload: dict[str, Any],
) -> AnthropicServerToolCallEvent | None:
    if name == WEB_SEARCH_TOOL_NAME and "query" in payload:
        return _build_web_search(call_id, payload)

    if name == WEB_FETCH_TOOL_NAME and "url" in payload:
        # Anthropic web_fetch input has `url`; Gemini grounding-derived web_fetch
        # carries `queries` instead — that case is handled by the Gemini mapper.
        return _build_web_fetch(call_id, payload)

    if name == CODE_EXECUTION_TOOL_NAME and payload.get("_kind") in CODE_EXECUTION_SDK_NAMES:
        return _build_code_execution(call_id, payload)

    return None


def _build_web_search(call_id: str, payload: dict[str, Any]) -> AnthropicServerToolCallEvent | None:
    return _build(call_id, sdk_name="web_search", agui_name=WEB_SEARCH_TOOL_NAME, payload=payload)


def _build_web_fetch(call_id: str, payload: dict[str, Any]) -> AnthropicServerToolCallEvent | None:
    return _build(call_id, sdk_name="web_fetch", agui_name=WEB_FETCH_TOOL_NAME, payload=payload)


def _build_code_execution(call_id: str, payload: dict[str, Any]) -> AnthropicServerToolCallEvent | None:
    sdk_name = payload["_kind"]
    block_input = {k: v for k, v in payload.items() if k != "_kind"}
    return _build(call_id, sdk_name=sdk_name, agui_name=CODE_EXECUTION_TOOL_NAME, payload=block_input)


def _build(
    call_id: str,
    *,
    sdk_name: str,
    agui_name: str,
    payload: dict[str, Any],
) -> AnthropicServerToolCallEvent | None:
    try:
        block = ServerToolUseBlock.model_validate({
            "id": call_id,
            "type": "server_tool_use",
            "name": sdk_name,
            "input": payload,
        })
    except Exception as exc:
        logger.warning("ag_ui anthropic mapper: %s restore failed for id=%s: %s", sdk_name, call_id, exc)
        return None

    return AnthropicServerToolCallEvent(
        id=call_id,
        name=agui_name,
        arguments=json.dumps({"_kind": sdk_name, **payload} if agui_name == CODE_EXECUTION_TOOL_NAME else payload),
        block=block,
    )


def anthropic_result_from_agui(
    call: object,
    content: str,
) -> AnthropicServerToolResultEvent | None:
    if not isinstance(call, AnthropicServerToolCallEvent):
        return None
    sdk_name = call.block.name
    parent_id = call.id
    parts = [TextInput(content)] if content else []

    if sdk_name == "web_search":
        return _synth_web_search(parent_id, content, parts)
    if sdk_name == "web_fetch":
        return _synth_web_fetch(parent_id, content, parts)
    if sdk_name == "code_execution":
        return _synth_code_execution(parent_id, content, parts)
    if sdk_name == "bash_code_execution":
        return _synth_bash_code_execution(parent_id, content, parts)
    if sdk_name == "text_editor_code_execution":
        return _synth_text_editor_code_execution(parent_id, content, parts)
    return None


def _synth_web_search(parent_id: str, content: str, parts: list) -> AnthropicServerToolResultEvent:
    # Best-effort: each non-empty line that looks like a URL becomes a result block.
    results = [
        WebSearchResultBlock(
            url=line.strip(),
            title="",
            encrypted_content="",
            page_age=None,
            type="web_search_result",
        )
        for line in content.splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]
    block = WebSearchToolResultBlock(
        tool_use_id=parent_id,
        type="web_search_tool_result",
        content=results,
    )
    return AnthropicServerToolResultEvent(
        parent_id=parent_id,
        name=WEB_SEARCH_TOOL_NAME,
        result=ToolResult(parts=parts),
        block=block,
    )


def _synth_web_fetch(parent_id: str, content: str, parts: list) -> AnthropicServerToolResultEvent:
    source = PlainTextSource(data=content, media_type="text/plain", type="text")
    document = DocumentBlock(source=source, title=None, type="document", citations=None)
    fetch = WebFetchBlock(content=document, retrieved_at=None, type="web_fetch_result", url="")
    block = WebFetchToolResultBlock(tool_use_id=parent_id, type="web_fetch_tool_result", content=fetch)
    return AnthropicServerToolResultEvent(
        parent_id=parent_id,
        name=WEB_FETCH_TOOL_NAME,
        result=ToolResult(parts=parts),
        block=block,
    )


def _synth_code_execution(parent_id: str, content: str, parts: list) -> AnthropicServerToolResultEvent:
    body = CodeExecutionResultBlock(
        content=[],
        return_code=0,
        stderr="",
        stdout=content,
        type="code_execution_result",
    )
    block = CodeExecutionToolResultBlock(
        tool_use_id=parent_id,
        type="code_execution_tool_result",
        content=body,
    )
    return AnthropicServerToolResultEvent(
        parent_id=parent_id,
        name=CODE_EXECUTION_TOOL_NAME,
        result=ToolResult(parts=parts),
        block=block,
    )


def _synth_bash_code_execution(parent_id: str, content: str, parts: list) -> AnthropicServerToolResultEvent:
    body = BashCodeExecutionResultBlock(
        content=[],
        return_code=0,
        stderr="",
        stdout=content,
        type="bash_code_execution_result",
    )
    block = BashCodeExecutionToolResultBlock(
        tool_use_id=parent_id,
        type="bash_code_execution_tool_result",
        content=body,
    )
    return AnthropicServerToolResultEvent(
        parent_id=parent_id,
        name=CODE_EXECUTION_TOOL_NAME,
        result=ToolResult(parts=parts),
        block=block,
    )


def _synth_text_editor_code_execution(
    parent_id: str,
    content: str,
    parts: list,
) -> AnthropicServerToolResultEvent:
    view = TextEditorCodeExecutionViewResultBlock(
        content=content,
        file_type="text",
        num_lines=len(content.splitlines()) or 1,
        start_line=1,
        total_lines=len(content.splitlines()) or 1,
        type="text_editor_code_execution_view_result",
    )
    block = TextEditorCodeExecutionToolResultBlock(
        tool_use_id=parent_id,
        type="text_editor_code_execution_tool_result",
        content=view,
    )
    return AnthropicServerToolResultEvent(
        parent_id=parent_id,
        name=CODE_EXECUTION_TOOL_NAME,
        result=ToolResult(parts=parts),
        block=block,
    )
