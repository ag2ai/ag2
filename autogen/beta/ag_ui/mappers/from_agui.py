# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from base64 import b64decode
from collections.abc import Iterable

from ag_ui.core import Message

from autogen.beta import ToolResult, events
from autogen.beta.config import ModelConfig
from autogen.beta.events import BuiltinToolCallEvent

from .registry import KNOWN_BUILTIN_NAMES, call_from_agui, result_from_agui

logger = logging.getLogger(__name__)


def map_agui_messages_to_events(
    messages: Iterable[Message],
    config: ModelConfig | None = None,
) -> tuple[list[str], list[events.BaseEvent]]:
    prompt: list[str] = []
    out: list[events.BaseEvent] = []
    # Pre-scan tool_calls: remember which AG-UI tool_call_id resolved to which
    # provider-specific builtin call. The matching `m.role == "tool"` message
    # uses this to synthesize the proper *ServerToolResultEvent.
    builtin_call_by_id: dict[str, BuiltinToolCallEvent] = {}

    input_buffer: list[events.BaseEvent] = []
    for m in messages:
        if m.role == "user":
            content = m.content
            if isinstance(content, str):
                input_buffer.append(events.TextInput(content))
                continue

            for c in content:
                if c.type == "text":
                    input_buffer.append(events.TextInput(c.text))

                elif c.url:
                    input_buffer.append(
                        events.UrlInput(
                            c.url,
                            kind=events.BinaryType.BINARY,
                        )
                    )

                elif c.id:
                    input_buffer.append(
                        events.FileIdInput(
                            c.id,
                            filename=c.filename,
                        )
                    )

                elif c.data:
                    input_buffer.append(
                        events.BinaryInput(
                            b64decode(c.data),
                            media_type=c.mime_type,
                        )
                    )

            continue

        if input_buffer:
            out.append(events.ModelRequest(list(input_buffer)))
            input_buffer.clear()

        if m.role in ["system", "developer"]:
            prompt.append(m.content)

        elif m.role == "assistant":
            tool_calls: list[events.ToolCallEvent] = []
            for t in m.tool_calls or ():
                builtin = call_from_agui(config, t.function.name, t.id, t.function.arguments)
                if builtin is not None:
                    tool_calls.append(builtin)
                    builtin_call_by_id[t.id] = builtin
                    continue
                if t.function.name in KNOWN_BUILTIN_NAMES:
                    logger.warning(
                        "ag_ui: builtin mapper failed to restore call '%s' (id=%s); "
                        "falling back to plain ToolCallEvent. Arguments shape may not "
                        "match any known SDK type — model will re-execute.",
                        t.function.name,
                        t.id,
                    )
                tool_calls.append(
                    events.ToolCallEvent(
                        id=t.id,
                        name=t.function.name,
                        arguments=t.function.arguments,
                    )
                )

            out.append(
                events.ModelResponse(
                    events.ModelMessage(m.content) if m.content else None,
                    tool_calls=events.ToolCallsEvent(tool_calls),
                )
            )

        elif m.role == "tool":
            raw = m.error or m.content
            if (builtin_call := builtin_call_by_id.get(m.tool_call_id)) is not None:
                builtin_result = result_from_agui(config, builtin_call, raw)
                if builtin_result is not None:
                    out.append(events.ToolResultsEvent([builtin_result]))
                    continue
                logger.warning(
                    "ag_ui: builtin result mapper returned None for tool_call_id=%s; "
                    "falling back to plain ToolResultEvent.",
                    m.tool_call_id,
                )
            out.append(
                events.ToolResultsEvent([
                    events.ToolResultEvent(
                        parent_id=m.tool_call_id,
                        result=ToolResult.ensure_result(raw),
                    )
                ])
            )

    if input_buffer:
        out.append(events.ModelRequest(input_buffer))

    return prompt, out


__all__ = ("map_agui_messages_to_events",)
