# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

from autogen import ConversableAgent
from autogen.agentchat.remote import AgentService, RequestMessage
from autogen.doc_utils import export_module

try:
    from starlette.endpoints import HTTPEndpoint
except ImportError:
    HTTPEndpoint = Any


@export_module("autogen.ag_ui")
class AGUIStream:
    def __init__(self, agent: ConversableAgent) -> None:
        self.service = AgentService(agent)

    async def dispatch(
        self,
        data: RunAgentInput,
        *,
        context: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> AsyncIterator[str]:
        encoder = EventEncoder(accept=accept)

        client_tools = []
        client_tools_names: set[str] = set()
        for t in data.tools:
            func = t.model_dump(exclude_none=True)
            client_tools.append({
                "type": "function",
                "function": func,
            })
            client_tools_names.add(func["name"])

        message = RequestMessage(
            messages=[m.model_dump(exclude_none=True) for m in data.messages],
            context=context,
            client_tools=client_tools,
        )

        try:
            yield encoder.encode(RunStartedEvent(thread_id=data.thread_id, run_id=data.run_id))

            async for response in self.service(message):
                msg_id = str(uuid4())

                if msg := response.message:
                    content = msg.get("content", "")

                    has_tool_result = False
                    for tool_response in msg.get("tool_responses", []):
                        has_tool_result = True
                        yield encoder.encode(
                            ToolCallResultEvent(
                                tool_call_id=tool_response["tool_call_id"],
                                content=tool_response["content"],
                                message_id=msg_id,
                            )
                        )
                        yield encoder.encode(
                            ToolCallEndEvent(
                                tool_call_id=tool_response["tool_call_id"],
                            )
                        )

                    has_tool, has_local_tool = False, False
                    for tool_call in msg.get("tool_calls", []):
                        func = tool_call["function"]

                        if (name := func.get("name")) in client_tools_names:
                            has_local_tool = True
                            yield encoder.encode(
                                ToolCallChunkEvent(
                                    parent_message_id=msg_id,
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                    delta=func.get("arguments"),
                                )
                            )

                        else:
                            has_tool = True
                            yield encoder.encode(
                                ToolCallStartEvent(
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                )
                            )
                            yield encoder.encode(
                                ToolCallArgsEvent(
                                    tool_call_id=tool_call.get("id"),
                                    delta=func.get("arguments"),
                                )
                            )

                    if (
                        content
                        and
                        # do not send tool result as chat message
                        not has_tool_result
                        and
                        # message attached to `tool_calls` mostly thinking block
                        # than refular text message
                        (not has_tool or has_local_tool)
                    ):
                        yield encoder.encode(TextMessageChunkEvent(message_id=msg_id, delta=content))

        except Exception as e:
            yield encoder.encode(RunErrorEvent(message=repr(e)))
            raise e

        else:
            yield encoder.encode(RunFinishedEvent(thread_id=data.thread_id, run_id=data.run_id))

    def build_asgi(self) -> "HTTPEndpoint":
        """Build an ASGI endpoint for the AGUIStream."""
        # import here to avoid Starlette requirements in the main package
        from .asgi import build_asgi

        return build_asgi(self)
