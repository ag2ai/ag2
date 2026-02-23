# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    BaseEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from pydantic_core import to_jsonable_python

from autogen import ConversableAgent
from autogen.agentchat import ContextVariables
from autogen.agentchat.remote import AgentService, RequestMessage, ServiceResponse
from autogen.doc_utils import export_module

try:
    from starlette.endpoints import HTTPEndpoint
except ImportError:
    # Fallback to Any until Starlette is installed
    HTTPEndpoint = Any  # type: ignore[misc,assignment]


def _get_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@export_module("autogen.ag_ui")
class AGUIStream:
    def __init__(self, agent: ConversableAgent) -> None:
        self.__agent = agent
        self.service = AgentService(agent)

    async def dispatch(
        self,
        incoming: RunAgentInput,
        *,
        context: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> AsyncIterator[str]:
        # TODO: put it into ContextVariables if it supports unpickleable objects
        write_events_stream, read_events_stream = create_memory_object_stream[BaseEvent]()

        state = ContextVariables()
        # add agent initial context
        state.update(self.__agent.context_variables.to_dict())
        # add frontend-passed context
        state.update(incoming.state or {})
        # add manual-passed context
        state.update(context or {})

        async with create_task_group() as tg:
            tg.start_soon(
                run_stream,
                AGStreamInput(incoming=incoming, context=state),
                self.service,
                write_events_stream,
            )

            # EventEncoder typed incompletely, so we need to ignore the type error
            encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]

            async with read_events_stream:
                async for event in read_events_stream:
                    yield encoder.encode(event)

    def build_asgi(self) -> "type[HTTPEndpoint]":
        """Build an ASGI endpoint for the AGUIStream."""
        # import here to avoid Starlette requirements in the main package
        from .asgi import build_asgi

        return build_asgi(self)


@dataclass(slots=True)
class AGStreamInput:
    incoming: RunAgentInput
    context: ContextVariables


async def run_stream(
    command: AGStreamInput,
    service: Callable[[RequestMessage], AsyncGenerator[ServiceResponse, None]],
    write_events_stream: MemoryObjectSendStream[BaseEvent],
) -> None:
    client_tools = []
    client_tools_names: set[str] = set()
    for t in command.incoming.tools:
        func = t.model_dump(exclude_none=True)
        client_tools.append({
            "type": "function",
            "function": func,
        })
        client_tools_names.add(func["name"])

    request = RequestMessage(
        messages=[m.model_dump(exclude_none=True) for m in command.incoming.messages],
        context=command.context.data,
        client_tools=client_tools,
    )

    snapshot = _encode_context(command.context.data)

    async with write_events_stream:
        try:
            await write_events_stream.send(
                RunStartedEvent(
                    thread_id=command.incoming.thread_id,
                    run_id=command.incoming.run_id,
                    timestamp=_get_timestamp(),
                )
            )

            if snapshot:
                await write_events_stream.send(
                    StateSnapshotEvent(
                        snapshot=snapshot,
                        timestamp=_get_timestamp(),
                    )
                )

            streaming_msg_id: str | None = None
            async for response in service(request):
                if response.streaming_text:
                    if not streaming_msg_id:
                        streaming_msg_id = str(uuid4())
                        await write_events_stream.send(
                            TextMessageStartEvent(
                                message_id=streaming_msg_id,
                                timestamp=_get_timestamp(),
                            )
                        )

                    await write_events_stream.send(
                        TextMessageContentEvent(
                            message_id=streaming_msg_id,
                            delta=response.streaming_text,
                            timestamp=_get_timestamp(),
                        )
                    )
                    continue

                if (ctx := _encode_context(response.context)) and ctx != snapshot:
                    snapshot = ctx
                    await write_events_stream.send(
                        StateSnapshotEvent(
                            snapshot=ctx,
                            timestamp=_get_timestamp(),
                        )
                    )

                if msg := response.message:
                    content = msg.get("content", "")

                    has_tool_result = False
                    for tool_response in msg.get("tool_responses", []):
                        has_tool_result = True
                        await write_events_stream.send(
                            ToolCallResultEvent(
                                tool_call_id=tool_response["tool_call_id"],
                                content=tool_response["content"],
                                message_id=str(uuid4()),
                                timestamp=_get_timestamp(),
                                role="tool",
                            )
                        )
                        await write_events_stream.send(
                            ToolCallEndEvent(
                                tool_call_id=tool_response["tool_call_id"],
                                timestamp=_get_timestamp(),
                            )
                        )

                    for tool_call in msg.get("tool_calls", []):
                        func = tool_call["function"]

                        if (name := func.get("name")) in client_tools_names:
                            await write_events_stream.send(
                                ToolCallChunkEvent(
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                    delta=func.get("arguments"),
                                    timestamp=_get_timestamp(),
                                )
                            )

                        else:
                            await write_events_stream.send(
                                ToolCallStartEvent(
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                    timestamp=_get_timestamp(),
                                )
                            )
                            await write_events_stream.send(
                                ToolCallArgsEvent(
                                    tool_call_id=tool_call.get("id"),
                                    delta=func.get("arguments"),
                                    timestamp=_get_timestamp(),
                                )
                            )

                    if content and not has_tool_result:
                        if streaming_msg_id:
                            await write_events_stream.send(
                                TextMessageEndEvent(
                                    message_id=streaming_msg_id,
                                    timestamp=_get_timestamp(),
                                )
                            )
                            streaming_msg_id = None

                        else:
                            await write_events_stream.send(
                                TextMessageChunkEvent(
                                    message_id=str(uuid4()),
                                    delta=content,
                                    timestamp=_get_timestamp(),
                                )
                            )

                if input_message := response.input_required:
                    await write_events_stream.send(
                        TextMessageChunkEvent(
                            message_id=str(uuid4()),
                            delta=input_message,
                            timestamp=_get_timestamp(),
                        )
                    )

        except Exception as e:
            await write_events_stream.send(
                RunErrorEvent(
                    message=repr(e),
                    timestamp=_get_timestamp(),
                )
            )
            raise e

        else:
            await write_events_stream.send(
                RunFinishedEvent(
                    thread_id=command.incoming.thread_id,
                    run_id=command.incoming.run_id,
                    timestamp=_get_timestamp(),
                )
            )


def _encode_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Drop all unserializable values from the context.

    It is required to share with AG-UI frontend application only data values.
    Any Python objects (like functions, classes, etc.) will be dropped from the context."""
    if not context:
        return {}

    context = to_jsonable_python(context, fallback=lambda _: None, exclude_none=True) or {}
    return {k: v for k, v in context.items() if v is not None}
