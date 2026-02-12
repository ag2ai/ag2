# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
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
from pydantic_core import to_jsonable_python

from autogen import ConversableAgent
from autogen.agentchat import ContextVariables
from autogen.agentchat.remote import AgentService, RequestMessage
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
        stream_queue = asyncio.Queue[BaseEvent]()

        state = ContextVariables()
        # add agent initial context
        state.update(self.__agent.context_variables.to_dict())
        # add frontend-passed context
        state.update(incoming.state or {})
        # add manual-passed context
        state.update(context or {})

        client_tools = []
        client_tools_names: set[str] = set()
        for t in incoming.tools:
            func = t.model_dump(exclude_none=True)
            client_tools.append({
                "type": "function",
                "function": func,
            })
            client_tools_names.add(func["name"])

        message = RequestMessage(
            messages=[m.model_dump(exclude_none=True) for m in incoming.messages],
            context=state.data,
            client_tools=client_tools,
        )

        async def run_stream() -> None:
            snapshot = _encode_context(state.data)

            try:
                await stream_queue.put(
                    RunStartedEvent(
                        thread_id=incoming.thread_id,
                        run_id=incoming.run_id,
                        timestamp=_get_timestamp(),
                    )
                )

                if snapshot:
                    await stream_queue.put(
                        StateSnapshotEvent(
                            snapshot=snapshot,
                            timestamp=_get_timestamp(),
                        )
                    )

                streaming_msg_id: str | None = None
                async for response in self.service(message):
                    if response.streaming_text:
                        if not streaming_msg_id:
                            streaming_msg_id = str(uuid4())
                            await stream_queue.put(
                                TextMessageStartEvent(
                                    message_id=streaming_msg_id,
                                    timestamp=_get_timestamp(),
                                )
                            )

                        await stream_queue.put(
                            TextMessageContentEvent(
                                message_id=streaming_msg_id,
                                delta=response.streaming_text,
                                timestamp=_get_timestamp(),
                            )
                        )
                        continue

                    if (ctx := _encode_context(response.context)) and ctx != snapshot:
                        snapshot = ctx
                        await stream_queue.put(
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
                            await stream_queue.put(
                                ToolCallResultEvent(
                                    tool_call_id=tool_response["tool_call_id"],
                                    content=tool_response["content"],
                                    message_id=str(uuid4()),
                                    timestamp=_get_timestamp(),
                                    role="tool",
                                )
                            )
                            await stream_queue.put(
                                ToolCallEndEvent(
                                    tool_call_id=tool_response["tool_call_id"],
                                    timestamp=_get_timestamp(),
                                )
                            )

                        for tool_call in msg.get("tool_calls", []):
                            func = tool_call["function"]

                            if (name := func.get("name")) in client_tools_names:
                                await stream_queue.put(
                                    ToolCallChunkEvent(
                                        tool_call_id=tool_call.get("id"),
                                        tool_call_name=name,
                                        delta=func.get("arguments"),
                                        timestamp=_get_timestamp(),
                                    )
                                )

                            else:
                                await stream_queue.put(
                                    ToolCallStartEvent(
                                        tool_call_id=tool_call.get("id"),
                                        tool_call_name=name,
                                        timestamp=_get_timestamp(),
                                    )
                                )
                                await stream_queue.put(
                                    ToolCallArgsEvent(
                                        tool_call_id=tool_call.get("id"),
                                        delta=func.get("arguments"),
                                        timestamp=_get_timestamp(),
                                    )
                                )

                        if content and not has_tool_result:
                            if streaming_msg_id:
                                await stream_queue.put(
                                    TextMessageEndEvent(
                                        message_id=streaming_msg_id,
                                        timestamp=_get_timestamp(),
                                    )
                                )
                                streaming_msg_id = None

                            else:
                                await stream_queue.put(
                                    TextMessageChunkEvent(
                                        message_id=str(uuid4()),
                                        delta=content,
                                        timestamp=_get_timestamp(),
                                    )
                                )

            except Exception as e:
                await stream_queue.put(
                    RunErrorEvent(
                        message=repr(e),
                        timestamp=_get_timestamp(),
                    )
                )
                raise e

            else:
                await stream_queue.put(
                    RunFinishedEvent(
                        thread_id=incoming.thread_id,
                        run_id=incoming.run_id,
                        timestamp=_get_timestamp(),
                    )
                )

        # EventEncoder typed incompletely, so we need to ignore the type error
        encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]

        tg = asyncio.ensure_future(run_stream())

        try:
            while not tg.done() or not stream_queue.empty():
                with suppress(TimeoutError):
                    event = await asyncio.wait_for(stream_queue.get(), timeout=0.01)
                    yield encoder.encode(event)

            await tg

        except Exception as e:
            tg.cancel()
            yield encoder.encode(
                RunErrorEvent(
                    message=repr(e),
                    timestamp=_get_timestamp(),
                )
            )
            raise e

        else:
            await tg

    def build_asgi(self) -> "type[HTTPEndpoint]":
        """Build an ASGI endpoint for the AGUIStream."""
        # import here to avoid Starlette requirements in the main package
        from .asgi import build_asgi

        return build_asgi(self)


def _encode_context(context: dict[str, Any]) -> dict[str, Any]:
    """Drop all unserializable values from the context.

    It is required to share with AG-UI frontend application only data values.
    Any Python objects (like functions, classes, etc.) will be dropped from the context."""
    context = to_jsonable_python(context, fallback=lambda _: None, exclude_none=True) or {}
    return {k: v for k, v in context.items() if v is not None}
