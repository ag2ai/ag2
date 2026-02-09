# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

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
        data: RunAgentInput,
        *,
        context: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> AsyncIterator[str]:
        state = ContextVariables()
        # add agent initial context
        state.update(self.__agent.context_variables.to_dict())
        # add frontend-passed context
        state.update(data.state or {})
        # add manual-passed context
        state.update(context or {})

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
            context=state.data,
            client_tools=client_tools,
        )

        # EventEncoder typed incompletely, so we need to ignore the type error
        encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]

        try:
            yield encoder.encode(
                RunStartedEvent(
                    thread_id=data.thread_id,
                    run_id=data.run_id,
                    timestamp=_get_timestamp(),
                )
            )

            if state.data != data.state:
                yield encoder.encode(
                    StateSnapshotEvent(
                        snapshot=state.data,
                        timestamp=_get_timestamp(),
                    )
                )

            async for response in self.service(message):
                msg_id = str(uuid4())

                if ctx := response.context:
                    yield encoder.encode(
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
                        yield encoder.encode(
                            ToolCallResultEvent(
                                tool_call_id=tool_response["tool_call_id"],
                                content=tool_response["content"],
                                message_id=msg_id,
                                timestamp=_get_timestamp(),
                                role="tool",
                            )
                        )
                        yield encoder.encode(
                            ToolCallEndEvent(
                                tool_call_id=tool_response["tool_call_id"],
                                timestamp=_get_timestamp(),
                            )
                        )

                    for tool_call in msg.get("tool_calls", []):
                        func = tool_call["function"]

                        if (name := func.get("name")) in client_tools_names:
                            yield encoder.encode(
                                ToolCallChunkEvent(
                                    parent_message_id=msg_id,
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                    delta=func.get("arguments"),
                                    timestamp=_get_timestamp(),
                                )
                            )

                        else:
                            yield encoder.encode(
                                ToolCallStartEvent(
                                    tool_call_id=tool_call.get("id"),
                                    tool_call_name=name,
                                    timestamp=_get_timestamp(),
                                )
                            )
                            yield encoder.encode(
                                ToolCallArgsEvent(
                                    tool_call_id=tool_call.get("id"),
                                    delta=func.get("arguments"),
                                    timestamp=_get_timestamp(),
                                )
                            )

                    if content and not has_tool_result:
                        yield encoder.encode(
                            TextMessageChunkEvent(
                                message_id=msg_id,
                                delta=content,
                                timestamp=_get_timestamp(),
                            )
                        )

        except Exception as e:
            yield encoder.encode(
                RunErrorEvent(
                    message=repr(e),
                    timestamp=_get_timestamp(),
                )
            )
            raise e

        else:
            yield encoder.encode(
                RunFinishedEvent(
                    thread_id=data.thread_id,
                    run_id=data.run_id,
                    timestamp=_get_timestamp(),
                )
            )

    def build_asgi(self) -> "type[HTTPEndpoint]":
        """Build an ASGI endpoint for the AGUIStream."""
        # import here to avoid Starlette requirements in the main package
        from .asgi import build_asgi

        return build_asgi(self)
