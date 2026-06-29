# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64decode
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    AudioInputContent,
    BaseEvent,
    BinaryInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContent,
    InputContentDataSource,
    InputContentUrlSource,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextInputContent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    VideoInputContent,
)
from ag_ui.encoder import EventEncoder
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from fast_depends.library.serializer import SerializerProto
from pydantic_core import to_jsonable_python

from ag2 import Agent, MemoryStream, ToolResult, events
from ag2.config import ModelConfig
from ag2.events import BinaryInput, BinaryType, DataInput, FileIdInput, TextInput, UrlInput
from ag2.hitl import HumanHook
from ag2.middleware.base import MiddlewareFactory
from ag2.observers import Observer
from ag2.tools.final import ClientTool
from ag2.tools.tool import Tool

from .events import AGUIEvent

try:
    from starlette.endpoints import HTTPEndpoint
except ImportError:
    # Fallback to Any until Starlette is installed
    HTTPEndpoint = Any  # type: ignore[misc,assignment]


class AGUIStream:
    def __init__(self, agent: Agent) -> None:
        self.__agent = agent

    def build_asgi(self) -> "type[HTTPEndpoint]":
        """Build an ASGI endpoint for the AGUIStream."""
        # import here to avoid Starlette requirements in the main package
        from .asgi import build_asgi

        return build_asgi(self)

    async def dispatch(
        self,
        incoming: RunAgentInput,
        *,
        variables: dict[str, Any] | None = None,
        prompt: Iterable[str] = (),
        dependencies: dict[Any, Any] | None = None,
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        hitl_hook: HumanHook | None = None,
        accept: str | None = None,
    ) -> AsyncIterator[str]:
        write_events_stream, read_events_stream = create_memory_object_stream[BaseEvent]()

        async with create_task_group() as tg:
            tg.start_soon(
                run_stream,
                AGStreamInput(
                    incoming=incoming,
                    variables=variables or {},
                    prompt=list(prompt),
                    dependencies=dependencies,
                    config=config,
                    tools=list(tools),
                    middleware=list(middleware),
                    observers=list(observers),
                    hitl_hook=hitl_hook,
                ),
                self.__agent,
                write_events_stream,
            )

            # EventEncoder typed incompletely, so we need to ignore the type error
            encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]

            async with read_events_stream:
                async for event in read_events_stream:
                    # ASYNC119: this is a true streaming generator that must hold the
                    # task group and stream open across yields; consumers are expected
                    # to use contextlib.aclosing for timely cleanup.
                    yield encoder.encode(event)  # noqa: ASYNC119


@dataclass(slots=True)
class AGStreamInput:
    incoming: RunAgentInput
    variables: dict[str, Any]
    prompt: list[str] = field(default_factory=list)
    dependencies: dict[Any, Any] | None = None
    config: ModelConfig | None = None
    tools: list[Tool] = field(default_factory=list)
    middleware: list[MiddlewareFactory] = field(default_factory=list)
    observers: list[Observer] = field(default_factory=list)
    hitl_hook: HumanHook | None = None


async def run_stream(
    command: AGStreamInput,
    agent: Agent,
    write_events_stream: MemoryObjectSendStream[BaseEvent],
) -> None:
    client_tools = []
    client_tools_names = set()
    for t in command.incoming.tools:
        func = t.model_dump(exclude_none=True)
        tool = ClientTool({"function": func})
        client_tools.append(tool)
        client_tools_names.add(tool.name)

    extracted_prompt, history_messages, current_turn = map_agui_messages_to_events(command)
    if extracted_prompt:
        command.prompt.extend(extracted_prompt)
    if client_tools:
        command.tools.extend(client_tools)

    stream = MemoryStream()
    await stream.history.replace(history_messages)

    streaming_msg_id: str | None = None
    reasoning_msg_id: str | None = None

    @stream.subscribe
    async def map_events_to_ag_ui(event: events.BaseEvent) -> None:
        nonlocal streaming_msg_id, reasoning_msg_id

        if reasoning_msg_id is not None and not isinstance(event, events.ModelReasoning):
            await write_events_stream.send(
                ReasoningMessageEndEvent(
                    message_id=reasoning_msg_id,
                    timestamp=_get_timestamp(),
                )
            )
            await write_events_stream.send(
                ReasoningEndEvent(
                    message_id=reasoning_msg_id,
                    timestamp=_get_timestamp(),
                )
            )
            reasoning_msg_id = None

        if isinstance(event, events.ModelReasoning):
            if not event.content:
                return

            if reasoning_msg_id is None:
                reasoning_msg_id = str(uuid4())
                await write_events_stream.send(
                    ReasoningStartEvent(
                        message_id=reasoning_msg_id,
                        timestamp=_get_timestamp(),
                    )
                )
                await write_events_stream.send(
                    ReasoningMessageStartEvent(
                        message_id=reasoning_msg_id,
                        role="reasoning",
                        timestamp=_get_timestamp(),
                    )
                )

            await write_events_stream.send(
                ReasoningMessageContentEvent(
                    message_id=reasoning_msg_id,
                    delta=event.content,
                    timestamp=_get_timestamp(),
                )
            )
            return

        if isinstance(event, events.ModelMessageChunk):
            if not event.content:
                return

            if streaming_msg_id is None:
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
                    delta=event.content,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.ModelMessage):
            if streaming_msg_id:
                await write_events_stream.send(
                    TextMessageEndEvent(
                        message_id=streaming_msg_id,
                        timestamp=_get_timestamp(),
                    )
                )
                streaming_msg_id = None

            elif event.content:
                await write_events_stream.send(
                    TextMessageChunkEvent(
                        message_id=str(uuid4()),
                        delta=event.content,
                        timestamp=_get_timestamp(),
                    )
                )

        elif isinstance(event, events.ClientToolCallEvent):
            await write_events_stream.send(
                ToolCallChunkEvent(
                    tool_call_id=event.id,
                    tool_call_name=event.name,
                    delta=event.arguments,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.ToolCallEvent):
            if event.name in client_tools_names:
                return

            await write_events_stream.send(
                ToolCallStartEvent(
                    tool_call_id=event.id,
                    tool_call_name=event.name,
                    timestamp=_get_timestamp(),
                )
            )
            await write_events_stream.send(
                ToolCallArgsEvent(
                    tool_call_id=event.id,
                    delta=event.arguments,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.ToolResultEvent):
            text_parts = []
            for p in event.result.parts:
                if isinstance(p, events.TextInput):
                    text_parts.append(p.content)
                elif isinstance(p, events.DataInput):
                    text_parts.append(agent._serializer.encode(p.data).decode())

            await write_events_stream.send(
                ToolCallResultEvent(
                    tool_call_id=event.parent_id,
                    content=_stringify_tool_result(event.result, agent._serializer),
                    message_id=str(uuid4()),
                    timestamp=_get_timestamp(),
                    role="tool",
                )
            )
            await write_events_stream.send(
                ToolCallEndEvent(
                    tool_call_id=event.parent_id,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.TaskStarted):
            await write_events_stream.send(StepStartedEvent(step_name=f"task:{event.agent_name}"))

        elif isinstance(event, events.TaskCompleted):
            await write_events_stream.send(StepFinishedEvent(step_name=f"task:{event.agent_name}"))

        elif isinstance(event, AGUIEvent):
            await write_events_stream.send(event.event)

    async with write_events_stream:
        try:
            await write_events_stream.send(
                RunStartedEvent(
                    thread_id=command.incoming.thread_id,
                    run_id=command.incoming.run_id,
                    timestamp=_get_timestamp(),
                )
            )

            initial_vars = agent._agent_variables | command.variables
            if vars := _encode_context(initial_vars):
                await write_events_stream.send(
                    StateSnapshotEvent(
                        snapshot=vars,
                        timestamp=_get_timestamp(),
                    )
                )

            initial_state = (command.incoming.state or {}) | initial_vars

            result = await agent.ask(
                *current_turn,
                prompt=command.prompt,
                tools=command.tools,
                variables=initial_state,
                dependencies=command.dependencies,
                config=command.config,
                middleware=command.middleware,
                observers=command.observers,
                hitl_hook=command.hitl_hook,
                stream=stream,
            )

            if (vars := _encode_context(result.context.variables)) != initial_state:
                await write_events_stream.send(
                    StateSnapshotEvent(
                        snapshot=vars,
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


def map_agui_content_to_input(content: InputContent) -> events.Input:
    if isinstance(content, BinaryInputContent):
        raise ValueError(
            "AG-UI 'binary' content type is deprecated; "
            "use ImageInputContent / AudioInputContent / "
            "VideoInputContent / DocumentInputContent instead."
        )

    if isinstance(content, TextInputContent):
        return events.TextInput(content.text)

    match content:
        case DocumentInputContent():
            kind = BinaryType.DOCUMENT
        case AudioInputContent():
            kind = BinaryType.AUDIO
        case VideoInputContent():
            kind = BinaryType.VIDEO
        case ImageInputContent():
            kind = BinaryType.IMAGE
        case _:
            raise ValueError(f"Unexpected content type: {type(content).__name__}")

    source = content.source
    if isinstance(source, InputContentDataSource):
        inp = events.BinaryInput(
            b64decode(source.value),
            media_type=source.mime_type,
            kind=kind,
        )
    elif isinstance(source, InputContentUrlSource):
        inp = events.UrlInput(source.value, kind=kind)
    else:
        raise ValueError(f"Unexpected source type: {type(source).__name__}")

    if content.metadata:
        inp.metadata = content.metadata
    return inp


def map_agui_messages_to_events(
    command: AGStreamInput,
) -> tuple[list[str], list[events.BaseEvent], list[events.Input]]:
    """Translate AG-UI history into the parts ``run_stream`` hands to the agent.

    Returns the system/developer ``prompt`` strings, the prior-turn ``history``
    events, and the parts of the current user turn (trailing run of
    ``UserMessage`` entries). The current turn is kept separate because
    ``Agent.ask`` always constructs a ``ModelRequest`` from ``*msg`` and sends
    it as the loop's initial event — putting the current turn there gives the
    LLM a meaningful ``messages[-1]`` instead of an empty placeholder.
    """
    prompt, messages = [], []

    input_buffer: list[events.Input] = []
    for m in command.incoming.messages:
        if m.role == "user":
            content = m.content
            if isinstance(content, str):
                input_buffer.append(events.TextInput(content))
                continue

            for c in content:
                input_buffer.append(map_agui_content_to_input(c))

            continue

        if input_buffer:
            messages.append(events.ModelRequest(input_buffer))
            input_buffer = []

        if m.role in ["system", "developer"]:
            prompt.append(m.content)

        elif m.role == "assistant":
            tool_calls = [
                events.ToolCallEvent(
                    id=t.id,
                    name=t.function.name,
                    arguments=t.function.arguments,
                )
                for t in (m.tool_calls or ())
            ]

            messages.append(
                events.ModelResponse(
                    events.ModelMessage(m.content) if m.content else None,
                    tool_calls=events.ToolCallsEvent(tool_calls),
                )
            )

        elif m.role == "reasoning":
            if m.content:
                messages.append(events.ModelReasoning(m.content))

        elif m.role == "tool":
            messages.append(
                events.ToolResultsEvent([
                    events.ToolResultEvent(
                        parent_id=m.tool_call_id,
                        result=ToolResult([m.error or m.content]),
                    )
                ])
            )

    return prompt, messages, input_buffer


def _stringify_tool_result(result: ToolResult, serializer: SerializerProto) -> str:
    """Flatten a multi-part ``ToolResult`` into a string for the AG-UI wire format.

    AG-UI's ``ToolCallResultEvent.content`` is a plain string, but AG2 tool
    results are now structured lists of ``Input`` parts (text, data, binary,
    urls, file-ids). Collapse them here so any kind of tool return still
    surfaces in the stream.
    """
    chunks: list[str] = []
    for part in result.parts:
        if isinstance(part, TextInput):
            chunks.append(part.content)
        elif isinstance(part, DataInput):
            chunks.append(serializer.encode(part.data).decode())
        elif isinstance(part, UrlInput):
            chunks.append(part.url)
        elif isinstance(part, FileIdInput):
            chunks.append(f"[file:{part.file_id}]")
        elif isinstance(part, BinaryInput):
            chunks.append(f"[binary:{part.media_type} {len(part.data)}B]")
        else:
            chunks.append(repr(part))
    if len(chunks) == 1:
        return chunks[0]
    return "\n".join(chunks)


def _get_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _encode_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Drop all unserializable values from the context.

    It is required to share with AG-UI frontend application only data values.
    Any Python objects (like functions, classes, etc.) will be dropped from the context."""
    if not context:
        return {}

    context = to_jsonable_python(context, fallback=lambda _: None, exclude_none=True) or {}
    return {k: v for k, v in context.items() if v is not None}
