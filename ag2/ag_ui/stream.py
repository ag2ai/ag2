# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ag_ui.core import (
    ActivityMessage,
    AgentCapabilities,
    AssistantMessage,
    AudioPart,
    ContentPart,
    DataSource,
    DeveloperMessage,
    DocumentPart,
    FileSource,
    ImagePart,
    ReasoningEncryptedValueEvent,
    ReasoningEndEvent,
    ReasoningMessage,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    StateSnapshotEvent,
    SubagentErrorEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    SystemMessage,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TextPart,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UrlSource,
    UserMessage,
    VideoPart,
)
from ag_ui.core import (
    Context as ContextEntry,
)
from ag_ui.encoder import EventEncoder
from fast_depends.library.serializer import SerializerProto
from pydantic_core import PydanticSerializationError, to_jsonable_python
from typing_extensions import assert_never

from ag2 import Agent, Context, MemoryStream, ToolResult, events
from ag2.config import ModelConfig
from ag2.context import strip_reserved_variables
from ag2.events import BinaryInput, BinaryType, DataInput, FileIdInput, TextInput, UrlInput, UsageEvent
from ag2.hitl import HumanHook
from ag2.middleware.base import AgentTurn, BaseMiddleware, Middleware, MiddlewareFactory
from ag2.observers import Observer
from ag2.tools.final import ClientTool
from ag2.tools.tool import Tool
from ag2.usage import collect_usage_events

from .capabilities import served_capabilities
from .events import AGUIEvent
from .input_acceptance import accepts_input
from .interrupts import (
    DEFAULT_RETENTION,
    ClientInterrupter,
    Retention,
    ServedTurn,
    ServedTurns,
    TurnOutput,
    drive_run,
    serve_exchange,
    timestamp_ms,
    utc_now,
)
from .provider import is_same_provider, provider_of
from .thought_signature import encrypted_signature_of, restore_tool_call

if TYPE_CHECKING:
    from starlette.endpoints import HTTPEndpoint

logger = logging.getLogger(__name__)

# The media part each kind of ag2 input travels as. An input of no particular
# kind is sent as a document, the one part that makes no claim about its bytes.
_PART_OF_KIND: dict[BinaryType, type[ImagePart | AudioPart | VideoPart | DocumentPart]] = {
    BinaryType.IMAGE: ImagePart,
    BinaryType.AUDIO: AudioPart,
    BinaryType.VIDEO: VideoPart,
    BinaryType.DOCUMENT: DocumentPart,
    BinaryType.BINARY: DocumentPart,
}


def client_context_prompt(entries: Sequence[ContextEntry]) -> str | None:
    """A run input's `context`, as the prompt block the model reads it in.

    A heading, then one `description: value` line per entry, each value exactly
    as the client sent it. `None` when there are no entries.
    """
    if not entries:
        return None
    lines = "\n".join(f"- {entry.description}: {entry.value}" for entry in entries)
    return f"## Context from the application\n\n{lines}"


class AGUIStream:
    """Serve an `Agent` over AG-UI.

    A turn's lifetime belongs to this object, not to the HTTP exchange that
    started it: an agent that asks a human a question is held here until the
    client answers. Call `aclose` on shutdown, or use the stream as an async
    context manager, so a turn still waiting is cancelled.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        retention: Retention = DEFAULT_RETENTION,
        require_resume_proof: bool = False,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        """Serve `agent`, holding a turn paused on a question for `retention`.

        Every interrupt is issued with a proof in its `metadata`. A resume that
        carries one must carry the one issued; `require_resume_proof` also
        refuses a resume that carries none, which a client that does not copy an
        interrupt's metadata into its answer cannot satisfy.

        `now` is the clock deadlines are read off, for tests that would
        otherwise have to outlast a retention bound to reach one.
        """
        self.__agent = agent
        self.__turns = ServedTurns(retention=retention, require_proof=require_resume_proof, now=now)

    async def __aenter__(self) -> "AGUIStream":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel every turn this stream is still running."""
        await self.__turns.release_all()

    def capabilities(self) -> AgentCapabilities:
        """What this agent tells a client it can do, before any run starts.

        Read off the agent: a hook passed to one `dispatch` is not known here.
        """
        return served_capabilities(self.__agent, client_tools=True, state_snapshots=True)

    def build_asgi(self) -> "type[HTTPEndpoint]":
        """Build an ASGI endpoint serving this stream: POST runs, GET capabilities."""
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
        context_prompt: Callable[[Sequence[ContextEntry]], str | None] | None = client_context_prompt,
    ) -> AsyncIterator[str]:
        """Run `incoming` and yield encoded AG-UI events.

        `accept` is the request's `Accept` header. The stream is SSE whatever it says.
        `hitl_hook` is where a question the agent asks goes — omit it and the
        question is put to the client as an interrupt instead.
        `context_prompt` renders the input's `context` entries into one prompt
        block for the model; `None` leaves them out.

        Wrap the returned iterator in `contextlib.aclosing`: it holds a
        channel open across yields.
        """
        command = AGStreamInput(
            incoming=incoming,
            variables=variables or {},
            prompt=list(prompt),
            dependencies=dependencies,
            config=config,
            tools=list(tools),
            middleware=list(middleware),
            observers=list(observers),
            hitl_hook=hitl_hook,
            context_prompt=context_prompt,
        )

        # EventEncoder typed incompletely, so we need to ignore the type error
        encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]

        async for chunk in serve_exchange(self.__turns, incoming, encoder, partial(self.__start, command)):
            # ASYNC119: a true streaming generator, holding its channel open
            # across yields; consumers are expected to use contextlib.aclosing.
            yield chunk  # noqa: ASYNC119

    def __start(self, command: "AGStreamInput", output: TurnOutput) -> ServedTurn:
        turn = ServedTurn(output)
        interrupter = None if self.__answers_in_process(command.hitl_hook) else ClientInterrupter(turn, self.__turns)
        # Started as a task the server owns rather than inside this request's
        # scope: the turn outlives the exchange, so the exchange must not own it.
        self.__turns.track(turn, turn.start(run_stream(command, self.__agent, output, interrupter)))
        return turn

    def __answers_in_process(self, hitl_hook: HumanHook | None) -> bool:
        # Read off what was supplied, never off the core's "nobody to ask"
        # default: only a run that passed no hook has its question sent out.
        return hitl_hook is not None or self.__agent._hitl_hook is not None


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
    context_prompt: Callable[[Sequence[ContextEntry]], str | None] | None = client_context_prompt


async def run_stream(
    command: AGStreamInput,
    agent: Agent,
    output: TurnOutput,
    interrupter: ClientInterrupter | None = None,
) -> None:
    """Run one served turn, writing its events to `output`.

    `interrupter` is where a question the agent asks goes when the caller
    supplied no hook of its own; `None` leaves the agent's own human-input
    arrangements untouched.
    """
    await drive_run(output, _serve_turn(command, agent, output, interrupter))


async def _serve_turn(
    command: AGStreamInput,
    agent: Agent,
    output: TurnOutput,
    interrupter: ClientInterrupter | None,
) -> None:
    client_tools = []
    client_tools_names = set()
    for t in command.incoming.tools or ():
        func = t.model_dump(exclude_none=True)
        tool = ClientTool({"function": func})
        client_tools.append(tool)
        client_tools_names.add(tool.name)

    # Inside the run rather than ahead of it: input that cannot be mapped fails
    # a run that has already started, and that failure has to reach the client.
    extracted_prompt, history_messages, current_turn = map_agui_messages_to_events(
        command, provider=provider_of(command.config or agent.config), config=command.config or agent.config
    )
    # A client that declares no version predates 1.0, and its schema reads a
    # tool result as a string only.
    predates_parts = command.incoming.protocol_version is None
    if extracted_prompt:
        command.prompt.extend(extracted_prompt)
    # Prose for the model, not state: the application shares it to be read.
    # Added to the prompt the turn resolves rather than passed as one, which
    # would stand in for the agent's own.
    if command.context_prompt is not None and (shared := command.context_prompt(command.incoming.context or [])):
        command.middleware.append(Middleware(_SharedContext, block=shared))
    if client_tools:
        command.tools.extend(client_tools)

    stream = MemoryStream()
    await stream.history.replace(history_messages)
    # Metered as it is spent, not read back from history: a turn carried by
    # several runs reports each run's own share.
    stream.where(UsageEvent).subscribe(collect_usage_events(output.usage))

    streaming_msg_id: str | None = None
    reasoning_msg_id: str | None = None
    # A signature belongs to its call and may only follow the call's start: a
    # consumer may drop a value whose entity it has not seen. A client tool's
    # call is announced later, from its own event.
    signatures: dict[str, str] = {}

    @stream.subscribe
    async def map_events_to_ag_ui(event: events.BaseEvent) -> None:
        nonlocal streaming_msg_id, reasoning_msg_id

        if reasoning_msg_id is not None and not isinstance(event, events.ModelReasoning):
            await output.send(
                ReasoningMessageEndEvent(
                    message_id=reasoning_msg_id,
                    timestamp=_get_timestamp(),
                )
            )
            await output.send(
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
                await output.send(
                    ReasoningStartEvent(
                        message_id=reasoning_msg_id,
                        timestamp=_get_timestamp(),
                    )
                )
                await output.send(
                    ReasoningMessageStartEvent(
                        message_id=reasoning_msg_id,
                        role="reasoning",
                        timestamp=_get_timestamp(),
                    )
                )

            await output.send(
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
                await output.send(
                    TextMessageStartEvent(
                        message_id=streaming_msg_id,
                        timestamp=_get_timestamp(),
                    )
                )

            await output.send(
                TextMessageContentEvent(
                    message_id=streaming_msg_id,
                    delta=event.content,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.ModelMessage):
            if streaming_msg_id:
                await output.send(
                    TextMessageEndEvent(
                        message_id=streaming_msg_id,
                        timestamp=_get_timestamp(),
                    )
                )
                streaming_msg_id = None

            elif event.content:
                await output.send(
                    TextMessageChunkEvent(
                        message_id=str(uuid4()),
                        delta=event.content,
                        timestamp=_get_timestamp(),
                    )
                )

        elif isinstance(event, events.ClientToolCallEvent):
            await output.send(
                ToolCallChunkEvent(
                    tool_call_id=event.id,
                    tool_call_name=event.name,
                    delta=event.arguments,
                    timestamp=_get_timestamp(),
                )
            )
            await _send_signature(output, signatures, event.id)

        elif isinstance(event, events.ToolCallEvent):
            if (signature := encrypted_signature_of(event)) is not None:
                signatures[event.id] = signature
            if event.name in client_tools_names:
                return

            await output.send(
                ToolCallStartEvent(
                    tool_call_id=event.id,
                    tool_call_name=event.name,
                    timestamp=_get_timestamp(),
                )
            )
            await _send_signature(output, signatures, event.id)
            await output.send(
                ToolCallArgsEvent(
                    tool_call_id=event.id,
                    delta=event.arguments,
                    timestamp=_get_timestamp(),
                )
            )
            # Closed as soon as its arguments are complete, not after it runs: a
            # call paused on a question ends its run with the call still pending,
            # and clients refuse a RUN_FINISHED while a call is open. The result
            # follows under the same id, possibly in a later run.
            await output.send(
                ToolCallEndEvent(
                    tool_call_id=event.id,
                    timestamp=_get_timestamp(),
                )
            )

        elif isinstance(event, events.ToolResultEvent):
            parts = map_tool_result_to_ag_ui(event.result, agent._serializer)
            await output.send(
                ToolCallResultEvent(
                    tool_call_id=event.parent_id,
                    content=downgrade_tool_result(parts) if predates_parts else parts,
                    message_id=str(uuid4()),
                    timestamp=_get_timestamp(),
                    role="tool",
                )
            )

        elif isinstance(event, _TASK_LIFECYCLE):
            await output.send(map_task_event_to_ag_ui(event))

        elif isinstance(event, AGUIEvent):
            await output.send(event.event)

    # A snapshot replaces the client's state wholesale, so each one sent is the
    # whole of it: the client's keys and the server's variables together, and
    # only when that differs from what the client already holds. A state that
    # is not an object has no keys to merge variables into; it seeds nothing
    # and is left as the client sent it.
    incoming_state = {} if command.incoming.state is None else command.incoming.state
    shares_state = isinstance(incoming_state, dict)
    # The client authors ``incoming.state``; it seeds this turn's variables
    # but must not reach the framework's own control-plane keys.
    client_state = strip_reserved_variables(incoming_state, source="inbound AG-UI state") if shares_state else {}
    initial_state = client_state | agent._agent_variables | command.variables

    held_by_client = _encode_context(incoming_state) if shares_state else None
    if shares_state and (opening := _encode_context(initial_state)) != held_by_client:
        await output.send(StateSnapshotEvent(snapshot=opening, timestamp=_get_timestamp()))
        held_by_client = opening

    with ExitStack() as stack:
        if interrupter is not None:
            # Registered *before* `ask` so it runs ahead of the "nobody
            # could be asked" default the agent registers for itself.
            stack.enter_context(
                stream.where(events.HumanInputRequest).sub_scope(interrupter, interrupt=True),
            )

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

    if shares_state and (closing := _encode_context(result.context.variables)) != held_by_client:
        await output.send(StateSnapshotEvent(snapshot=closing, timestamp=_get_timestamp()))


async def _send_signature(output: "TurnOutput", signatures: dict[str, str], call_id: str) -> None:
    if (value := signatures.pop(call_id, None)) is not None:
        await output.send(
            ReasoningEncryptedValueEvent(
                subtype="tool-call",
                entity_id=call_id,
                encrypted_value=value,
                timestamp=_get_timestamp(),
            )
        )


# The task lifecycle events a delegation reaches the client through.
_TASK_LIFECYCLE = (
    events.TaskStarted,
    events.TaskCompleted,
    events.TaskFailed,
    events.TaskCancelled,
    events.TaskExpired,
)


def map_task_event_to_ag_ui(
    event: events.TaskStarted | events.TaskCompleted | events.TaskFailed | events.TaskCancelled | events.TaskExpired,
) -> SubagentStartedEvent | SubagentFinishedEvent | SubagentErrorEvent:
    """One delegation's lifecycle event as the subagent invocation event the client reads."""
    # Under the task's own id: two parallel delegations to one agent must be
    # told apart, and its name cannot do that. No usage rides on these: the
    # run's own total already holds the delegated spend.
    if isinstance(event, events.TaskStarted):
        return SubagentStartedEvent(
            subagent_run_id=event.task_id,
            name=event.agent_name,
            description=event.objective,
            parent_tool_call_id=event.parent_tool_call_id,
            timestamp=_get_timestamp(),
        )
    if isinstance(event, events.TaskCompleted):
        return SubagentFinishedEvent(
            subagent_run_id=event.task_id,
            result=to_jsonable_python(event.result, fallback=str),
            timestamp=_get_timestamp(),
        )
    # A stopped invocation did not succeed, and the protocol has no outcome
    # for it on SUBAGENT_FINISHED: success and suspension are all it names.
    if isinstance(event, events.TaskCancelled):
        message = f"cancelled: {event.reason}" if event.reason else "cancelled"
    elif isinstance(event, events.TaskExpired):
        message = "expired"
    else:
        # The run carries on: the delegating tool reports the failure to the
        # parent's model, which may well recover from it.
        message = str(event.error) or type(event.error).__name__
    return SubagentErrorEvent(subagent_run_id=event.task_id, message=message, timestamp=_get_timestamp())


def map_agui_content_to_input(content: ContentPart, *, provider: str | None = None) -> events.Input | None:
    """One AG-UI content part as the ag2 input it carries, or `None` for a part to skip.

    `provider` is the run's, which a provider file handle has to belong to.
    """
    match content:
        case TextPart():
            return events.TextInput(content.text)
        case DocumentPart():
            kind = BinaryType.DOCUMENT
        case AudioPart():
            kind = BinaryType.AUDIO
        case VideoPart():
            kind = BinaryType.VIDEO
        case ImagePart():
            kind = BinaryType.IMAGE
        case _:
            assert_never(content)

    source = content.source
    inp: events.Input
    if isinstance(source, DataSource):
        inp = events.BinaryInput(
            b64decode(source.value),
            media_type=source.mime_type,
            kind=kind,
        )
    elif isinstance(source, UrlSource):
        inp = events.UrlInput(source.value, kind=kind)
    elif isinstance(source, FileSource):
        # A handle is opaque and only the provider that minted it can resolve
        # it. An untagged one is taken to be the run's own, since the client
        # need not say; one tagged for another provider is useless here, and
        # the protocol forbids failing the run over it. Never log the value.
        if source.provider is not None and not is_same_provider(source.provider, provider):
            logger.warning(
                "skipping a %s part holding a file handle issued by %s: this run's provider is %s",
                content.type,
                source.provider,
                provider or "unknown",
            )
            return None
        inp = events.FileIdInput(source.value)
    else:
        assert_never(source)

    if content.metadata:
        inp.metadata = content.metadata
    return inp


def map_agui_parts_to_inputs(content: str | list[ContentPart], *, provider: str | None = None) -> list[events.Input]:
    """A message body, plain or in parts, as the ag2 inputs it carries."""
    if isinstance(content, str):
        return [events.TextInput(content)]
    return [inp for c in content if (inp := map_agui_content_to_input(c, provider=provider)) is not None]


def map_agui_messages_to_events(
    command: AGStreamInput,
    *,
    provider: str | None = None,
    config: ModelConfig | None = None,
) -> tuple[list[str], list[events.BaseEvent], list[events.Input]]:
    """Translate AG-UI history into the parts `run_stream` hands to the agent.

    Returns the system/developer `prompt` strings, the prior-turn `history`
    events, and the parts of the current user turn (trailing run of
    `UserMessage` entries). The current turn is kept separate because
    `Agent.ask` always constructs a `ModelRequest` from `*msg` and sends
    it as the loop's initial event — putting the current turn there gives the
    LLM a meaningful `messages[-1]` instead of an empty placeholder.

    `provider` is the run's, resolved where its configuration is known; a
    provider file handle issued by anyone else is skipped.
    """
    prompt: list[str] = []
    messages: list[events.BaseEvent] = []
    # A tool message carries only the call id; the name is the restated call's.
    tool_names: dict[str, str] = {}

    input_buffer: list[events.Input] = []
    for m in command.incoming.messages:
        if isinstance(m, UserMessage):
            input_buffer.extend(_accepted_parts(m.content, provider=provider, config=config, position="user"))
            continue

        if input_buffer:
            messages.append(events.ModelRequest(input_buffer))
            input_buffer = []

        match m:
            case SystemMessage() | DeveloperMessage():
                prompt.append(m.content)

            case AssistantMessage():
                tool_calls: list[events.ToolCallEvent] = []
                for t in m.tool_calls or ():
                    tool_calls.append(
                        restore_tool_call(
                            provider,
                            id=t.id,
                            name=t.function.name,
                            arguments=t.function.arguments,
                            encrypted_value=t.encrypted_value,
                        )
                    )
                tool_names.update((t.id, t.name) for t in tool_calls)

                messages.append(
                    events.ModelResponse(
                        events.ModelMessage(m.content) if m.content else None,
                        tool_calls=events.ToolCallsEvent(tool_calls),
                    )
                )

            case ReasoningMessage():
                if m.content:
                    messages.append(events.ModelReasoning(m.content))

            case ToolMessage():
                # An error is what the model must hear, and leads; what came with
                # it is kept, so a partial result survives. Still answered when
                # every part was skipped: the call is owed a result, and one of
                # no parts is the empty string.
                parts = _accepted_parts(m.content, provider=provider, config=config, position="tool")
                if m.error:
                    parts = [TextInput(m.error), *parts]
                elif not parts:
                    parts = [TextInput("")]
                messages.append(
                    events.ToolResultsEvent([
                        events.ToolResultEvent(
                            parent_id=m.tool_call_id,
                            name=tool_names.get(m.tool_call_id),
                            result=ToolResult(parts=parts),
                        )
                    ])
                )

            case ActivityMessage():
                # Not part of the model's conversation.
                pass

            case _:
                assert_never(m)

    return prompt, messages, input_buffer


def _accepted_parts(
    content: str | list[ContentPart], *, provider: str | None, config: ModelConfig | None, position: str
) -> list[events.Input]:
    result = []
    for part in map_agui_parts_to_inputs(content, provider=provider):
        if accepts_input(config, position, part):
            result.append(part)
            continue
        kind = part.kind.value if isinstance(part, (BinaryInput, UrlInput)) else "file"
        media_type = part.media_type if isinstance(part, BinaryInput) else "unknown"
        logger.warning("skipping %s part (%s) for %s in a %s message", kind, media_type, provider, position)
    return result


def map_tool_result_to_ag_ui(result: ToolResult, serializer: SerializerProto) -> str | list[ContentPart]:
    """A tool result as a 1.0 client reads it: a lone text as a string, anything else in parts."""
    parts = [_content_part(part, serializer) for part in result.parts]
    if not parts:
        return ""
    if (
        len(parts) == 1
        and isinstance(parts[0], TextPart)
        and isinstance(parts[0].text, str)
        and parts[0].metadata is None
    ):
        return parts[0].text
    return parts


def downgrade_tool_result(content: str | list[ContentPart]) -> str:
    """A tool result for a client predating 1.0, which reads only a string.

    Its text, in order. Media cannot be put into a string without inventing
    something in their place, so they are dropped, and the loss is logged.
    """
    if isinstance(content, str):
        return content
    dropped = sorted({part.type for part in content if not isinstance(part, TextPart)})
    if dropped:
        logger.warning(
            "dropping the %s parts of a tool result for an AG-UI client that declares no protocol version; "
            "upgrade the client to @ag-ui/* 1.0 to receive them",
            ", ".join(dropped),
        )
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def _content_part(part: events.Input, serializer: SerializerProto) -> ContentPart:
    metadata = part.metadata or None
    if isinstance(part, TextInput):
        return TextPart(text=part.content, metadata=metadata)
    if isinstance(part, DataInput):
        # The protocol has no JSON part: structured output travels as its text.
        return TextPart(text=serializer.encode(part.data).decode(), metadata=metadata)
    if isinstance(part, UrlInput):
        return _PART_OF_KIND[part.kind](source=UrlSource(value=part.url), metadata=metadata)
    if isinstance(part, BinaryInput):
        source = DataSource(value=b64encode(part.data).decode(), mime_type=part.media_type)
        return _PART_OF_KIND[part.kind](source=source, metadata=metadata)
    if isinstance(part, FileIdInput):
        # Named as a file at a provider, never as something to fetch. Which
        # provider is not recorded on the input, so it is not claimed here.
        return DocumentPart(source=FileSource(value=part.file_id), metadata=metadata)
    raise TypeError(f"no AG-UI content part for {type(part).__name__}")


class _SharedContext(BaseMiddleware):
    """Adds the application's shared context to the prompt of the turn it runs in."""

    def __init__(self, event: events.BaseEvent, context: Context, *, block: str) -> None:
        super().__init__(event, context)
        self._block = block

    async def on_turn(self, call_next: AgentTurn, event: events.BaseEvent, context: Context) -> events.ModelResponse:
        context.prompt.append(self._block)
        return await call_next(event, context)


def _get_timestamp() -> int:
    return timestamp_ms()


def _encode_context(context: dict[str, Any]) -> dict[str, Any]:
    """The context as state: the framework's reserved keys and unserializable values left out.

    A `None` value is data, and is kept.
    """
    encoded = {}
    for key, value in strip_reserved_variables(context, source="an outgoing AG-UI state snapshot", warn=False).items():
        try:
            encoded[key] = to_jsonable_python(value)
        except PydanticSerializationError:
            # Server-side values — a client, a connection — have no JSON form,
            # and the client has no use for one.
            continue
    return encoded
