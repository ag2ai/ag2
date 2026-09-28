# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from math import isfinite
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    AgentCapabilities,
    AudioPart,
    ContentPart,
    DataSource,
    DocumentPart,
    FileSource,
    ImagePart,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    StateSnapshotEvent,
    SubagentErrorEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TextPart,
    TokenUsage,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    UrlSource,
    VideoPart,
)
from ag_ui.encoder import EventEncoder
from fast_depends.library.serializer import SerializerProto
from pydantic_core import to_jsonable_python

from ag2 import Agent, MemoryStream, ToolResult, events
from ag2.config import ModelConfig
from ag2.context import strip_reserved_variables
from ag2.events import BinaryInput, BinaryType, DataInput, FileIdInput, TextInput, UrlInput, Usage
from ag2.hitl import HumanHook
from ag2.middleware.base import MiddlewareFactory
from ag2.observers import Observer
from ag2.tools.final import ClientTool
from ag2.tools.tool import Tool
from ag2.usage import UsageRecord, UsageReport

from .events import AGUIEvent
from .interrupts import (
    DEFAULT_RETENTION,
    ClientInterrupter,
    Retention,
    ServedTurn,
    ServedTurns,
    TurnOutput,
    interrupt_capabilities,
    serve_exchange,
    timestamp_ms,
    utc_now,
)

try:
    from starlette.endpoints import HTTPEndpoint
except ImportError:
    # Fallback to Any until Starlette is installed
    HTTPEndpoint = Any  # type: ignore[misc,assignment]

logger = logging.getLogger("ag2.ag_ui")

# The media part each kind of ag2 input travels as. An input of no particular
# kind is sent as a document, the one part that makes no claim about its bytes.
_PART_OF_KIND: dict[BinaryType, type[ImagePart | AudioPart | VideoPart | DocumentPart]] = {
    BinaryType.IMAGE: ImagePart,
    BinaryType.AUDIO: AudioPart,
    BinaryType.VIDEO: VideoPart,
    BinaryType.DOCUMENT: DocumentPart,
    BinaryType.BINARY: DocumentPart,
}


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
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        """Serve `agent`, holding a turn paused on a question for `retention`.

        `now` is the clock deadlines are read off, for tests that would
        otherwise have to outlast a retention bound to reach one.
        """
        self.__agent = agent
        self.__turns = ServedTurns(retention=retention, now=now)

    async def __aenter__(self) -> "AGUIStream":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel every turn this stream is still running."""
        await self.__turns.release_all()

    def capabilities(self) -> AgentCapabilities:
        """What this agent tells a client it can do, before any run starts."""
        return interrupt_capabilities(self.__agent.name)

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
    ) -> AsyncIterator[str]:
        """Run `incoming` and yield encoded AG-UI events.

        `accept` is the request's `Accept` header, selecting SSE or NDJSON.
        `hitl_hook` is where a question the agent asks goes — omit it and the
        question is put to the client as an interrupt instead.

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
    client_tools = []
    client_tools_names = set()
    for t in command.incoming.tools or ():
        func = t.model_dump(exclude_none=True)
        tool = ClientTool({"function": func})
        client_tools.append(tool)
        client_tools_names.add(tool.name)

    extracted_prompt, history_messages, current_turn = map_agui_messages_to_events(
        command, provider=provider_of(command.config or agent.config)
    )
    # A client that declares no version predates 1.0, and its schema reads a
    # tool result as a string only.
    predates_parts = command.incoming.protocol_version is None
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

        elif isinstance(event, events.ToolCallEvent):
            if event.name in client_tools_names:
                return

            await output.send(
                ToolCallStartEvent(
                    tool_call_id=event.id,
                    tool_call_name=event.name,
                    timestamp=_get_timestamp(),
                )
            )
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

    try:
        initial_vars = agent._agent_variables | command.variables
        if vars := _encode_context(initial_vars):
            await output.send(
                StateSnapshotEvent(
                    snapshot=vars,
                    timestamp=_get_timestamp(),
                )
            )

        # The client authors ``incoming.state``; it seeds this turn's variables
        # but must not reach the framework's own control-plane keys.
        client_state = strip_reserved_variables(command.incoming.state or {}, source="inbound AG-UI state")
        initial_state = client_state | initial_vars

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

        if (vars := _encode_context(result.context.variables)) != initial_state:
            await output.send(
                StateSnapshotEvent(
                    snapshot=vars,
                    timestamp=_get_timestamp(),
                )
            )

    except Exception as e:
        await output.send(
            RunErrorEvent(
                message=repr(e),
                timestamp=_get_timestamp(),
                usage=await _run_token_usage(stream),
            )
        )
        raise e

    else:
        await output.send(
            RunFinishedEvent(
                thread_id=output.thread_id,
                run_id=output.run_id,
                timestamp=_get_timestamp(),
                usage=await _run_token_usage(stream),
                outcome=output.success_outcome(),
            )
        )

    finally:
        # The exchange reading this turn ends on its terminating event, but
        # the channel is the turn's: closed here, once there is nothing more
        # to say, on every path including cancellation while held.
        await output.aclose()


# The task lifecycle events a delegation reaches the client through.
_TASK_LIFECYCLE = (events.TaskStarted, events.TaskCompleted, events.TaskFailed)


def map_task_event_to_ag_ui(
    event: events.TaskStarted | events.TaskCompleted | events.TaskFailed,
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
            timestamp=_get_timestamp(),
        )
    if isinstance(event, events.TaskCompleted):
        return SubagentFinishedEvent(
            subagent_run_id=event.task_id,
            result=to_jsonable_python(event.result, fallback=str),
            timestamp=_get_timestamp(),
        )
    # The run carries on: the delegating tool reports the failure to the
    # parent's model, which may well recover from it.
    return SubagentErrorEvent(
        subagent_run_id=event.task_id,
        message=str(event.error) or type(event.error).__name__,
        timestamp=_get_timestamp(),
    )


async def _run_token_usage(stream: MemoryStream) -> list[TokenUsage] | None:
    # Safe on the failure path: the stream awaits its subscribers on send, so
    # persistence has seen every usage event emitted before the exception.
    return map_usage_events_to_ag_ui(await stream.history.get_events())


def map_usage_events_to_ag_ui(usage_events: Iterable[events.BaseEvent]) -> list[TokenUsage] | None:
    """Attributed spend for a set of events, as AG-UI's per-(provider, model) list."""
    # Both AG-UI transports call this, so the two cannot compose attribution and
    # grouping differently. They differ only in where the events come from.
    return map_usage_records_to_ag_ui(UsageReport.from_events(usage_events).records)


def map_usage_records_to_ag_ui(records: Iterable[UsageRecord]) -> list[TokenUsage] | None:
    """Attributed spend, as AG-UI's per-(provider, model) list, in the protocol's accounting.

    Counts a provider did not report are omitted, never zero-filled.
    """
    # Records, not the report's by_model / by_provider: those are independent
    # maps, so the (provider, model) pair cannot be recovered from them, and each
    # drops what the other side did not label — where a sub-agent's spend lives.
    grouped: dict[tuple[str | None, str | None], list[Usage]] = {}
    for record in records:
        grouped.setdefault((record.provider, record.model), []).append(record.usage)

    # Pairs are never folded together: absent counts add as zero, so merging a
    # provider that reports reasoning tokens with one that does not would read as
    # a complete measurement. Within a pair the calls are summed, because there an
    # absent additive count does mean the provider had nothing to report.
    entries = []
    for (provider, model), usages in grouped.items():
        summed = sum(usages, Usage())
        input_tokens = _token_count(_input_total(provider, summed))
        output_tokens = _token_count(_output_total(provider, summed))
        entries.append(
            TokenUsage(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                # Computed, never copied: a provider's own total need not count
                # the way the two totals beside it now do.
                total_tokens=None if input_tokens is None or output_tokens is None else input_tokens + output_tokens,
                reasoning_tokens=_token_count(summed.thinking_tokens),
                cached_input_tokens=_token_count(summed.cache_read_input_tokens),
                cache_write_input_tokens=_token_count(summed.cache_creation_input_tokens),
            )
        )
    return entries or None


# The correction to AG-UI 1.0's accounting is made here, where usage leaves for
# the wire, and nowhere else. `Usage` keeps each provider's numbers as the
# provider reported them, because budgets and limiters read them that way;
# "fixing" the provider normalizers instead would shift every one of those.
# AG-UI's input and output are totals, and its cache and reasoning counts parts
# of them, so where a provider reports those beside a smaller count, they are
# added in here.

# Providers whose prompt count leaves out the tokens read from and written to
# the cache. Bedrock is not among them until a live call confirms that Converse
# counts the way Anthropic does.
_CACHE_OUTSIDE_PROMPT = frozenset({"anthropic"})

# Providers whose completion count leaves out the reasoning tokens:
# Gemini's `thoughts_token_count` sits beside `candidates_token_count`. xAI's
# reasoning count is not known to do either, so it is left as reported.
_REASONING_OUTSIDE_COMPLETION = frozenset({"google"})


def _input_total(provider: str | None, usage: Usage) -> float | None:
    if usage.prompt_tokens is None or provider not in _CACHE_OUTSIDE_PROMPT:
        return usage.prompt_tokens
    return usage.prompt_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)


def _output_total(provider: str | None, usage: Usage) -> float | None:
    if usage.completion_tokens is None or provider not in _REASONING_OUTSIDE_COMPLETION:
        return usage.completion_tokens
    return usage.completion_tokens + (usage.thinking_tokens or 0)


def _token_count(value: float | None) -> int | None:
    # The wire type admits only non-negative integers, and this runs on the
    # failure path before the run's own exception is re-raised — so a value the
    # wire would reject is omitted rather than left to raise over the real cause.
    if value is None or not isfinite(value) or value < 0:
        return None
    return int(value)


def provider_of(config: ModelConfig | None) -> str | None:
    """The provider `config` serves from, in ag2's vocabulary, or `None` if it does not say."""
    if config is None:
        return None
    try:
        return config.provider.value
    except NotImplementedError:
        return None


def map_agui_content_to_input(content: ContentPart, *, provider: str | None = None) -> events.Input | None:
    """One AG-UI content part as the ag2 input it carries, or `None` for a part to skip.

    `provider` is the run's, which a provider file handle has to belong to.
    """
    if isinstance(content, TextPart):
        return events.TextInput(content.text)

    match content:
        case DocumentPart():
            kind = BinaryType.DOCUMENT
        case AudioPart():
            kind = BinaryType.AUDIO
        case VideoPart():
            kind = BinaryType.VIDEO
        case ImagePart():
            kind = BinaryType.IMAGE
        case _:
            raise ValueError(f"Unexpected content type: {type(content).__name__}")

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
        if source.provider is not None and source.provider != provider:
            logger.warning(
                "skipping a %s part holding a file handle issued by %s: this run's provider is %s",
                content.type,
                source.provider,
                provider or "unknown",
            )
            return None
        inp = events.FileIdInput(source.value)
    else:
        raise ValueError(f"Unexpected source type: {type(source).__name__}")

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
    prompt, messages = [], []

    input_buffer: list[events.Input] = []
    for m in command.incoming.messages:
        if m.role == "user":
            input_buffer.extend(map_agui_parts_to_inputs(m.content, provider=provider))
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
            # An error is what the model must hear, whatever came with it.
            parts = [m.error] if m.error else map_agui_parts_to_inputs(m.content, provider=provider)
            messages.append(
                events.ToolResultsEvent([
                    events.ToolResultEvent(
                        parent_id=m.tool_call_id,
                        result=ToolResult(parts=parts),
                    )
                ])
            )

    return prompt, messages, input_buffer


def map_tool_result_to_ag_ui(result: ToolResult, serializer: SerializerProto) -> str | list[ContentPart]:
    """A tool result as a 1.0 client reads it: a lone text as a string, anything else in parts."""
    parts = [_content_part(part, serializer) for part in result.parts]
    if not parts:
        return ""
    if len(parts) == 1 and isinstance(parts[0], TextPart) and parts[0].metadata is None:
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


def _get_timestamp() -> int:
    return timestamp_ms()


def _encode_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unserializable values and the framework's reserved keys from the context."""
    if not context:
        return {}

    context = strip_reserved_variables(context, source="an outgoing AG-UI state snapshot", warn=False)
    context = to_jsonable_python(context, fallback=lambda _: None, exclude_none=True) or {}
    return {k: v for k, v in context.items() if v is not None}
