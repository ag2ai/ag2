# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""One run through `AGUIStream`: how it starts, streams, ends and fails, and how long its turn lives."""

import asyncio
import gc
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import pytest
from ag_ui.core import (
    AssistantMessage,
    DataSource,
    Event,
    ImagePart,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
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
    UserMessage,
)
from dirty_equals import IsInt, IsPartialDict, IsStr

from ag2 import Agent, ToolResult
from ag2.ag_ui import UNSUPPORTED_PROTOCOL_VERSION, AGUIStream
from ag2.events import ModelMessage, ModelMessageChunk, ModelResponse, ToolCallEvent, ToolCallsEvent
from ag2.middleware import Middleware
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import tool
from test.ag_ui.harness import (
    decode_events,
    dispatch_events,
    dispatch_run,
    each,
    events_of_failing_run,
    exploding_agent,
    kinds_of,
    recording_history,
    run_input,
    sole,
    weather_tool,
    wire,
)

pytestmark = pytest.mark.asyncio

# Every wait below is on an event another task sets. The bound is only so a
# regression fails the test instead of hanging CI until the suite timeout.
_NEVER = 5.0

# Not base64: decoding it fails while AG-UI input becomes ag2 input.
_MALFORMED = ImagePart(source=DataSource(value="a", mime_type="image/png"))


async def _events(config: TestConfig) -> list[Any]:
    return await dispatch_events(
        AGUIStream(Agent("test_agent", config=config)), run_input(UserMessage(id="m1", content="hi"))
    )


class TestASuccessfulRun:
    async def test_it_is_identified_by_its_ids_and_declares_the_version_ag2_speaks(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello!"))
        incoming = run_input(UserMessage(id="m1", content="Hello, how are you?"))

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert kinds_of(events) == [RunStartedEvent, TextMessageChunkEvent, RunFinishedEvent]
        assert wire(events[0]) == {
            "type": "RUN_STARTED",
            "threadId": incoming.thread_id,
            "runId": incoming.run_id,
            "protocolVersion": "1.0",
        }
        assert wire(events[-1]) == {
            "type": "RUN_FINISHED",
            "threadId": incoming.thread_id,
            "runId": incoming.run_id,
            "outcome": {"type": "success"},
        }

    async def test_the_reply_is_one_text_chunk_with_a_message_id(self) -> None:
        events = await _events(TestConfig("Hello world!"))

        assert wire(sole(events, TextMessageChunkEvent)) == {
            "type": "TEXT_MESSAGE_CHUNK",
            "messageId": IsStr(),
            "delta": "Hello world!",
        }

    async def test_a_conversation_with_history_is_answered(self) -> None:
        agent = Agent("test_agent", config=TestConfig("It's sunny today!"))
        incoming = run_input(
            UserMessage(id="m1", content="What's the weather like?"),
            AssistantMessage(id="m2", content="I'll check the weather for you."),
            UserMessage(id="m3", content="Thanks! And tomorrow?"),
        )

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert kinds_of(events) == [RunStartedEvent, TextMessageChunkEvent, RunFinishedEvent]
        assert sole(events, TextMessageChunkEvent).delta == "It's sunny today!"

    async def test_every_event_is_stamped(self) -> None:
        events = await _events(TestConfig("Hello"))

        assert [event.timestamp for event in events] == [IsInt(), IsInt(), IsInt()]


class TestTheProtocolVersion:
    async def test_it_declares_its_own_version_not_the_client_s(self) -> None:
        agent = Agent("test_agent", config=TestConfig("hello"))

        events = await dispatch_events(
            AGUIStream(agent), run_input(UserMessage(id="m1", content="hi"), protocol_version="1.9")
        )

        assert sole(events, RunStartedEvent).protocol_version == "1.0"

    @pytest.mark.parametrize("version", ["2.0", "0.9"])
    async def test_a_client_on_another_major_is_refused_before_any_run_starts(self, version: str) -> None:
        tracking = TrackingConfig(TestConfig("hello"))
        incoming = run_input(UserMessage(id="m1", content="hi"), protocol_version=version)

        events = await dispatch_events(AGUIStream(Agent("test_agent", config=tracking)), incoming)

        assert kinds_of(events) == [RunErrorEvent]
        assert sole(events, RunErrorEvent).code == UNSUPPORTED_PROTOCOL_VERSION
        assert tracking.mock.call_args_list == []

    @pytest.mark.parametrize("version", ["1.9", "one point oh", "1.0.1"])
    async def test_a_newer_minor_or_an_unreadable_version_is_served_with_a_warning(
        self, version: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        incoming = run_input(UserMessage(id="m1", content="hi"), protocol_version=version)

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            events = await dispatch_events(AGUIStream(Agent("test_agent", config=TestConfig("hello"))), incoming)

        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()
        [warning] = caplog.records
        assert version in warning.getMessage()

    @pytest.mark.parametrize("version", [None, "1.0"])
    async def test_a_client_on_this_version_or_predating_versions_is_served_quietly(
        self, version: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        incoming = run_input(UserMessage(id="m1", content="hi"), protocol_version=version)

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            events = await dispatch_events(AGUIStream(Agent("test_agent", config=TestConfig("hello"))), incoming)

        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()
        assert caplog.records == []


class TestClientToolCallsLeftPending:
    async def test_the_calls_left_for_the_client_are_named_in_call_order(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}'),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )
        incoming = run_input(UserMessage(id="m1", content="Paris and London?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        paris, london = each(events, ToolCallChunkEvent)
        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome(
            pending_tool_call_ids=[paris.tool_call_id, london.tool_call_id]
        )

    async def test_a_server_call_answered_in_the_run_is_not_pending(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig([
                ToolCallEvent(name="get_time"),
                ToolCallEvent(name="get_weather", arguments='{"location":"London"}'),
            ]),
        )

        @agent.tool
        def get_time() -> str:
            return "noon"

        incoming = run_input(UserMessage(id="m1", content="time and weather?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        [client_call] = each(events, ToolCallChunkEvent)
        assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome(
            pending_tool_call_ids=[client_call.tool_call_id]
        )

    async def test_a_run_with_no_client_calls_names_none(self) -> None:
        """An empty list would say nothing, so none is sent at all."""
        agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="get_time"), "it is noon"))

        @agent.tool
        def get_time() -> str:
            return "noon"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="time?")))

        assert wire(sole(events, RunFinishedEvent))["outcome"] == {"type": "success"}

    async def test_calls_left_for_the_client_send_no_text(self) -> None:
        """The executor ends such a call with a text-less `ModelResponse`; it must not open a message."""
        agent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="get_weather", arguments='{"location":"Paris"}')),
        )
        incoming = run_input(UserMessage(id="m1", content="Paris?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert kinds_of(events) == [RunStartedEvent, ToolCallChunkEvent, RunFinishedEvent]


class TestEmptyModelOutput:
    """The protocol forbids empty deltas, so an empty chunk or message leaves no text frame behind."""

    async def test_an_empty_first_chunk_does_not_open_a_text_message(self) -> None:
        events = await _events(TestConfig(ModelMessageChunk(""), ModelMessageChunk("Hello"), "Hello"))

        assert kinds_of(events) == [
            RunStartedEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            RunFinishedEvent,
        ]
        assert [event.delta for event in each(events, TextMessageContentEvent)] == ["Hello"]

    async def test_an_empty_chunk_between_real_ones_is_dropped(self) -> None:
        events = await _events(
            TestConfig(ModelMessageChunk("Hello"), ModelMessageChunk(""), ModelMessageChunk(" world"), "Hello world")
        )

        assert [event.delta for event in each(events, TextMessageContentEvent)] == ["Hello", " world"]

    async def test_an_empty_non_streaming_message_emits_no_text_frame(self) -> None:
        empty = ModelMessage("")

        events = await _events(TestConfig(empty, ModelResponse(empty)))

        assert kinds_of(events) == [RunStartedEvent, RunFinishedEvent]


@tool
def escalate(ticket_id: str) -> ToolResult:
    """Hand the ticket to a person; what it returns is the turn's reply."""
    return ToolResult(f"Ticket {ticket_id} was escalated.", final=True)


@tool
def forecast() -> str:
    """Read the forecast."""
    return "sunny"


class TestARepliedTurnNotPublishedAsAModelMessage:
    """A reply that reaches the turn only inside its `ModelResponse` still goes out as text.

    A provider publishes a `ModelMessage` (or chunks closed by one) for its text, but a final
    tool result, a middleware answering in the model's place, and a client that only streams
    chunks produce the `ModelResponse` alone.
    """

    async def test_a_final_tool_result_is_sent_as_the_reply(self) -> None:
        call = ToolCallEvent(name="escalate", arguments='{"ticket_id": "42"}')
        agent = Agent("test_agent", config=TestConfig(call), tools=[escalate])

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="help")))

        assert kinds_of(events) == [
            RunStartedEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            TextMessageChunkEvent,
            RunFinishedEvent,
        ]
        assert wire(sole(events, TextMessageChunkEvent)) == {
            "type": "TEXT_MESSAGE_CHUNK",
            "messageId": IsStr(),
            "delta": "Ticket 42 was escalated.",
        }

    async def test_text_the_model_sent_before_a_final_tool_result_goes_out_once(self) -> None:
        call = ToolCallEvent(name="escalate", arguments='{"ticket_id": "42"}')
        said = ModelMessage("Escalating.")
        agent = Agent(
            "test_agent",
            config=TestConfig(said, ModelResponse(said, tool_calls=ToolCallsEvent([call]))),
            tools=[escalate],
        )

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="help")))

        assert [event.delta for event in each(events, TextMessageChunkEvent)] == [
            "Escalating.",
            "Ticket 42 was escalated.",
        ]

    async def test_a_middleware_answering_in_the_model_s_place_is_sent_as_the_reply(self) -> None:
        middleware, _ = recording_history(reply="Answered without the model.")

        events = await dispatch_events(
            AGUIStream(Agent("test_agent", config=TestConfig("unused"))),
            run_input(UserMessage(id="m1", content="hi")),
            middleware=[middleware],
        )

        assert kinds_of(events) == [RunStartedEvent, TextMessageChunkEvent, RunFinishedEvent]
        assert wire(sole(events, TextMessageChunkEvent)) == {
            "type": "TEXT_MESSAGE_CHUNK",
            "messageId": IsStr(),
            "delta": "Answered without the model.",
        }

    async def test_chunks_no_message_closes_end_with_their_model_call(self) -> None:
        call = ToolCallEvent(name="forecast", arguments="{}")
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelMessageChunk("Checking."),
                ModelResponse(ModelMessage("Checking."), tool_calls=ToolCallsEvent([call])),
                ModelMessageChunk("Sunny."),
                ModelResponse(ModelMessage("Sunny.")),
            ),
            tools=[forecast],
        )

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="weather?")))

        assert kinds_of(events) == [
            RunStartedEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            RunFinishedEvent,
        ]
        first, second = each(events, TextMessageStartEvent)
        assert first.message_id != second.message_id
        assert [event.delta for event in each(events, TextMessageContentEvent)] == ["Checking.", "Sunny."]


def _provider_reply() -> tuple[Agent, list[Middleware]]:
    return Agent("test_agent", config=TestConfig("Hello.")), []


def _streamed_provider_reply() -> tuple[Agent, list[Middleware]]:
    config = TestConfig(ModelMessageChunk("Hello"), ModelMessageChunk(" there."), "Hello there.")
    return Agent("test_agent", config=config), []


def _final_tool_result() -> tuple[Agent, list[Middleware]]:
    call = ToolCallEvent(name="escalate", arguments='{"ticket_id": "42"}')
    return Agent("test_agent", config=TestConfig(call), tools=[escalate]), []


def _middleware_answer() -> tuple[Agent, list[Middleware]]:
    middleware, _ = recording_history(reply="Answered without the model.")
    return Agent("test_agent", config=TestConfig("unused")), [middleware]


def _chunks_without_a_closing_message() -> tuple[Agent, list[Middleware]]:
    config = TestConfig(ModelMessageChunk("Sun"), ModelMessageChunk("ny."), ModelResponse(ModelMessage("Sunny.")))
    return Agent("test_agent", config=config), []


def _text_messages(events: Sequence[Event]) -> list[str]:
    """Every text message the client can assemble, in order: one whole chunk, or a started message's content."""
    texts: list[str] = []
    for event in events:
        if isinstance(event, TextMessageChunkEvent):
            texts.append(event.delta or "")
        elif isinstance(event, TextMessageStartEvent):
            texts.append("")
        elif isinstance(event, TextMessageContentEvent):
            texts[-1] += event.delta
    return texts


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_provider_reply, id="provider-message"),
        pytest.param(_streamed_provider_reply, id="provider-chunks-closed-by-a-message"),
        pytest.param(_final_tool_result, id="final-tool-result"),
        pytest.param(_middleware_answer, id="middleware-answering-for-the-model"),
        pytest.param(_chunks_without_a_closing_message, id="chunks-without-a-closing-message"),
    ],
)
async def test_the_client_receives_the_reply_ask_returns(build: Callable[[], tuple[Agent, list[Middleware]]]) -> None:
    """However a turn's reply is produced, an AG-UI client sees it once, as its last text message.

    A new way to end a model call that `AGUIStream` does not map shows up here as a missing
    or duplicated reply, instead of passing every test that only goes through `agent.ask()`.
    """
    agent, middleware = build()
    reply = await agent.ask("hi", middleware=middleware)

    agent, middleware = build()
    events = await dispatch_events(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")), middleware=middleware
    )

    assert _text_messages(events) == [reply.body]


class TestAFailingRun:
    async def test_run_error_reports_the_failure(self) -> None:
        incoming = run_input(UserMessage(id="m1", content="go"))

        events = await events_of_failing_run(exploding_agent(), incoming)

        error = sole(events, RunErrorEvent)
        assert "downstream is down" in error.message
        assert error.timestamp is not None

    async def test_events_emitted_before_the_failure_are_observable(self) -> None:
        incoming = run_input(UserMessage(id="m1", content="go"))

        events = await events_of_failing_run(exploding_agent(), incoming)

        assert kinds_of(events)[0] is RunStartedEvent
        assert each(events, ToolCallResultEvent) != []

    async def test_the_run_is_identified_by_run_started_not_by_run_error(self) -> None:
        """`RUN_ERROR` carries no correlation ids, by protocol design.

        `RunErrorEvent` declares only `message`, `code` and `usage`, unlike `RunStartedEvent`
        and `RunFinishedEvent`. Setting ids anyway would serialise as keys the protocol does
        not define, so ag2 does not: a client identifies the run from `RUN_STARTED` on the same
        stream. Read from the raw frames, because extra keys exist only on the wire and parsing
        would hide exactly the failure this pins.
        """
        incoming = run_input(UserMessage(id="m1", content="go"))

        frames = await dispatch_run(AGUIStream(exploding_agent()), incoming)

        assert frames[0] == IsPartialDict({"threadId": incoming.thread_id, "runId": incoming.run_id})
        assert not {"threadId", "runId", "thread_id", "run_id"} & frames[-1].keys()

    async def test_the_original_exception_reaches_the_server_log(self, caplog: pytest.LogCaptureFixture) -> None:
        incoming = run_input(UserMessage(id="m1", content="go"))

        with caplog.at_level(logging.ERROR, logger="ag2.ag_ui"):
            events = await dispatch_events(AGUIStream(exploding_agent()), incoming)

        assert kinds_of(events)[-1] is RunErrorEvent
        [record] = [r for r in caplog.records if r.name.startswith("ag2.ag_ui")]
        assert record.exc_info is not None
        assert (type(record.exc_info[1]), str(record.exc_info[1])) == (RuntimeError, "downstream is down")

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([UserMessage(id="m1", content=[TextPart(text="look"), _MALFORMED])], id="user-part"),
            pytest.param(
                [
                    UserMessage(id="m1", content="go"),
                    ToolMessage(id="m2", tool_call_id="c1", content=[_MALFORMED]),
                    UserMessage(id="m3", content="and now?"),
                ],
                id="tool-part",
            ),
        ],
    )
    async def test_a_part_that_cannot_be_decoded_ends_the_run_with_run_error(self, messages: list[Any]) -> None:
        agent = Agent("test_agent", config=TestConfig("never reached"))

        events = await dispatch_events(AGUIStream(agent), run_input(*messages))

        assert kinds_of(events) == [RunStartedEvent, RunErrorEvent]


async def _drain_until(chunks: AsyncIterator[str], kind: type) -> None:
    """Read the response until an event of `kind` has been delivered."""
    async for chunk in chunks:
        if any(isinstance(event, kind) for event in decode_events([chunk])):
            return
    raise AssertionError(f"the run ended without emitting {kind.__name__}")


def _parked_run(park: Any, *turns: Any) -> tuple[AGUIStream, RunAgentInput]:
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="park", arguments="{}"), *turns), tools=[park])
    return AGUIStream(agent), run_input(UserMessage(id="m1", content="go"))


class TestATurnOutlivesItsExchange:
    """A served turn's lifetime belongs to the server, not to the HTTP exchange.

    An interrupt ends its exchange while the agent is still suspended mid-function, so the
    response is closed by hand here and what is asserted is that the agent reaches code past it.
    """

    async def test_turn_runs_on_past_the_end_of_its_exchange(self) -> None:
        gate, reached = asyncio.Event(), asyncio.Event()

        @tool
        async def park() -> str:
            """Wait for the test to let the turn through."""
            await gate.wait()
            reached.set()
            return "through"

        stream, incoming = _parked_run(park, "done")

        chunks = stream.dispatch(incoming)
        await _drain_until(chunks, ToolCallArgsEvent)
        await chunks.aclose()

        gate.set()
        await asyncio.wait_for(reached.wait(), timeout=_NEVER)

        await stream.aclose()

    async def test_a_turn_nobody_awaits_does_not_report_an_unretrieved_exception(self) -> None:
        gate, reached = asyncio.Event(), asyncio.Event()

        @tool
        async def park() -> str:
            """Fail once the test lets the turn through."""
            await gate.wait()
            reached.set()
            raise RuntimeError("nobody is listening")

        stream, incoming = _parked_run(park)

        reported: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        try:
            chunks = stream.dispatch(incoming)
            await _drain_until(chunks, ToolCallArgsEvent)
            await chunks.aclose()

            gate.set()
            await asyncio.wait_for(reached.wait(), timeout=_NEVER)

            await stream.aclose()
            # Let the turn unwind and be collected: an unretrieved task exception is
            # only reported when the task is finalised, which no assertion can force.
            for _ in range(3):
                await asyncio.sleep(0)
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous)

        assert reported == []

    async def test_a_turn_still_running_is_cancelled_on_shutdown(self) -> None:
        gate, parked, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        @tool
        async def park() -> str:
            """Park until cancelled, and say so on the way out."""
            parked.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "through"

        stream, incoming = _parked_run(park, "done")

        chunks = stream.dispatch(incoming)
        await _drain_until(chunks, ToolCallArgsEvent)
        await chunks.aclose()
        await asyncio.wait_for(parked.wait(), timeout=_NEVER)

        await stream.aclose()

        await asyncio.wait_for(cancelled.wait(), timeout=_NEVER)
