# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Events a run sends besides its text and tool calls: the model's reasoning, and delegations to subagents.

A provider streams reasoning as ordinary events on the agent's stream, so it is scripted that way
here and the framing the client receives is what is asserted: one session, opened once, closed
before the answer begins, and never opened for a thought with nothing in it. Reasoning arriving
*inbound*, in an AG-UI history, belongs to the input tests.
"""

import logging
from base64 import b64encode
from uuid import uuid4

import pytest
from ag_ui.core import (
    Event,
    ReasoningEncryptedValueEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    StepFinishedEvent,
    StepStartedEvent,
    SubagentErrorEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    TextMessageChunkEvent,
    ToolCallChunkEvent,
    ToolCallStartEvent,
    UserMessage,
)
from dirty_equals import IsStr

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.events import ModelReasoning, TaskCompleted, TaskStarted, ToolCallEvent
from ag2.testing import TestConfig, Turn
from test.ag_ui.harness import dispatch_events, each, kinds_of, run_input, sole, weather_tool, wire

pytestmark = pytest.mark.asyncio

_REASONING = (
    ReasoningStartEvent,
    ReasoningMessageStartEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningEndEvent,
)


async def _events(*script: Turn) -> list[Event]:
    agent = Agent("test_agent", config=TestConfig(*script))
    return await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))


class TestReasoning:
    async def test_chunks_emit_one_session_under_one_message_id(self) -> None:
        events = await _events(ModelReasoning("Thinking"), ModelReasoning(" more"), "Done")

        message_id = sole(events, ReasoningStartEvent).message_id
        assert wire(sole(events, ReasoningMessageStartEvent)) == {
            "type": "REASONING_MESSAGE_START",
            "messageId": message_id,
            "role": "reasoning",
        }
        assert [wire(event) for event in each(events, ReasoningMessageContentEvent)] == [
            {"type": "REASONING_MESSAGE_CONTENT", "messageId": message_id, "delta": "Thinking"},
            {"type": "REASONING_MESSAGE_CONTENT", "messageId": message_id, "delta": " more"},
        ]
        assert sole(events, ReasoningMessageEndEvent).message_id == message_id
        assert sole(events, ReasoningEndEvent).message_id == message_id

    async def test_the_session_closes_before_the_answer_begins(self) -> None:
        """A client renders reasoning and answer in separate places; they must not interleave."""
        events = await _events(ModelReasoning("thinking"), "Final answer")

        kinds = kinds_of(events)
        assert kinds.index(ReasoningEndEvent) < kinds.index(TextMessageChunkEvent)

    async def test_an_empty_chunk_is_skipped(self) -> None:
        events = await _events(ModelReasoning(""), ModelReasoning("real thought"), "Done")

        assert [event.delta for event in each(events, ReasoningMessageContentEvent)] == ["real thought"]

    async def test_a_run_that_did_no_reasoning_opens_no_session(self) -> None:
        events = await _events("Hello")

        assert [kind for kind in kinds_of(events) if kind in _REASONING] == []

    async def test_a_gemini_tool_signature_is_sent_for_client_replay(self) -> None:
        signature = b"gemini-signature"
        agent = Agent(
            "test_agent",
            config=TestConfig(
                GeminiToolCallEvent(id="call-1", name="get_weather", arguments="{}", thought_signature=signature)
            ),
        )
        incoming = run_input(UserMessage(id="m1", content="weather?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        assert wire(sole(events, ReasoningEncryptedValueEvent)) == {
            "type": "REASONING_ENCRYPTED_VALUE",
            "subtype": "tool-call",
            "entityId": "call-1",
            "encryptedValue": b64encode(signature).decode(),
        }

    async def test_a_client_tool_s_signature_follows_the_call_it_belongs_to(self) -> None:
        """A consumer may drop a value whose entity it has not seen."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                GeminiToolCallEvent(id="call-1", name="get_weather", arguments="{}", thought_signature=b"sig")
            ),
        )
        incoming = run_input(UserMessage(id="m1", content="weather?"), tools=[weather_tool()])

        events = await dispatch_events(AGUIStream(agent), incoming)

        kinds = kinds_of(events)
        assert kinds.index(ReasoningEncryptedValueEvent) > kinds.index(ToolCallChunkEvent)

    async def test_a_server_tool_s_signature_follows_the_call_s_start(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                GeminiToolCallEvent(id="call-1", name="lookup", arguments="{}", thought_signature=b"sig"), "done"
            ),
        )

        @agent.tool
        def lookup() -> str:
            """Look something up."""
            return "found"

        events = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="go")))

        kinds = kinds_of(events)
        assert kinds.index(ReasoningEncryptedValueEvent) > kinds.index(ToolCallStartEvent)


def _delegating(worker: Agent, *objectives: str) -> Agent:
    """A parent whose model delegates each of `objectives` to `worker`, all in one response."""
    calls = [ToolCallEvent(name="task_worker", arguments=f'{{"objective": "{o}"}}') for o in objectives]
    return Agent(
        "parent",
        config=TestConfig(calls, "summarised"),
        tools=[worker.as_tool(description="Delegate to the worker.")],
    )


async def _run(agent: Agent) -> list[Event]:
    return await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="go")))


def _completed(task_id: str, objective: str, result: str) -> TaskCompleted:
    return TaskCompleted(task_id=task_id, agent_name="worker", objective=objective, result=result, task_stream=uuid4())


def _succeeded(events: list[Event]) -> bool:
    return sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()


class TestSubagents:
    """Each delegation reaches the client as one subagent invocation, started and then ended."""

    async def test_a_delegation_starts_with_its_agent_and_objective_and_finishes_with_its_result(self) -> None:
        events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "look into it"))

        started = sole(events, SubagentStartedEvent)
        assert wire(started) == {
            "type": "SUBAGENT_STARTED",
            "subagentRunId": IsStr(),
            "name": "worker",
            "description": "look into it",
            "parentToolCallId": IsStr(),
        }
        assert wire(sole(events, SubagentFinishedEvent)) == {
            "type": "SUBAGENT_FINISHED",
            "subagentRunId": started.subagent_run_id,
            "result": "researched",
        }

    async def test_two_parallel_delegations_to_one_agent_are_told_apart(self) -> None:
        events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "first", "second"))

        started = {event.subagent_run_id for event in each(events, SubagentStartedEvent)}
        finished = {event.subagent_run_id for event in each(events, SubagentFinishedEvent)}
        assert len(started) == 2
        assert finished == started

    async def test_parallel_delegations_name_their_spawning_tool_calls(self) -> None:
        events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "first", "second"))

        calls = {event.tool_call_id for event in each(events, ToolCallStartEvent)}
        parents = {event.parent_tool_call_id for event in each(events, SubagentStartedEvent)}
        assert len(calls) == 2
        assert parents == calls

    async def test_a_failed_delegation_is_a_subagent_error_and_the_run_carries_on(self) -> None:
        worker = Agent("worker", config=TestConfig(RuntimeError("the worker fell over")))

        events = await _run(_delegating(worker, "look into it"))

        started = sole(events, SubagentStartedEvent)
        assert wire(sole(events, SubagentErrorEvent)) == {
            "type": "SUBAGENT_ERROR",
            "subagentRunId": started.subagent_run_id,
            "message": "the worker fell over",
        }
        assert each(events, SubagentFinishedEvent) == []
        assert _succeeded(events)

    async def test_delegations_are_no_longer_steps(self) -> None:
        events = await _run(_delegating(Agent("worker", config=TestConfig("researched")), "look into it"))

        assert each(events, SubagentStartedEvent) != []
        assert [kind for kind in kinds_of(events) if kind in (StepStartedEvent, StepFinishedEvent)] == []

    async def test_a_task_id_already_announced_in_the_run_is_not_announced_again(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An id a tool picks itself can repeat; the wire must still never see one invocation twice."""
        parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

        @parent.tool
        async def delegate_twice(context: Context) -> str:
            """Report two delegations under one id."""
            for objective in ("first", "second"):
                await context.send(TaskStarted(task_id="same", agent_name="worker", objective=objective))
                await context.send(_completed("same", objective, "researched"))
            return "delegated"

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            events = await _run(parent)

        assert [event.description for event in each(events, SubagentStartedEvent)] == ["first"]
        assert len(each(events, SubagentFinishedEvent)) == 1
        assert "same" in caplog.records[0].getMessage()
        assert _succeeded(events)

    async def test_a_task_id_still_open_stays_open_until_its_last_delegation_ends(self) -> None:
        """Ends under one id cannot be told apart, so the first to arrive is not the one sent."""
        parent = Agent("parent", config=TestConfig(ToolCallEvent(name="delegate_twice"), "summarised"))

        @parent.tool
        async def delegate_twice(context: Context) -> str:
            """Report a second delegation under one id that ends while the first still runs."""
            await context.send(TaskStarted(task_id="same", agent_name="slow", objective="first"))
            await context.send(TaskStarted(task_id="same", agent_name="fast", objective="second"))
            await context.send(_completed("same", "second", "fast result"))
            await context.send(_completed("same", "first", "slow result"))
            return "delegated"

        events = await _run(parent)

        started = sole(events, SubagentStartedEvent)
        assert (started.subagent_run_id, started.description) == ("same", "first")
        assert [(e.subagent_run_id, e.result) for e in each(events, SubagentFinishedEvent)] == [("same", "slow result")]
        assert _succeeded(events)


class TestAnInvocationThatEndsWithoutFinishing:
    """Stopped, expired or never ended: each invocation still closes before its run does."""

    @staticmethod
    def _owning(end: str) -> Agent:
        parent = Agent("parent", config=TestConfig(ToolCallEvent(name="research"), "done"))

        @parent.tool
        async def research(context: Context) -> str:
            """Research, and stop the task the given way."""
            task = parent.task("research", context=context)
            await task.__aenter__()
            if end == "cancel":
                await task.cancel("no longer needed")
            elif end == "expire":
                await task.expire()
            return "stopped"

        return parent

    @pytest.mark.parametrize(("end", "message"), [("cancel", "cancelled: no longer needed"), ("expire", "expired")])
    async def test_a_stopped_task_is_a_subagent_error(self, end: str, message: str) -> None:
        events = await _run(self._owning(end))

        started = sole(events, SubagentStartedEvent)
        error = sole(events, SubagentErrorEvent)
        assert (error.subagent_run_id, error.message) == (started.subagent_run_id, message)
        assert _succeeded(events)

    async def test_an_invocation_still_open_when_the_run_ends_is_closed_first(self) -> None:
        events = await _run(self._owning("never"))

        started = sole(events, SubagentStartedEvent)
        assert kinds_of(events)[-2:] == [SubagentErrorEvent, RunFinishedEvent]
        assert sole(events, SubagentErrorEvent).subagent_run_id == started.subagent_run_id
        assert _succeeded(events)
