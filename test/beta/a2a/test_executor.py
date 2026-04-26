# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from autogen.beta import Agent, Context
from autogen.beta.a2a.executor import AgentExecutor
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


def _user_message(
    text: str = "hi",
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    return Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


def _request_context(message: Message, *, task: Task | None = None) -> RequestContext:
    return RequestContext(request=MessageSendParams(message=message), task=task)


async def _drain(queue: EventQueue, timeout: float = 0.2) -> list:
    events: list = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.dequeue_event(), timeout)
            events.append(ev)
            queue.task_done()
    except asyncio.TimeoutError:
        return events


def _is_state(state: TaskState) -> Callable[[object], bool]:
    def matcher(ev: object) -> bool:
        return isinstance(ev, TaskStatusUpdateEvent) and ev.status.state == state

    return matcher


def _hitl_agent() -> Agent:
    """Agent that calls a tool which then asks the human to confirm."""
    agent = Agent(
        "specialist",
        "p",
        config=TestConfig(
            ToolCallEvent(name="confirm", arguments="{}"),
            "after-confirm",
        ),
    )

    @agent.tool
    async def confirm(context: Context) -> str:
        return await context.input("Confirm?")

    return agent


@pytest.mark.asyncio
class TestSimpleAsk:
    async def test_emits_text_artifact_with_agent_reply(self) -> None:
        agent = Agent("specialist", "be helpful", config=TestConfig("answer-text"))
        executor = AgentExecutor(agent)
        queue = EventQueue()

        await executor.execute(_request_context(_user_message("hi")), queue)
        events = await _drain(queue)

        artifacts = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        final = artifacts[-1]
        assert final.last_chunk is True
        assert final.artifact.name == "result"
        assert final.artifact.parts == [Part(root=TextPart(text="answer-text"))]

    async def test_terminates_with_completed_status(self) -> None:
        agent = Agent("specialist", "be helpful", config=TestConfig("done"))
        executor = AgentExecutor(agent)
        queue = EventQueue()

        await executor.execute(_request_context(_user_message()), queue)
        events = await _drain(queue)

        statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert statuses[-1].status.state == TaskState.completed
        assert statuses[-1].final is True


@pytest.mark.asyncio
class TestHITLForwarding:
    async def test_initial_call_emits_input_required_status(self) -> None:
        executor = AgentExecutor(_hitl_agent())
        queue = EventQueue()

        await executor.execute(_request_context(_user_message("please confirm")), queue)
        events = await _drain(queue)

        assert any(_is_state(TaskState.input_required)(e) for e in events)
        assert not any(_is_state(TaskState.completed)(e) for e in events)

    async def test_followup_with_replay_history_completes_task(self) -> None:
        executor = AgentExecutor(_hitl_agent())
        original = _user_message("please confirm")

        await executor.execute(_request_context(original), EventQueue())

        # Simulate the second HTTP request: a2a-sdk would have appended the
        # follow-up to task.history. We construct that state explicitly.
        followup = _user_message("yes", context_id=original.context_id, task_id=original.task_id)
        existing_task = Task(
            id=original.task_id or uuid4().hex,
            context_id=original.context_id or uuid4().hex,
            status=TaskStatus(state=TaskState.input_required),
            history=[original, followup],
        )
        queue = EventQueue()
        await executor.execute(_request_context(followup, task=existing_task), queue)
        events = await _drain(queue)

        artifacts = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        assert artifacts[-1].artifact.parts == [Part(root=TextPart(text="after-confirm"))]
        assert any(_is_state(TaskState.completed)(e) for e in events)


@pytest.mark.asyncio
async def test_cancel_emits_canceled_status() -> None:
    agent = Agent("specialist", "p", config=TestConfig("ok"))
    executor = AgentExecutor(agent)

    task = Task(
        id="t-1",
        context_id="c-1",
        status=TaskStatus(state=TaskState.working),
        history=[_user_message()],
    )
    queue = EventQueue()
    await executor.cancel(_request_context(_user_message(), task=task), queue)
    events = await _drain(queue)

    assert any(_is_state(TaskState.canceled)(e) for e in events)


@pytest.mark.asyncio
async def test_cancel_without_task_is_noop() -> None:
    agent = Agent("specialist", "p", config=TestConfig("ok"))
    executor = AgentExecutor(agent)

    queue = EventQueue()
    await executor.cancel(_request_context(_user_message()), queue)
    events = await _drain(queue)

    assert events == []
