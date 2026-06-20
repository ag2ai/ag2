# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public mid-run message injection API (Agent.run / RunHandle).

These exercise the in-process layer over the existing stream inbox: a producer
pushes a follow-up message into a *live* run and the agent picks it up at the
next step boundary (between tool rounds), never mid-LLM-call.

The tool announces it has started (``started`` event) so the test injects while
the tool is genuinely in flight — otherwise an injection sent before the run
begins would be drained into the very first request (drain at turn start) and
never reach the LLM as a standalone follow-up.

Delivery is asserted via ``TrackingConfig``: it records the last message handed
to the LLM on each call, so we can check the injected text actually reached the
model rather than inspecting internal stream state.
"""

import asyncio
import threading

import pytest

from autogen.beta import Agent, Context, tool
from autogen.beta.events import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
)
from autogen.beta.run import RunHandle, RunRegistry
from autogen.beta.testing import TestConfig, TrackingConfig


def _reached_llm(tracking: TrackingConfig, text: str) -> bool:
    """Whether any request the framework sent to the LLM carried ``text``."""
    sent = [call.args[0] for call in tracking.mock.call_args_list]
    return any(isinstance(m, ModelRequest) and TextInput(text) in m.parts for m in sent)


def _injectable_agent(started: asyncio.Event) -> tuple[Agent, TrackingConfig]:
    """Agent whose tool signals start then yields until an externally-injected
    message lands in the inbox, then a final reply. Returns the agent and the
    TrackingConfig used to assert what reached the LLM."""
    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="wait_for_injection", arguments="{}"),
            ModelResponse(ModelMessage("Done.")),
        )
    )

    @tool
    async def wait_for_injection(ctx: Context) -> str:
        """Yield until the externally-injected message reaches the inbox."""
        started.set()
        for _ in range(300):
            if ctx.pending_messages:
                return "ok"
            await asyncio.sleep(0.01)
        raise AssertionError("injected message never arrived")

    agent = Agent("agent", config=tracking, tools=[wait_for_injection])
    return agent, tracking


@pytest.mark.asyncio
class TestRunHandle:
    async def test_send_mid_run_reaches_llm(self) -> None:
        """A message pushed via handle.send() during a tool round is delivered
        to the LLM on the following call."""
        injected = "URGENT: also summarize in French"
        started = asyncio.Event()
        agent, tracking = _injectable_agent(started)

        handle = agent.run("Start the task")
        await asyncio.wait_for(started.wait(), timeout=3.0)
        handle.send(injected)  # pushed while the tool is in flight

        reply = await asyncio.wait_for(handle.result(), timeout=3.0)
        assert reply.body == "Done."
        assert _reached_llm(tracking, injected)

    async def test_task_id_is_stream_id(self) -> None:
        agent = Agent("agent", config=TestConfig(ModelResponse(ModelMessage("hi"))))
        handle = agent.run("hello")
        assert isinstance(handle, RunHandle)
        assert handle.task_id == handle.stream.id
        await asyncio.wait_for(handle.result(), timeout=3.0)
        assert handle.done()

    async def test_registry_routes_and_autoremoves(self) -> None:
        """RunRegistry.send routes by task_id; the handle is auto-removed when
        the run finishes, after which send() returns False."""
        injected = "extra instruction via registry"
        started = asyncio.Event()
        agent, tracking = _injectable_agent(started)

        registry = RunRegistry()
        handle = agent.run("Start", registry=registry)
        assert handle.task_id in registry.active()

        await asyncio.wait_for(started.wait(), timeout=3.0)
        assert registry.send(handle.task_id, injected) is True

        reply = await asyncio.wait_for(handle.result(), timeout=3.0)
        assert reply.body == "Done."
        assert _reached_llm(tracking, injected)

        # Auto-removed after completion (done-callback runs on the loop).
        await asyncio.sleep(0)
        assert handle.task_id not in registry.active()
        assert registry.send(handle.task_id, "too late") is False

    async def test_send_from_other_thread(self) -> None:
        """handle.send() from a non-loop thread is marshalled onto the run's
        loop via call_soon_threadsafe and still reaches the LLM."""
        injected = "pushed from a worker thread"
        started = asyncio.Event()
        agent, tracking = _injectable_agent(started)

        handle = agent.run("Start")
        await asyncio.wait_for(started.wait(), timeout=3.0)
        threading.Thread(target=handle.send, args=(injected,)).start()

        reply = await asyncio.wait_for(handle.result(), timeout=3.0)
        assert reply.body == "Done."
        assert _reached_llm(tracking, injected)
