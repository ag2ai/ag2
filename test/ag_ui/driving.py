# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Driving a served agent the way a client drives it: two POSTs over in-process HTTP.

A paused turn is two exchanges sharing state, which the single-exchange
generator seam cannot express — so these helpers post real requests at the
built ASGI application instead, and read only what a client can see on the wire.

Importing this module needs ``starlette`` and ``httpx``; guard the importing
test module with ``pytest.importorskip("starlette")``.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.exceptions import HumanInputError
from ag2.testing import TestConfig

QUESTION = "What is your favourite colour?"


class Clock:
    """A hand-advanced UTC clock, so a deadline can pass without waiting for it.

    Injected where the transport is built. Nothing in these tests sleeps to
    reach an expiry: a wall-clock test of a fifteen-minute default would either
    take fifteen minutes or force the bound down to something no operator would
    configure.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def ahead(self, seconds: float) -> str:
        """``seconds`` from now, spelled the way the wire spells a deadline."""
        return (self._now + timedelta(seconds=seconds)).isoformat()


class Asked:
    """What the agent's own code did with the questions it was asked.

    The assertion that matters throughout: a turn that completes proves only
    that *something* answered it, and a turn that vanishes proves nothing at
    all about where its coroutine went.
    """

    def __init__(self) -> None:
        self.answers: list[str] = []
        self.ending: str | None = None
        self._ended = asyncio.Event()

    def ended(self, how: str) -> None:
        self.ending = how
        self._ended.set()

    async def ending_within(self, timeout: float = 1.0) -> str | None:
        """Wait for the ask to end without an answer, and say how it ended.

        Awaited rather than slept through: cancellation reaches the suspended
        coroutine when the loop next runs it, which is soon but not now.
        """
        await asyncio.wait_for(self._ended.wait(), timeout)
        return self.ending


def asking_agent(
    *,
    questions: tuple[str, ...] = (QUESTION,),
    timeout: float | None = None,
    **agent_kwargs: Any,
) -> tuple[Agent, Asked]:
    """An agent whose one tool puts ``questions`` to the human, and what came back."""
    asked = Asked()

    agent = Agent(
        "test_agent",
        config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "all done"),
        **agent_kwargs,
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Put the questions to the human and report the answers."""
        try:
            for question in questions:
                asked.answers.append(await context.input(question, timeout=timeout))
        except asyncio.CancelledError:
            asked.ended("cancelled")
            raise
        except HumanInputError:
            asked.ended("no answer")
            raise
        return " / ".join(asked.answers)

    return agent, asked


def app_for(stream: AGUIStream) -> Starlette:
    return Starlette(routes=[Route("/", stream.build_asgi())])


async def shut_down(app: Any) -> None:
    """Take an ASGI app down the way a server does, over the lifespan protocol.

    Both AG-UI transports hang their "cancel what is still held" hook off
    shutdown, and that wiring is only real if a server running the app would in
    fact reach it.
    """
    messages = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]

    async def receive() -> dict[str, str]:
        return messages.pop(0)

    async def send(message: dict[str, str]) -> None:
        assert not message["type"].endswith(".failed"), message

    await app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)


def run_body(
    *,
    thread_id: str,
    run_id: str,
    text: str | None = "go",
    resume: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One ``RunAgentInput``, spelled the way a browser client spells it."""
    body: dict[str, Any] = {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": text}] if text is not None else [],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    if resume is not None:
        body["resume"] = resume
    return body


async def post_run(app: Any, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive one exchange over in-process HTTP and decode its AG-UI events."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ag-ui.test") as client:
        response = await client.post("/", json=body)
        assert response.status_code == 200
        return [
            json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
        ]


async def ask_once(app: Any, thread_id: str = "t1", run_id: str = "r1") -> dict[str, Any]:
    """Drive a run up to the question it asks, and return the interrupt it left."""
    return sole_interrupt(await post_run(app, run_body(thread_id=thread_id, run_id=run_id)))


def only(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    [event] = [e for e in events if e["type"] == event_type]
    return event


def types_of(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def outcome_of(events: list[dict[str, Any]]) -> dict[str, Any]:
    outcome = only(events, "RUN_FINISHED")["outcome"]
    assert isinstance(outcome, dict)
    return outcome


def sole_interrupt(events: list[dict[str, Any]]) -> dict[str, Any]:
    [interrupt] = outcome_of(events)["interrupts"]
    return interrupt


def answer(interrupt: dict[str, Any], payload: Any) -> list[dict[str, Any]]:
    """A compliant client's resume: the answer, under the envelope it was given.

    A client carries an interrupt's ``metadata`` back untouched, so anything the
    server put there to recognise its own question comes home with the answer.
    """
    return resolved(interrupt["id"], payload, metadata=interrupt.get("metadata"))


def abandon(interrupt: dict[str, Any]) -> list[dict[str, Any]]:
    """A compliant client giving up on the question."""
    return [{"interruptId": interrupt["id"], "status": "cancelled", "metadata": interrupt.get("metadata")}]


def resolved(interrupt_id: str, payload: Any, *, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One resume entry, spelled out — for tests about ids and envelopes themselves."""
    return [{"interruptId": interrupt_id, "status": "resolved", "payload": payload, "metadata": metadata}]
