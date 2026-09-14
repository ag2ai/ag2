# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An agent served over A2UI's AG-UI transport asks a question, and is answered.

The project has two independent AG-UI transports emitting their own run
lifecycle events, and a client cannot tell which one it is talking to. So this
is the same feature as ``test/ag_ui/test_interrupts.py``, driven the same way,
against the other one — the registry, the mapping and the proof underneath are
shared code, and what is tested here is that this transport reaches them.
"""

import asyncio
from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict, IsStr

pytest.importorskip("ag_ui")
pytest.importorskip("starlette")

from ag2 import Agent, Context  # noqa: E402
from ag2.a2ui import A2UIServer  # noqa: E402
from ag2.a2ui.transports import AgUiTransport  # noqa: E402
from ag2.ag_ui.interrupts import (  # noqa: E402
    AG2_METADATA_KEY,
    NOT_PROVEN,
    NO_HELD_TURN,
    PROOF_KEY,
    TOOL_APPROVAL_REASON,
    Retention,
)
from ag2.events import HumanInputRequest, ToolCallEvent  # noqa: E402
from ag2.exceptions import HumanInputError  # noqa: E402
from ag2.middleware import approval_required  # noqa: E402
from ag2.testing import TestConfig  # noqa: E402
from test.ag_ui.driving import (  # noqa: E402
    QUESTION,
    Asked,
    Clock,
    abandon,
    answer,
    ask_once,
    only,
    outcome_of,
    post_run,
    resolved,
    run_body,
    shut_down,
    sole_interrupt,
    types_of,
)

pytestmark = pytest.mark.asyncio

TTL = 60.0
_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def asking_server(*, timeout: float | None = None, **server_kwargs: Any) -> tuple[A2UIServer, Asked]:
    """An A2UI server whose agent puts a question to the human mid-turn."""
    asked = Asked()

    agent = Agent(
        "test_agent",
        config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), _A2UI_RESPONSE),
        **server_kwargs.pop("agent_kwargs", {}),
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Put the question to the human and report the answer."""
        try:
            asked.answers.append(await context.input(QUESTION, timeout=timeout))
        except asyncio.CancelledError:
            asked.ended("cancelled")
            raise
        except HumanInputError:
            asked.ended("no answer")
            raise
        return asked.answers[-1]

    server = A2UIServer(agent, **server_kwargs)
    return server, asked


class TestAQuestionIsAskedAndAnswered:
    async def test_the_question_ends_the_exchange_as_an_interrupt(self) -> None:
        app, _ = asking_server(transport=AgUiTransport())

        events = await post_run(app, run_body(thread_id="t1", run_id="r1"))

        assert types_of(events)[-1] == "RUN_FINISHED"
        assert sole_interrupt(events) == IsPartialDict({
            "reason": "human_input",
            "message": QUESTION,
            "expiresAt": IsStr(),
            "metadata": {AG2_METADATA_KEY: {PROOF_KEY: IsStr()}},
        })

    async def test_a_later_run_on_the_same_thread_answers_it(self) -> None:
        app, asked = asking_server(transport=AgUiTransport())

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))

        assert asked.answers == ["blue"]
        assert outcome_of(events) == {"type": "success"}
        assert only(events, "RUN_STARTED") == IsPartialDict({"threadId": "t1", "runId": "r2"})

    async def test_the_turn_finishes_its_own_work_in_the_resuming_exchange(self) -> None:
        """The A2UI surface this transport exists to deliver arrives after the pause."""
        app, _ = asking_server(transport=AgUiTransport())

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))

        assert only(events, "ACTIVITY_SNAPSHOT") == IsPartialDict({"activityType": "a2ui-surface"})

    async def test_an_agent_with_its_own_hook_answers_in_process(self) -> None:
        async def hook(event: HumanInputRequest) -> str:
            return "green"

        app, asked = asking_server(transport=AgUiTransport(), agent_kwargs={"hitl_hook": hook})

        events = await post_run(app, run_body(thread_id="t1", run_id="r1"))

        assert asked.answers == ["green"]
        assert outcome_of(events) == {"type": "success"}
        assert "RUN_ERROR" not in types_of(events)

    async def test_a_run_that_asks_nothing_states_a_success_outcome(self) -> None:
        agent = Agent("test_agent", config=TestConfig(_A2UI_RESPONSE))
        app = A2UIServer(agent, transport=AgUiTransport())

        events = await post_run(app, run_body(thread_id="t1", run_id="r1"))

        assert outcome_of(events) == {"type": "success"}


class TestParityWithTheOtherTransport:
    async def test_abandonment_ends_the_turn(self) -> None:
        app, asked = asking_server(transport=AgUiTransport())

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=abandon(interrupt)))

        assert outcome_of(events) == {"type": "success"}
        assert await asked.ending_within() == "cancelled"

    async def test_an_unknown_interrupt_is_refused(self) -> None:
        app, _ = asking_server(transport=AgUiTransport())

        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r1", text=None, resume=resolved("no-such-interrupt", "blue")),
        )

        assert types_of(events)[-1] == "RUN_ERROR"
        assert only(events, "RUN_ERROR") == IsPartialDict({"code": NO_HELD_TURN})

    async def test_a_stale_id_is_refused(self) -> None:
        app, asked = asking_server(transport=AgUiTransport())

        interrupt = await ask_once(app)
        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))
        events = await post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "red")))

        assert only(events, "RUN_ERROR") == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == ["blue"]

    async def test_an_unproven_resume_is_refused(self) -> None:
        app, asked = asking_server(transport=AgUiTransport())

        interrupt = await ask_once(app)
        events = await post_run(
            app,
            run_body(thread_id="t1", run_id="r2", text=None, resume=resolved(interrupt["id"], "blue")),
        )

        assert only(events, "RUN_ERROR") == IsPartialDict({"code": NOT_PROVEN})
        assert asked.answers == []

    async def test_it_declares_the_interrupt_capability(self) -> None:
        app, _ = asking_server(transport=AgUiTransport())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://a2ui.test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        assert response.json() == IsPartialDict({
            "humanInTheLoop": IsPartialDict({"supported": True, "interrupts": True}),
        })


class TestAToolCallAsksForApproval:
    """The other occasion for a pause, on the transport that is not the first one."""

    def _gated_server(self) -> tuple[A2UIServer, list[str]]:
        ran: list[str] = []
        agent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="delete_everything", arguments='{"path": "/"}'), _A2UI_RESPONSE),
        )

        @agent.tool(middleware=[approval_required()])
        async def delete_everything(context: Context, path: str) -> str:
            """Delete a path, once a human has said so."""
            ran.append(path)
            return f"deleted {path}"

        return A2UIServer(agent, transport=AgUiTransport()), ran

    async def test_approval_lets_the_tool_run(self) -> None:
        app, ran = self._gated_server()

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, True)))

        assert interrupt == IsPartialDict({"reason": TOOL_APPROVAL_REASON, "toolCallId": IsStr()})
        assert ran == ["/"]
        assert outcome_of(events) == {"type": "success"}

    async def test_refusal_stops_the_tool_without_failing_the_turn(self) -> None:
        app, ran = self._gated_server()

        interrupt = await ask_once(app)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, False)))

        assert ran == []
        assert "RUN_ERROR" not in types_of(events)
        assert outcome_of(events) == {"type": "success"}


class TestWhatItCostsTheServer:
    async def test_retention_is_configured_where_the_transport_is_built(self) -> None:
        clock = Clock()
        app, _ = asking_server(transport=AgUiTransport(retention=Retention(ttl=TTL), now=clock))

        interrupt = await ask_once(app)

        assert interrupt["expiresAt"] == clock.ahead(TTL)

    async def test_a_turn_past_its_deadline_is_gone(self) -> None:
        clock = Clock()
        app, asked = asking_server(transport=AgUiTransport(retention=Retention(ttl=TTL), now=clock))

        interrupt = await ask_once(app)
        clock.advance(TTL + 1)
        events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))

        assert only(events, "RUN_ERROR") == IsPartialDict({"code": NO_HELD_TURN})
        assert asked.answers == []

    async def test_shutting_the_server_down_cancels_a_held_turn(self) -> None:
        app, asked = asking_server(transport=AgUiTransport())

        await ask_once(app)
        await shut_down(app)

        assert await asked.ending_within() == "cancelled"
