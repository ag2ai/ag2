# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A held interrupt is never silently abandoned, on either AG-UI server.

`basic/patterns/interrupt-resume.mdx`: an interrupt the producer can tell is
still open and has no entry is *uncovered*. Omission is not abandonment, so the
question stays held and the run is refused before it starts. An entry naming an
interrupt this server did not raise is dropped with a warning.
"""

import logging
from collections.abc import Callable
from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import AgUiTransport
from ag2.ag_ui import NOT_COVERED, AGUIStream
from test.ag_ui.harness import outcome_of, types_of
from test.ag_ui.serving import answer, app_for, ask_once, asking_agent, post_run, resolved, run_body

pytestmark = pytest.mark.asyncio


def _ag_ui_app(agent: Agent) -> Any:
    return app_for(AGUIStream(agent))


def _a2ui_app(agent: Agent) -> Any:
    return A2UIServer(agent, transport=AgUiTransport(), validate_responses=False)


_APPS = pytest.mark.parametrize("make_app", [_ag_ui_app, _a2ui_app], ids=["AGUIStream", "A2UI"])


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("ag2.ag_ui") and r.levelno == logging.WARNING]


async def _still_resumable(app: Any, interrupt: dict[str, Any]) -> dict[str, Any]:
    events = await post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "blue")))
    return outcome_of(events)


@_APPS
async def test_a_fresh_run_on_a_thread_holding_a_question_is_refused_before_it_starts(
    make_app: Callable[[Agent], Any],
) -> None:
    agent, asked = asking_agent()
    app = make_app(agent)
    interrupt = await ask_once(app)

    events = await post_run(app, run_body(thread_id="t1", run_id="r2"))

    assert events == [IsPartialDict({"type": "RUN_ERROR", "code": NOT_COVERED})]
    assert await _still_resumable(app, interrupt) == {"type": "success"}
    assert asked.answers == ["blue"]


@_APPS
async def test_an_entry_of_an_unknown_status_is_stripped_and_leaves_the_question_uncovered(
    make_app: Callable[[Agent], Any], caplog: pytest.LogCaptureFixture
) -> None:
    agent, asked = asking_agent()
    app = make_app(agent)
    interrupt = await ask_once(app)
    [entry] = answer(interrupt, "blue")

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await post_run(
            app, run_body(thread_id="t1", run_id="r2", text=None, resume=[{**entry, "status": "deferred"}])
        )

    assert events == [IsPartialDict({"type": "RUN_ERROR", "code": NOT_COVERED})]
    [warning] = _warnings(caplog)
    assert "/resume/0" in warning
    assert await _still_resumable(app, interrupt) == {"type": "success"}
    assert asked.answers == ["blue"]


@_APPS
async def test_an_entry_for_an_interrupt_not_raised_beside_a_valid_one_is_dropped_with_a_warning(
    make_app: Callable[[Agent], Any], caplog: pytest.LogCaptureFixture
) -> None:
    agent, asked = asking_agent()
    app = make_app(agent)
    interrupt = await ask_once(app)

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await post_run(
            app,
            run_body(
                thread_id="t1",
                run_id="r2",
                text=None,
                resume=[*answer(interrupt, "blue"), *resolved("no-such-interrupt", "red")],
            ),
        )

    assert outcome_of(events) == {"type": "success"}
    assert asked.answers == ["blue"]
    [warning] = _warnings(caplog)
    assert "no-such-interrupt" in warning


@_APPS
async def test_an_entry_for_an_interrupt_not_raised_alone_leaves_the_question_uncovered(
    make_app: Callable[[Agent], Any], caplog: pytest.LogCaptureFixture
) -> None:
    agent, asked = asking_agent()
    app = make_app(agent)
    interrupt = await ask_once(app)

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await post_run(
            app, run_body(thread_id="t1", run_id="r2", text=None, resume=resolved("no-such-interrupt", "red"))
        )

    assert events == [IsPartialDict({"type": "RUN_ERROR", "code": NOT_COVERED})]
    [warning] = _warnings(caplog)
    assert "no-such-interrupt" in warning
    assert await _still_resumable(app, interrupt) == {"type": "success"}


@_APPS
async def test_a_resume_on_a_thread_holding_nothing_is_set_aside_with_a_warning(
    make_app: Callable[[Agent], Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The protocol has a producer treat such entries as unrecognised and serve the run."""
    agent, asked = asking_agent()
    app = make_app(agent)

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await post_run(
            app, run_body(thread_id="t1", run_id="r1", text=None, resume=resolved("no-such-interrupt", "blue"))
        )

    assert events[0] == IsPartialDict({"type": "RUN_STARTED"})
    assert "RUN_ERROR" not in types_of(events)
    [warning] = _warnings(caplog)
    assert "no-such-interrupt" in warning
    assert asked.answers == []
