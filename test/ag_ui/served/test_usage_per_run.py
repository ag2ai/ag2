# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each run's `usage` covers the model calls made in that run, and no others.

A turn paused on a question is carried by two runs: the one that asks reports
what was spent up to the pause, and the one that resumes reports only what it
spends itself. Both AG-UI servers draw that line in the same place.
"""

from typing import Any

import pytest

from ag2 import Agent, Context
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import AgUiTransport
from ag2.ag_ui import AGUIStream
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.testing import TestConfig
from test.ag_ui.harness import only, sole_interrupt
from test.ag_ui.serving import QUESTION, abandon, answer, app_for, post_run, run_body

pytestmark = pytest.mark.asyncio


def _spending_asker() -> Agent:
    """An agent that spends 100/10 asking, then 7/3 answering."""
    agent = Agent(
        "test_agent",
        config=TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent(name="ask_human", arguments="{}")]),
                usage=Usage(prompt_tokens=100, completion_tokens=10),
                model="m",
                provider="stub",
            ),
            ModelResponse(
                ModelMessage("done"), usage=Usage(prompt_tokens=7, completion_tokens=3), model="m", provider="stub"
            ),
        ),
    )

    @agent.tool
    async def ask_human(context: Context) -> str:
        """Ask the human."""
        return await context.input(QUESTION)

    return agent


def _usage(events: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    return only(events, "RUN_FINISHED").get("usage")


def _entry(input_tokens: int, output_tokens: int) -> list[dict[str, Any]]:
    return [
        {
            "provider": "stub",
            "model": "m",
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        }
    ]


def _ag_ui_app(agent: Agent) -> Any:
    return app_for(AGUIStream(agent))


def _a2ui_app(agent: Agent) -> Any:
    return A2UIServer(agent, transport=AgUiTransport(), validate_responses=False)


_APPS = pytest.mark.parametrize("make_app", [_ag_ui_app, _a2ui_app], ids=["AGUIStream", "A2UI"])


@_APPS
async def test_the_interrupted_and_the_resumed_run_each_report_their_own_calls(make_app: Any) -> None:
    app = make_app(_spending_asker())

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(sole_interrupt(first), "blue"))
    )

    assert _usage(first) == _entry(100, 10)
    assert _usage(second) == _entry(7, 3)


@_APPS
async def test_a_run_abandoning_the_question_reports_nothing_spent_twice(make_app: Any) -> None:
    app = make_app(_spending_asker())

    first = await post_run(app, run_body(thread_id="t1", run_id="r1"))
    second = await post_run(
        app, run_body(thread_id="t1", run_id="r2", text=None, resume=abandon(sole_interrupt(first)))
    )

    assert _usage(first) == _entry(100, 10)
    assert _usage(second) is None
