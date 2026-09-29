# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A run input's `context` entries are made available to the model, as part of its prompt."""

from collections.abc import Sequence

import pytest
from ag_ui.core import Context as ContextEntry
from ag_ui.core import UserMessage

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, run_input

pytestmark = pytest.mark.asyncio

_ENTRIES = [
    ContextEntry(description="The user's locale", value="en-GB"),
    ContextEntry(description="Open document", value='{"title": "Q3 plan"}'),
]


def _prompt_watcher() -> tuple[Agent, list[list[str]]]:
    """An agent whose one tool records the prompt it runs under."""
    seen: list[list[str]] = []
    agent = Agent("test_agent", prompt="Be helpful.", config=TestConfig(ToolCallEvent(name="peek"), "done"))

    @agent.tool
    def peek(context: Context) -> str:
        """Look at the prompt."""
        seen.append(list(context.prompt))
        return "seen"

    return agent, seen


async def test_each_context_entry_reaches_the_prompt() -> None:
    agent, seen = _prompt_watcher()
    incoming = run_input(UserMessage(id="m1", content="hi"))
    incoming.context = _ENTRIES

    await dispatch_run(AGUIStream(agent), incoming)

    [prompt] = seen
    assert prompt == [
        "Be helpful.",
        '## Context from the application\n\n- The user\'s locale: en-GB\n- Open document: {"title": "Q3 plan"}',
    ]


async def test_without_context_the_prompt_is_unchanged() -> None:
    agent, seen = _prompt_watcher()

    await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))

    assert seen == [["Be helpful."]]


async def test_the_application_can_turn_the_context_block_off() -> None:
    agent, seen = _prompt_watcher()
    incoming = run_input(UserMessage(id="m1", content="hi"))
    incoming.context = _ENTRIES

    await dispatch_run(AGUIStream(agent), incoming, context_prompt=None)

    assert seen == [["Be helpful."]]


def _as_pairs(entries: Sequence[ContextEntry]) -> str:
    return "\n".join(f"{e.description}={e.value}" for e in entries)


async def test_the_application_can_render_the_context_its_own_way() -> None:
    agent, seen = _prompt_watcher()
    incoming = run_input(UserMessage(id="m1", content="hi"))
    incoming.context = _ENTRIES

    await dispatch_run(AGUIStream(agent), incoming, context_prompt=_as_pairs)

    assert seen == [["Be helpful.", 'The user\'s locale=en-GB\nOpen document={"title": "Q3 plan"}']]
