# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest
from fast_depends.use import SerializerCls

from autogen.beta import MemoryStream, ToolResult
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    ToolCallEvent,
    ToolNotFoundEvent,
)
from autogen.beta.exceptions import ToolNotFoundError
from autogen.beta.tools.executor import ToolExecutor


def _not_found_events(events: list) -> list[ToolNotFoundEvent]:
    return [e for e in events if isinstance(e, ToolNotFoundEvent)]


@pytest.mark.asyncio
class TestToolNotFoundFallback:
    """Fallback `_tool_not_found` fires for unknown client-side tools only."""

    async def test_regular_unknown_tool_triggers_not_found(self) -> None:
        stream = MemoryStream()
        context = ConversationContext(stream=stream)

        with ExitStack() as stack:
            ToolExecutor(SerializerCls).register(stack, context, tools=[], known_tools={"known_func"})
            await context.send(ToolCallEvent(id="tc_1", name="unknown_func", arguments="{}"))

        expected_err = ToolNotFoundError("unknown_func")
        assert _not_found_events(list(await stream.history.get_events())) == [
            ToolNotFoundEvent(
                parent_id="tc_1",
                name="unknown_func",
                content=repr(expected_err),
                error=expected_err,
                result=ToolResult(),
            ),
        ]

    async def test_regular_known_tool_is_skipped(self) -> None:
        stream = MemoryStream()
        context = ConversationContext(stream=stream)

        with ExitStack() as stack:
            ToolExecutor(SerializerCls).register(stack, context, tools=[], known_tools={"known_func"})
            await context.send(ToolCallEvent(id="tc_1", name="known_func", arguments="{}"))

        assert _not_found_events(list(await stream.history.get_events())) == []
