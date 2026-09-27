# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from acp import schema

from ag2 import MemoryStream
from ag2.acp.permissions import resolve_permission_option_id
from ag2.context import ConversationContext
from ag2.events import HumanInputRequest
from ag2.events.types import HumanMessage

ALLOW = schema.PermissionOption(option_id="ok", kind="allow_once", name="Allow")
REJECT = schema.PermissionOption(option_id="no", kind="reject_once", name="Reject")


def _tool_call(title: str | None = None) -> schema.ToolCallUpdate:
    return schema.ToolCallUpdate(tool_call_id="tc1", title=title)


class _Human:
    """A real context whose every question is answered with ``reply``."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []
        self.context = ConversationContext(stream=MemoryStream())
        self.context.stream.where(HumanInputRequest).subscribe(self._answer, sync_to_thread=False)

    async def _answer(self, event: HumanInputRequest) -> None:
        self.prompts.append(event.content)
        await self.context.send(HumanMessage(self._reply, parent_id=event.id))


@pytest.mark.asyncio
async def test_auto_allows() -> None:
    assert await resolve_permission_option_id("auto", [ALLOW, REJECT], _tool_call(), None) == "ok"


@pytest.mark.asyncio
async def test_deny_rejects() -> None:
    assert await resolve_permission_option_id("deny", [ALLOW, REJECT], _tool_call(), None) == "no"


@pytest.mark.asyncio
async def test_ask_human_allows() -> None:
    human = _Human("yes")
    chosen = await resolve_permission_option_id("ask", [ALLOW, REJECT], _tool_call("Edit a.py"), human.context)
    assert chosen == "ok"
    assert human.prompts  # human was actually asked


@pytest.mark.asyncio
async def test_ask_human_rejects() -> None:
    human = _Human("no")
    chosen = await resolve_permission_option_id("ask", [ALLOW, REJECT], _tool_call("Edit a.py"), human.context)
    assert chosen == "no"


@pytest.mark.asyncio
async def test_ask_without_context_rejects() -> None:
    # No way to ask a human -> safe default is reject.
    assert await resolve_permission_option_id("ask", [ALLOW, REJECT], _tool_call(), None) == "no"


@pytest.mark.asyncio
async def test_returns_none_when_no_matching_option() -> None:
    # auto but only reject options available -> None (caller denies)
    assert await resolve_permission_option_id("auto", [REJECT], _tool_call(), None) is None
