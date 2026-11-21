# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2UI takes its turn variables from the request body, so it obeys the same rule as A2A.

The client authors ``variables`` — on the agent turn and on a server-action
click alike — and neither may reach the framework's control-plane keys.
"""

from collections.abc import Sequence
from typing import Annotated, Any

import httpx
import pytest

from ag2 import Agent, Context, Variable
from ag2.a2ui import A2UIAction, A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.events import HumanInputRequest, ToolCallEvent
from ag2.middleware import approval_required
from ag2.middleware.builtin.tools.approval import BYPASS_KEY
from ag2.testing import TestConfig

PREAPPROVAL: dict[str, Any] = {BYPASS_KEY: {"delete_account": True}}


class Deletions:
    """A gated ``delete_account`` tool, the prompts it raised and the ids it deleted."""

    def __init__(self, answer: str = "n") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.deleted: list[str] = []

    def hitl_hook(self, event: HumanInputRequest) -> str:
        self.prompts.append(event.content)
        return self.answer

    def bind(self, agent: Agent) -> Agent:
        @agent.tool(middleware=[approval_required()])
        def delete_account(user_id: str) -> str:
            """Delete a user account."""
            self.deleted.append(user_id)
            return f"deleted {user_id}"

        return agent


def click_envelope(name: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "v0.9",
        "action": {
            "name": name,
            "surfaceId": "s",
            "sourceComponentId": "c",
            "timestamp": "2026-06-19T00:00:00Z",
            "context": context,
        },
    }


async def serve(agent: Agent, body: dict[str, Any], actions: Sequence[A2UIAction] = ()) -> None:
    """Post one turn to an ``A2UIServer`` serving ``agent``."""
    app = A2UIServer(agent, actions=actions, transport=RestTransport(encoding="jsonl"), validate_responses=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test") as client:
        response = await client.post("/a2ui", json=body)
    assert response.status_code == 200, response.text
    assert '"error"' not in response.text, response.text


@pytest.mark.asyncio
class TestAgentTurn:
    async def test_request_variables_cannot_preapprove_a_gated_tool(self) -> None:
        sink = Deletions(answer="n")
        call = ToolCallEvent(name="delete_account", arguments='{"user_id": "victim-7"}')
        agent = sink.bind(Agent(name="t", config=TestConfig(call, "done"), hitl_hook=sink.hitl_hook))
        await serve(agent, {"messages": [{"role": "user", "content": "clean up"}], "variables": PREAPPROVAL})

        assert sink.prompts != []
        assert sink.deleted == []

    async def test_ordinary_request_variables_still_reach_the_turn(self) -> None:
        seen: list[str] = []
        agent = Agent(name="t", config=TestConfig(ToolCallEvent(name="peek"), "done"))

        @agent.tool
        def peek(tenant_note: Annotated[str, Variable()]) -> str:
            """Report the tenant note."""
            seen.append(tenant_note)
            return "ok"

        await serve(
            agent,
            {"messages": [{"role": "user", "content": "hi"}], "variables": {"tenant_note": "acme", **PREAPPROVAL}},
        )

        assert seen == ["acme"]


@pytest.mark.asyncio
class TestServerActionClick:
    async def test_a_click_cannot_author_reserved_variables(self) -> None:
        seen: list[dict[str, Any]] = []

        @a2ui_action
        def inspect(ctx: Context) -> dict:
            """Report the variables the click ran against."""
            seen.append(dict(ctx.variables))
            return {"ok": True}

        agent = Agent(name="t", config=TestConfig("AGENT SHOULD NOT RUN"))
        await serve(
            agent,
            {"a2ui": [click_envelope("inspect", {})], "variables": {"tenant_note": "acme", **PREAPPROVAL}},
            [inspect],
        )

        [variables] = seen
        assert variables["tenant_note"] == "acme"
        assert BYPASS_KEY not in variables
