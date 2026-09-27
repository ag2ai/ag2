# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG-UI ``state`` seeds the turn's variables, so it obeys the same rule as A2A.

The browser is a peer like any other: it does not author the framework's
control-plane keys, and it is not shown them in a state snapshot.
"""

from typing import Annotated, Any

import pytest
from ag_ui.core import UserMessage

from ag2 import Agent, Variable
from ag2.ag_ui import AGUIStream
from ag2.events import HumanInputRequest, ToolCallEvent
from ag2.middleware import approval_required
from ag2.middleware.builtin.tools.approval import BYPASS_KEY
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, every, run_input

pytestmark = pytest.mark.asyncio

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


async def test_client_state_cannot_preapprove_a_gated_tool() -> None:
    sink = Deletions(answer="n")
    call = ToolCallEvent(name="delete_account", arguments='{"user_id": "victim-7"}')
    agent = sink.bind(Agent("test_agent", config=TestConfig(call, "Done"), hitl_hook=sink.hitl_hook))
    incoming = run_input(UserMessage(id="msg_1", content="clean up"), state=PREAPPROVAL)

    await dispatch_run(AGUIStream(agent), incoming)

    assert sink.prompts != []
    assert sink.deleted == []


async def test_ordinary_client_state_still_reaches_the_turn() -> None:
    seen: list[str] = []
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="peek"), "Done"))

    @agent.tool
    def peek(tenant_note: Annotated[str, Variable()]) -> str:
        """Report the tenant note."""
        seen.append(tenant_note)
        return "ok"

    incoming = run_input(
        UserMessage(id="msg_1", content="hi"),
        state={"tenant_note": "acme", **PREAPPROVAL},
    )

    events = await dispatch_run(AGUIStream(agent), incoming)

    assert seen == ["acme"]
    assert all(BYPASS_KEY not in snap["snapshot"] for snap in every(events, "STATE_SNAPSHOT"))


async def test_state_snapshot_hides_reserved_variables() -> None:
    agent = Agent("test_agent", config=TestConfig("Done"), variables={"tenant_note": "acme", **PREAPPROVAL})
    incoming = run_input(UserMessage(id="msg_1", content="hi"))

    events = await dispatch_run(AGUIStream(agent), incoming)

    snapshots = every(events, "STATE_SNAPSHOT")
    assert snapshots != []
    assert all(snap["snapshot"] == {"tenant_note": "acme"} for snap in snapshots)
