# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Elicitation over a real ACP connection, not a direct call into the bridge.

``test_elicitation.py`` drives the in-process double, which calls the bridge
directly. That leaves two things it cannot prove: that the capability really
reaches the agent through ``initialize``, and that ``elicitation/create`` really
dispatches to the bridge rather than being answered with method-not-found by the
SDK's router. Both are exercised here against an agent reached over genuine SDK
connections — real capability negotiation, real JSON-RPC framing, real routers.
"""

from typing import Any

import acp
import pytest
from acp import schema

from ag2 import Agent
from ag2.acp import ACPConfig
from ag2.acp.testing import duplex_acp_config
from ag2.events import HumanInputRequest

AUTH_URL = "https://example.com/authorize"


class ElicitingAgent:
    """An ACP agent that asks the user to complete a url flow on every prompt.

    The turn's only message chunk is the outcome, which is what the tests assert
    on: ``"not advertised"`` when AG2 did not offer the elicitation capability,
    otherwise the ``action`` of the response AG2 sent back.
    """

    def __init__(self, conn: acp.Client) -> None:
        # The reverse handle: an ACP Agent talks back to the Client through it.
        self.conn = conn
        self.elicitation_offered = False

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        capabilities = kwargs.get("client_capabilities")
        self.elicitation_offered = bool(capabilities is not None and capabilities.elicitation is not None)
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=schema.Implementation(name="elicitor", version="test"),
        )

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id="elicit-session-1")

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        await self.conn.session_update(
            session_id=session_id,
            update=acp.update_agent_message_text(await self._outcome(session_id)),
        )
        return schema.PromptResponse(stop_reason="end_turn")

    async def _outcome(self, session_id: str) -> str:
        if not self.elicitation_offered:
            return "not advertised"
        response = await self.conn.create_elicitation(
            message="Authorize the test",
            mode=schema.ElicitationUrlSessionMode(
                session_id=session_id,
                elicitation_id="elicit-1",
                url=AUTH_URL,
            ),
        )
        return response.action


def _config(**overrides: str) -> ACPConfig:
    return duplex_acp_config(ElicitingAgent, **overrides)


@pytest.mark.asyncio
async def test_the_agent_can_ask_and_gets_the_answer() -> None:
    prompts: list[str] = []

    def human(event: HumanInputRequest) -> str:
        prompts.append(event.content)
        return "yes"

    cfg = _config()
    agent = Agent("acp", config=cfg, hitl_hook=human)

    try:
        reply = await agent.ask("do the thing")
    finally:
        await cfg.aclose()

    # The agent reports the action AG2 sent back, so this is the response as the
    # agent itself saw it after a full round trip over the wire.
    assert reply.body == "accept"
    [prompt] = prompts
    assert AUTH_URL in prompt


@pytest.mark.asyncio
async def test_declining_the_policy_stops_the_agent_asking_at_all() -> None:
    prompts: list[str] = []

    def human(event: HumanInputRequest) -> str:
        prompts.append(event.content)
        return "yes"

    cfg = _config(elicitation_policy="decline")
    agent = Agent("acp", config=cfg, hitl_hook=human)

    try:
        reply = await agent.ask("do the thing")
    finally:
        await cfg.aclose()

    # The agent saw no elicitation capability in `initialize`, so it never asked.
    assert reply.body == "not advertised"
    assert prompts == []
