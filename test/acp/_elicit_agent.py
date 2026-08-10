# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A standalone ACP agent process that asks the user something, then reports the answer.

Run as ``python -m test.acp._elicit_agent``. AG2's ``ACPConfig`` launches this the
same way it launches Claude Code or Codex, so the elicitation path is exercised
over a real subprocess: real capability negotiation, real ``elicitation/create``
dispatch through the SDK's client router, real JSON-RPC framing.

The turn's only message chunk is the outcome, which is what the test asserts on:
``"not advertised"`` when AG2 did not offer the elicitation capability, otherwise
the ``action`` of the response AG2 sent back.

Nothing may be written to stdout except the protocol itself — stdout *is* the
transport.
"""

import asyncio
from typing import Any

import acp
from acp import schema

AUTH_URL = "https://example.com/authorize"


class ElicitingAgent:
    """Minimal ACP Agent: on every prompt, asks the user to complete a url flow."""

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


if __name__ == "__main__":
    asyncio.run(acp.run_agent(ElicitingAgent))  # type: ignore[arg-type]
