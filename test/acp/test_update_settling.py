# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A turn reads out every update that preceded its response, in the order sent.

Two guarantees, from two places. Completeness is the SDK's: ``session/prompt``
does not return until the session's in-flight ``session/update`` handlers have
returned, so a turn read after the response is not short of its tail. Order is
AG2's: the SDK spawns a task per notification, so handlers that await — and this
one does, it sends to the run's stream — would otherwise interleave and scramble
the chunks (see :mod:`ag2.acp.dispatch`).

This needs a real ACP connection: ``fake_acp_config`` calls the client's
``session_update`` directly, so it never spawns the tasks that cause this.
``duplex_acp_config`` gives the genuine connection without a subprocess.
"""

from typing import Any

import acp
import pytest
from acp import schema

from ag2 import Agent
from ag2.acp.testing import duplex_acp_config
from ag2.events import ModelMessageChunk

# Enough that unserialized handlers cannot plausibly stay in order by luck, and
# small enough to stay fast.
CHUNKS = 200

# Each chunk is one digit of its own index, so the assembled text pins order as
# well as completeness: a short read is a truncated tail, a scrambled one is not.
EXPECTED = "".join(str(i % 10) for i in range(CHUNKS))


class ChunkingAgent:
    """An ACP agent that streams ``CHUNKS`` message chunks, then ends the turn.

    The shape is the point: every ``session/update`` is written, then
    ``session/prompt`` returns, with nothing in between that would let the client
    catch up. That is what a real CLI agent does on a fast turn.
    """

    def __init__(self, conn: acp.Client) -> None:
        self.conn = conn

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=schema.Implementation(name="chunker", version="test"),
        )

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id="chunk-session-1")

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        for i in range(CHUNKS):
            await self.conn.session_update(
                session_id=session_id,
                update=acp.update_agent_message_text(str(i % 10)),
            )
        return schema.PromptResponse(stop_reason="end_turn")


@pytest.mark.asyncio
async def test_every_streamed_chunk_reaches_the_reply_in_order() -> None:
    """Unguarded, this loses the tail (no settle) or scrambles it (no ordering)."""
    cfg = duplex_acp_config(ChunkingAgent)

    try:
        reply = await Agent("acp", config=cfg).ask("stream")
    finally:
        await cfg.aclose()

    # Compared whole rather than by length: a truncated tail and a reordered
    # middle both fail, and the digits say which one happened.
    assert reply.body == EXPECTED


@pytest.mark.asyncio
async def test_the_stream_and_the_reply_carry_the_same_chunks() -> None:
    """Nothing is dropped or reordered between the stream and the assembled body.

    Order is asserted on the stream too, not just on the body: updates are handled
    one at a time in the order they were read off the wire, so a subscriber sees
    the agent's chunks in the order the agent sent them. Drop that serialization
    and this is the test that fails — the stream shows the interleaving directly.
    """
    chunks: list[str] = []
    cfg = duplex_acp_config(ChunkingAgent)

    try:
        async with Agent("acp", config=cfg).run("stream") as run:
            run.stream.subscribe(lambda e: chunks.append(e.content) if isinstance(e, ModelMessageChunk) else None)
            reply = await run.result()
    finally:
        await cfg.aclose()

    assert len(chunks) == CHUNKS
    assert "".join(chunks) == reply.body == EXPECTED
