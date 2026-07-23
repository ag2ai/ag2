# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager

import pytest
from acp import schema

from ag2.acp import MCPCapabilityError
from ag2.acp.bridge import make_bridge
from ag2.acp.session import ACPSession, new_prompt_text
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.events import ModelRequest, TextInput
from ag2.events.types import ModelMessage


def _req(text: str) -> ModelRequest:
    return ModelRequest(parts=[TextInput(text)])


def test_delta_returns_only_new_requests() -> None:
    msgs = [_req("first"), ModelMessage("reply"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=0)
    assert "first" in text and "second" in text
    assert count == 2


def test_delta_skips_already_sent() -> None:
    msgs = [_req("first"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == "second"
    assert count == 2


def test_delta_empty_when_nothing_new() -> None:
    msgs = [_req("only")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == ""
    assert count == 1


def _msg(text: str) -> schema.AgentMessageChunk:
    return schema.AgentMessageChunk(
        session_update="agent_message_chunk",
        content=schema.TextContentBlock(type="text", text=text),
    )


@pytest.mark.asyncio
async def test_ensure_passes_mcp_servers_to_new_session() -> None:
    server = schema.HttpMcpServer(type="http", name="ext", url="http://127.0.0.1:1/mcp", headers=[])
    cfg = fake_acp_config(ACPTurn(updates=[_msg("hi")]), permission_policy="auto")

    session = ACPSession()
    session.bridge = make_bridge(cfg)
    await session.ensure(
        session.bridge,
        cfg.command,
        cwd=".",
        env=None,
        protocol_version=1,
        mcp_servers=[server],
        connect=cfg._connect,
    )
    try:
        assert session.conn.new_session_kwargs["mcp_servers"] == [server]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_ensure_rejects_agent_without_http_mcp() -> None:
    server = schema.HttpMcpServer(type="http", name="ext", url="http://127.0.0.1:1/mcp", headers=[])
    cfg = fake_acp_config(
        ACPTurn(updates=[_msg("hi")]),
        permission_policy="auto",
        agent_capabilities=schema.AgentCapabilities(),  # mcp http defaults to False
    )

    conns = []

    @asynccontextmanager
    async def connect(client):
        async with cfg._connect(client) as (conn, proc):
            conns.append(conn)
            yield conn, proc

    session = ACPSession()
    session.bridge = make_bridge(cfg)
    with pytest.raises(MCPCapabilityError):
        await session.ensure(
            session.bridge,
            cfg.command,
            cwd=".",
            env=None,
            protocol_version=1,
            mcp_servers=[server],
            connect=connect,
        )
    assert session.started is False
    (conn,) = conns
    assert conn.closed  # ensure tore the connection down, not just left it dangling


@pytest.mark.asyncio
async def test_ensure_skips_capability_check_without_servers() -> None:
    cfg = fake_acp_config(
        ACPTurn(updates=[_msg("hi")]),
        permission_policy="auto",
        agent_capabilities=schema.AgentCapabilities(),  # http False, but nothing to expose
    )

    session = ACPSession()
    session.bridge = make_bridge(cfg)
    await session.ensure(session.bridge, cfg.command, cwd=".", env=None, protocol_version=1, connect=cfg._connect)
    try:
        assert session.started is True
    finally:
        await session.close()
