# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from acp import schema

from ag2 import Agent
from ag2.acp import ACPConfig, MCPCapabilityError
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.tools.builtin.mcp_server import MCPServerTool


def _says(text: str) -> ACPTurn:
    return ACPTurn(
        updates=[
            schema.AgentMessageChunk(
                session_update="agent_message_chunk",
                content=schema.TextContentBlock(type="text", text=text),
            )
        ]
    )


@pytest.mark.asyncio
async def test_each_turn_sends_the_agent_only_what_it_has_not_seen() -> None:
    prompts: list[str] = []
    cfg = fake_acp_config(_says("one"), _says("two"), permission_policy="auto", prompt_calls=prompts)

    async with cfg:
        reply = await Agent("acp", config=cfg).ask("first")
        await reply.ask("second")

    assert prompts == ["first", "second"]


@pytest.mark.asyncio
async def test_external_mcp_server_is_refused_when_the_agent_cannot_reach_http_mcp() -> None:
    cfg = fake_acp_config(
        _says("hi"),
        permission_policy="auto",
        agent_capabilities=schema.AgentCapabilities(),  # mcp http defaults to False
    )
    agent = Agent("acp", config=cfg, tools=[MCPServerTool(server_url="https://example.com/mcp", server_label="ext")])

    async with cfg:
        with pytest.raises(MCPCapabilityError):
            await agent.ask("hello")

        assert cfg.sessions == {}  # the refused session was torn down, not left dangling


@pytest.mark.asyncio
async def test_agent_without_http_mcp_is_fine_when_nothing_needs_it() -> None:
    cfg = fake_acp_config(
        _says("hi"),
        permission_policy="auto",
        agent_capabilities=schema.AgentCapabilities(),  # http False, but nothing to expose
    )

    async with cfg:
        reply = await Agent("acp", config=cfg).ask("hello")

    assert reply.body == "hi"


@pytest.mark.asyncio
async def test_agent_that_cannot_be_launched_surfaces_the_failure() -> None:
    cfg = ACPConfig(command=["ag2-no-such-acp-agent"], permission_policy="auto")

    async with cfg:
        with pytest.raises(FileNotFoundError):
            await Agent("acp", config=cfg).ask("hello")
