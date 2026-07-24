# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from acp import schema

from ag2 import Agent
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.events import BaseEvent, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent


def _text(text: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=text)


def _model_option(current: str, *values: str) -> schema.SessionConfigOptionSelect:
    return schema.SessionConfigOptionSelect(
        id="model",
        name="Model",
        category="model",
        type="select",
        current_value=current,
        options=[schema.SessionConfigSelectOption(value=v, name=v) for v in values],
    )


def _hi_turn() -> ACPTurn:
    return ACPTurn(updates=[schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("hi"))])


@pytest.mark.asyncio
async def test_ask_streams_thoughts_tools_and_returns_text() -> None:
    cfg = fake_acp_config(
        ACPTurn(
            updates=[
                schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("planning")),
                schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("done")),
                schema.ToolCallStart(session_update="tool_call", tool_call_id="t1", title="Echo", status="pending"),
                schema.ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="t1",
                    status="completed",
                    content=[schema.ContentToolCallContent(type="content", content=_text("ok"))],
                ),
            ],
            usage=schema.Usage(input_tokens=3, output_tokens=1, total_tokens=4),
        ),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)

    seen: list[BaseEvent] = []

    try:
        async with agent.run("hello") as run:
            run.stream.subscribe(lambda e: seen.append(e))
            result = await run.result()
    finally:
        await cfg.aclose()

    assert result.body == "done"
    assert any(isinstance(e, ModelReasoning) and e.content == "planning" for e in seen)
    assert any(isinstance(e, BuiltinToolCallEvent) and e.name == "Echo" for e in seen)
    assert any(isinstance(e, BuiltinToolResultEvent) for e in seen)


@pytest.mark.asyncio
async def test_turn_timeout_surfaces_timeout() -> None:
    cfg = fake_acp_config(ACPTurn(hang=True), permission_policy="auto", turn_timeout=0.5)
    agent = Agent("acp", config=cfg)

    try:
        async with agent.run("hang") as run:
            result = await run.result()
    finally:
        await cfg.aclose()
    # The turn timed out; body is whatever streamed before the timeout (empty here).
    assert result.body == ""


@pytest.mark.asyncio
class TestModelSelection:
    async def test_selected_via_config_option(self) -> None:
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default", "provider/smart")],
            config_option_calls=calls,
            permission_policy="auto",
            model="provider/smart",
        )
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert calls == [("model", "provider/smart")]

    async def test_matching_current_is_not_resent(self) -> None:
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default", "provider/smart")],
            config_option_calls=calls,
            permission_policy="auto",
            model="provider/default",
        )
        agent = Agent("acp", config=cfg)

        try:
            await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert calls == []

    async def test_ignored_when_agent_has_no_model_option(self) -> None:
        # No configOptions advertised — `model` stays response metadata, as before.
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(_hi_turn(), config_option_calls=calls, permission_policy="auto", model="provider/smart")
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert calls == []
        assert reply.response.model == "provider/smart"

    async def test_not_offered_raises(self) -> None:
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default")],
            permission_policy="auto",
            model="provider/nonexistent",
        )
        agent = Agent("acp", config=cfg)

        try:
            with pytest.raises(ValueError, match="not offered by the ACP agent"):
                await agent.ask("hello")
        finally:
            await cfg.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_session() -> None:
    cfg = fake_acp_config(
        ACPTurn(updates=[schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("hi"))]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)

    async with agent.run("hello") as run:
        await run.result()

    assert cfg._sessions  # a live session was created
    conns = [s.conn for s in cfg._sessions.values()]
    await cfg.aclose()
    assert cfg._sessions == {}
    for conn in conns:
        assert conn is not None and conn.closed  # the connection context was exited
