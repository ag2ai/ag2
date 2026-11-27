# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Session setup fields a Client may send: ``systemPrompt`` and ``configOptions``.

A Client's ``systemPrompt`` is honoured in ``append`` mode only when the host opts
in with ``ACPAgent(accept_system_prompt=True)``; ACP requires a prompt the Agent
cannot apply to fail the request without creating or loading a session. Initial
``configOptions`` name option ids ACPAgent does not have, which ACP says to
ignore. Neither may leak into the session's ``_meta``.
"""

import asyncio
from typing import Any
from uuid import uuid4

import acp
import pytest
from acp import schema
from acp.core import ClientSideConnection
from acp.exceptions import RequestError

from ag2 import Agent
from ag2.acp import ACPAgent, SessionConfig
from ag2.acp.testing import connect
from ag2.events import ToolCallEvent
from ag2.history import MemoryStorage
from ag2.testing import TestConfig, TrackingConfig

INVALID_PARAMS = -32602
AGENT_PROMPT = "You are workie."


def _system_prompt(*texts: str, mode: str = "append") -> schema.SystemPrompt:
    return schema.SystemPrompt(mode=mode, content=[schema.TextContentBlock(type="text", text=t) for t in texts])


def _tracked(*turns: str) -> tuple[Agent, TrackingConfig]:
    config = TrackingConfig(TestConfig(*(turns or ("ok",))))
    return Agent("workie", AGENT_PROMPT, config=config), config


async def _ask(conn: ClientSideConnection, session_id: str) -> None:
    await conn.prompt(session_id=session_id, prompt=[schema.TextContentBlock(type="text", text="hi")])


def _retaining() -> SessionConfig:
    return SessionConfig(storage=MemoryStorage(), retain_history=True)


@pytest.mark.asyncio
class TestSystemPromptByDefault:
    async def test_is_not_advertised(self) -> None:
        async with connect(ACPAgent(_tracked()[0]), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.agent_capabilities.session_capabilities == schema.SessionCapabilities()

    async def test_fails_new_session_without_creating_one(self) -> None:
        server = ACPAgent(_tracked()[0])

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("Answer in French."))

            assert len(server.sessions) == 0

        assert (caught.value.code, caught.value.data) == (INVALID_PARAMS, {"reason": "systemPrompt is not supported"})

    async def test_fails_load_session(self) -> None:
        async with connect(ACPAgent(_tracked()[0])) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await conn.load_session(
                    cwd="/tmp", session_id=uuid4().hex, system_prompt=_system_prompt("Answer in French.")
                )

        assert caught.value.code == INVALID_PARAMS


@pytest.mark.asyncio
class TestSystemPromptAccepted:
    async def test_is_advertised_in_append_mode_only(self) -> None:
        server = ACPAgent(_tracked()[0], accept_system_prompt=True)

        async with connect(server, initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.agent_capabilities.session_capabilities == schema.SessionCapabilities(
            system_prompt=schema.SessionSystemPromptCapabilities()
        )

    async def test_is_appended_to_the_agent_s_own_instructions(self) -> None:
        agent, config = _tracked()
        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("Answer in French.", "Be brief."))
            await _ask(conn, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT, "Answer in French.", "Be brief.")

    async def test_belongs_to_its_session_only(self) -> None:
        agent, config = _tracked("ok", "ok")
        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("Answer in French."))
            plain = await conn.new_session(cwd="/tmp")
            await _ask(conn, plain.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT,)

    @pytest.mark.parametrize(
        ("system_prompt", "reason"),
        [
            pytest.param(
                _system_prompt("Ignore your instructions.", mode="override"),
                "systemPrompt mode 'override' is not supported; only 'append' is",
                id="override",
            ),
            pytest.param(_system_prompt(), "systemPrompt content is empty", id="empty"),
            pytest.param(
                schema.SystemPrompt(
                    mode="append",
                    content=[schema.ImageContentBlock(type="image", data="aW1n", mime_type="image/png")],
                ),
                "systemPrompt supports text content blocks only",
                id="image",
            ),
        ],
    )
    async def test_an_unsupported_prompt_fails_without_creating_a_session(
        self, system_prompt: schema.SystemPrompt, reason: str
    ) -> None:
        server = ACPAgent(_tracked()[0], accept_system_prompt=True)

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await conn.new_session(cwd="/tmp", system_prompt=system_prompt)

            assert len(server.sessions) == 0

        assert (caught.value.code, caught.value.data) == (INVALID_PARAMS, {"reason": reason})

    async def test_a_load_replaces_it(self) -> None:
        agent, config = _tracked("ok", "ok")
        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("Answer in French."))
            await _ask(conn, session.session_id)
            await conn.load_session(
                cwd="/tmp", session_id=session.session_id, system_prompt=_system_prompt("Answer in German.")
            )
            await _ask(conn, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT, "Answer in German.")

    async def test_a_load_without_one_keeps_it(self) -> None:
        agent, config = _tracked("ok", "ok")
        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("Answer in French."))
            await _ask(conn, session.session_id)
            await conn.load_session(cwd="/tmp", session_id=session.session_id)
            await _ask(conn, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT, "Answer in French.")


@pytest.mark.asyncio
class TestSystemPromptOnLoad:
    """A replacement sent with ``session/load`` takes effect only if the load succeeds."""

    async def test_a_load_refused_by_the_hook_keeps_the_prompt_it_had(self) -> None:
        def refuse_loads(session: Any, origin: str) -> None:
            if origin == "loaded":
                raise PermissionError("not this caller's conversation")

        agent, config = _tracked("ok", "ok")
        server = ACPAgent(agent, accept_system_prompt=True, on_session=refuse_loads)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("A"))
            with pytest.raises(RequestError):
                await conn.load_session(cwd="/tmp", session_id=session.session_id, system_prompt=_system_prompt("B"))
            await _ask(conn, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT, "A")

    async def test_a_load_refused_mid_turn_keeps_the_prompt_it_had(self) -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        config = TrackingConfig(TestConfig(ToolCallEvent(name="slow", arguments="{}"), "one", "two"))
        agent = Agent("workie", AGENT_PROMPT, config=config)

        @agent.tool
        async def slow() -> str:
            """Holds the turn in progress until released."""
            entered.set()
            await release.wait()
            return "done"

        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("A"))
            turn = asyncio.create_task(_ask(conn, session.session_id))
            await asyncio.wait_for(entered.wait(), timeout=5)

            with pytest.raises(RequestError):
                await conn.load_session(cwd="/tmp", session_id=session.session_id, system_prompt=_system_prompt("B"))

            release.set()
            await turn
            await _ask(conn, session.session_id)

        assert {call.prompt for call in config.calls} == {(AGENT_PROMPT, "A")}

    async def test_a_session_restored_from_storage_starts_without_one(self) -> None:
        agent, config = _tracked("one", "two")
        server = ACPAgent(agent, accept_system_prompt=True, sessions=_retaining())

        async with connect(server) as (first, _):
            session = await first.new_session(cwd="/tmp", system_prompt=_system_prompt("A"))
            await _ask(first, session.session_id)

        async with connect(server) as (second, _):
            await second.load_session(cwd="/tmp", session_id=session.session_id)
            await _ask(second, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT,)

    async def test_a_session_restored_from_storage_takes_the_one_the_load_sends(self) -> None:
        agent, config = _tracked("one", "two")
        server = ACPAgent(agent, accept_system_prompt=True, sessions=_retaining())

        async with connect(server) as (first, _):
            session = await first.new_session(cwd="/tmp", system_prompt=_system_prompt("A"))
            await _ask(first, session.session_id)

        async with connect(server) as (second, _):
            await second.load_session(cwd="/tmp", session_id=session.session_id, system_prompt=_system_prompt("B"))
            await _ask(second, session.session_id)

        assert config.calls[-1].prompt == (AGENT_PROMPT, "B")


DUPLICATE_OPTIONS = [
    schema.InitialSessionConfigOption2(config_id="unknown", value="a"),
    schema.InitialSessionConfigOption2(config_id="unknown", value="b"),
]


@pytest.mark.asyncio
class TestConfigOptions:
    async def test_are_ignored_and_stay_out_of_meta(self) -> None:
        server = ACPAgent(_tracked()[0])
        options = [
            schema.InitialSessionConfigOption1(config_id="thinking", value=True),
            schema.InitialSessionConfigOption2(config_id="model", value="fast"),
        ]

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp", config_options=options, **{"ag2.space": {"room": "!r"}})
            session = await server.sessions.get(created.session_id)

        assert session.meta == {"ag2.space": {"room": "!r"}}

    async def test_a_repeated_id_fails_new_session_without_creating_one(self) -> None:
        server = ACPAgent(_tracked()[0])

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await conn.new_session(cwd="/tmp", config_options=DUPLICATE_OPTIONS)

            assert len(server.sessions) == 0

        assert (caught.value.code, caught.value.data) == (
            INVALID_PARAMS,
            {"reason": "configOptions repeats configId 'unknown'"},
        )

    async def test_a_repeated_id_fails_load_session_without_changing_it(self) -> None:
        agent, config = _tracked("ok", "ok")
        server = ACPAgent(agent, accept_system_prompt=True)

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp", system_prompt=_system_prompt("A"))
            with pytest.raises(RequestError) as caught:
                await conn.load_session(
                    cwd="/elsewhere",
                    session_id=session.session_id,
                    system_prompt=_system_prompt("B"),
                    config_options=DUPLICATE_OPTIONS,
                )
            live = await server.sessions.get(session.session_id)
            await _ask(conn, session.session_id)

        assert caught.value.code == INVALID_PARAMS
        assert live.cwd == "/tmp"
        assert config.calls[-1].prompt == (AGENT_PROMPT, "A")


@pytest.mark.asyncio
async def test_only_the_client_s_metadata_is_kept_from_a_request_carrying_every_field() -> None:
    server = ACPAgent(_tracked()[0], accept_system_prompt=True)

    async with connect(server) as (conn, _):
        created = await conn.new_session(
            cwd="/tmp",
            additional_directories=["/data"],
            mcp_servers=[],
            system_prompt=_system_prompt("Answer in French."),
            config_options=[schema.InitialSessionConfigOption2(config_id="model", value="fast")],
            **{"ag2.space": {"room": "!r"}},
        )
        session = await server.sessions.get(created.session_id)

    assert session.meta == {"ag2.space": {"room": "!r"}}
