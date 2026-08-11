# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACP ``_meta`` must never displace a validated request field.

The SDK router merges a request's ``_meta`` *over* its canonical parameters, so a
colliding key silently replaces the real value and leaves no trace. ``_meta`` is
exactly where an application is encouraged to put data from elsewhere — chat-room
provenance, say — so that data must not be able to name the session a prompt runs
against.
"""

from typing import Any

import acp
import pytest
from acp.exceptions import RequestError

from ag2 import Agent
from ag2.acp import ACPAgent
from ag2.acp.guard import colliding_meta_keys
from ag2.acp.testing import connect
from ag2.testing import TestConfig


def _agent(*turns: str) -> Agent:
    return Agent("workie", config=TestConfig(*(turns or ("ok",))))


def _raw(conn: Any) -> Any:
    return conn._conn if hasattr(conn, "_conn") else conn._connection


class TestCollisionDetection:
    def test_a_reserved_field_is_spotted(self) -> None:
        params = {"sessionId": "a", "prompt": [], "_meta": {"session_id": "b"}}

        assert colliding_meta_keys("session/prompt", params) == frozenset({"session_id"})

    def test_a_json_alias_is_spotted_too(self) -> None:
        params = {"sessionId": "a", "prompt": [], "_meta": {"sessionId": "b"}}

        assert colliding_meta_keys("session/prompt", params) == frozenset({"sessionId"})

    def test_namespaced_application_metadata_is_fine(self) -> None:
        params = {"cwd": "/tmp", "mcpServers": [], "_meta": {"ag2.space": {"room": "!r"}}}

        assert colliding_meta_keys("session/new", params) == frozenset()

    def test_a_request_without_meta_is_fine(self) -> None:
        assert colliding_meta_keys("session/new", {"cwd": "/tmp", "mcpServers": []}) == frozenset()


@pytest.mark.asyncio
class TestOverrideIsRefused:
    async def test_meta_cannot_redirect_a_prompt_to_another_session(self) -> None:
        server = ACPAgent(_agent("ok", "ok"))

        async with connect(server) as (conn, _):
            victim = (await conn.new_session(cwd="/tmp")).session_id
            mine = (await conn.new_session(cwd="/tmp")).session_id

            with pytest.raises(RequestError):
                await _raw(conn).send_request(
                    "session/prompt",
                    {
                        "sessionId": mine,
                        "prompt": [{"type": "text", "text": "hi"}],
                        "_meta": {"session_id": victim},
                    },
                )

            target = await server.sessions.get(victim)
            assert list(await server.sessions.stream(target).history.get_events()) == []

    async def test_meta_cannot_replace_the_prompt_payload(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")

            with pytest.raises(RequestError):
                await _raw(conn).send_request(
                    "session/prompt",
                    {
                        "sessionId": session.session_id,
                        "prompt": [{"type": "text", "text": "harmless"}],
                        "_meta": {"prompt": [{"type": "text", "text": "substituted"}]},
                    },
                )

    async def test_meta_cannot_retarget_a_new_session_s_cwd(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await _raw(conn).send_request(
                    "session/new",
                    {"cwd": "/tmp", "mcpServers": [], "_meta": {"cwd": "/etc"}},
                )

    async def test_the_error_names_the_offending_key(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError) as caught:
                await _raw(conn).send_request(
                    "session/new",
                    {"cwd": "/tmp", "mcpServers": [], "_meta": {"cwd": "/etc"}},
                )

        assert "cwd" in caught.value.data["reason"]


@pytest.mark.asyncio
class TestLegitimateMetadataStillWorks:
    async def test_namespaced_metadata_reaches_the_session(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            response = await _raw(conn).send_request(
                "session/new",
                {"cwd": "/tmp", "mcpServers": [], "_meta": {"ag2.space": {"room": "!r"}}},
            )
            session = await server.sessions.get(response["sessionId"])

        assert session.meta == {"ag2.space": {"room": "!r"}}

    async def test_an_ordinary_prompt_is_unaffected(self) -> None:
        async with connect(ACPAgent(_agent("200"))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block("what's 100 + 100")],
            )

        assert response.stop_reason == "end_turn"
        assert recorder.updates_for(session.session_id)
