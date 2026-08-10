# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import acp
import pytest
from acp.exceptions import RequestError

from ag2 import Agent
from ag2.acp import ACPAgent, PromptContent, StaticTokenAuth
from ag2.acp.testing import connect
from ag2.testing import TestConfig


def _agent(*turns: str) -> Agent:
    return Agent("workie", config=TestConfig(*(turns or ("ok",))))


@pytest.mark.asyncio
class TestHandshake:
    async def test_negotiates_the_protocol_version(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.protocol_version == acp.PROTOCOL_VERSION

    async def test_never_negotiates_above_what_it_implements(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION + 5)

        assert response.protocol_version == acp.PROTOCOL_VERSION

    async def test_advertises_the_agent_name_and_version(self) -> None:
        server = ACPAgent(_agent(), name="custom", version="9.9.9", title="Custom Agent")

        async with connect(server, initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert (response.agent_info.name, response.agent_info.version) == ("custom", "9.9.9")
        assert response.agent_info.title == "Custom Agent"

    async def test_name_defaults_to_the_agent_name(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.agent_info.name == "workie"


@pytest.mark.asyncio
class TestCapabilitiesAreTruthful:
    """Anything advertised here must actually work — a Client relies on it."""

    async def test_does_not_advertise_session_loading(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.agent_capabilities.load_session is False

    async def test_does_not_advertise_mcp_support(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        mcp = response.agent_capabilities.mcp_capabilities
        assert (mcp.http, mcp.sse, mcp.acp) == (False, False, False)

    async def test_advertises_no_session_operations_beyond_new_prompt_cancel(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        sessions = response.agent_capabilities.session_capabilities
        assert (sessions.list, sessions.delete, sessions.fork, sessions.resume, sessions.close) == (
            None,
            None,
            None,
            None,
            None,
        )

    async def test_advertises_only_the_prompt_content_it_was_told_to(self) -> None:
        """See :class:`TestPromptContentIsDeclared` for why audio is not assumed."""
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        prompt = response.agent_capabilities.prompt_capabilities
        assert (prompt.image, prompt.audio, prompt.embedded_context) == (True, False, True)

    async def test_load_session_is_rejected_since_it_is_not_advertised(self) -> None:
        async with connect(ACPAgent(_agent())) as (conn, _):
            with pytest.raises(RequestError):
                await conn.load_session(session_id="anything", cwd="/tmp")


@pytest.mark.asyncio
class TestAuthentication:
    async def test_no_methods_advertised_without_a_provider(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.auth_methods == []

    async def test_authenticate_is_rejected_without_a_provider(self) -> None:
        async with connect(ACPAgent(_agent())) as (conn, _):
            with pytest.raises(RequestError):
                await conn.authenticate(method_id="token")

    async def test_provider_methods_are_advertised(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server, initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert [m.id for m in response.auth_methods] == ["token"]

    async def test_sessions_are_gated_until_authenticated(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await conn.new_session(cwd="/tmp")

    async def test_a_valid_credential_unlocks_sessions(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            session = await conn.new_session(cwd="/tmp")

        assert session.session_id

    async def test_a_wrong_credential_is_rejected(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await conn.authenticate(method_id="token", token="wrong")


@pytest.mark.asyncio
class TestConnectionScope:
    """Authentication and sessions belong to a connection, not to the instance."""

    async def test_a_new_connection_must_authenticate_again(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            await conn.new_session(cwd="/tmp")

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await conn.new_session(cwd="/tmp")

    async def test_a_new_connection_cannot_reach_the_previous_one_s_sessions(self) -> None:
        server = ACPAgent(_agent("ok", "ok"))

        async with connect(server) as (conn, _):
            leaked = (await conn.new_session(cwd="/tmp")).session_id

        async with connect(server) as (conn, _):
            with pytest.raises(RequestError):
                await conn.prompt(session_id=leaked, prompt=[acp.text_block("hi")])

    async def test_re_initializing_one_connection_keeps_its_sessions(self) -> None:
        """A Client may call initialize again; that must not wipe its own work."""
        server = ACPAgent(_agent("ok", "ok"))

        async with connect(server) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

            response = await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
class TestPromptContentIsDeclared:
    """Advertised modalities must match what the model behind the agent accepts."""

    async def test_audio_is_off_by_default(self) -> None:
        """Most providers AG2 ships reject audio; advertising it invites a failure."""
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        assert response.agent_capabilities.prompt_capabilities.audio is False

    async def test_image_and_documents_are_on_by_default(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        prompt = response.agent_capabilities.prompt_capabilities
        assert (prompt.image, prompt.embedded_context) == (True, True)

    async def test_a_deployment_can_declare_what_its_model_accepts(self) -> None:
        server = ACPAgent(_agent(), prompt_content=PromptContent(image=False, audio=True, embedded_context=False))

        async with connect(server, initialize=False) as (conn, _):
            response = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        prompt = response.agent_capabilities.prompt_capabilities
        assert (prompt.image, prompt.audio, prompt.embedded_context) == (False, True, False)


@pytest.mark.asyncio
class TestInitializeIsRequired:
    """The SDK router does not enforce handshake order, so this class does.

    Without it, a reconnecting Client could skip ``initialize`` and thereby skip
    the scope reset that revokes the previous connection's authentication.
    """

    async def test_sessions_are_refused_before_initialize(self) -> None:
        async with connect(ACPAgent(_agent()), initialize=False) as (conn, _):
            with pytest.raises(RequestError):
                await conn.new_session(cwd="/tmp")

    async def test_authenticate_is_refused_before_initialize(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server, initialize=False) as (conn, _):
            with pytest.raises(RequestError):
                await conn.authenticate(method_id="token", token="s3cret")

    async def test_a_reconnect_that_skips_initialize_gets_nothing(self) -> None:
        """The bypass this guards: authenticate once, reconnect, skip the handshake."""
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            leaked = (await conn.new_session(cwd="/tmp")).session_id

        async with connect(server, initialize=False) as (conn, _):
            with pytest.raises(RequestError):
                await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError):
                await conn.prompt(session_id=leaked, prompt=[acp.text_block("hi")])

    async def test_initializing_then_authenticating_works(self) -> None:
        server = ACPAgent(_agent(), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (conn, _):
            await conn.authenticate(method_id="token", token="s3cret")
            session = await conn.new_session(cwd="/tmp")

        assert session.session_id


@pytest.mark.asyncio
class TestConcurrentConnectionsAreIsolated:
    """Each connection gets its own authorization and sessions.

    A request carries no connection identity, so authorization state kept on a
    shared object would answer for whichever connection touched it last. Every
    connection therefore gets its own scope.
    """

    async def test_one_connection_s_credential_does_not_authorize_another(self) -> None:
        server = ACPAgent(_agent("ok", "ok"), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (first, _), connect(server) as (second, _):
            await second.authenticate(method_id="token", token="s3cret")
            await second.new_session(cwd="/tmp")

            with pytest.raises(RequestError):
                await first.new_session(cwd="/tmp")

    async def test_one_connection_cannot_prompt_another_s_session(self) -> None:
        server = ACPAgent(_agent("ok", "ok"), auth=StaticTokenAuth("s3cret"))

        async with connect(server) as (first, _), connect(server) as (second, _):
            await second.authenticate(method_id="token", token="s3cret")
            theirs = (await second.new_session(cwd="/tmp")).session_id

            await first.authenticate(method_id="token", token="s3cret")
            with pytest.raises(RequestError):
                await first.prompt(session_id=theirs, prompt=[acp.text_block("hi")])

    async def test_updates_go_only_to_the_connection_that_prompted(self) -> None:
        server = ACPAgent(_agent("ok", "ok"))

        async with connect(server) as (first, watcher), connect(server) as (second, owner):
            session = await second.new_session(cwd="/tmp")
            await second.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

            assert owner.updates
            assert watcher.updates == []

    async def test_each_connection_keeps_its_own_sessions(self) -> None:
        server = ACPAgent(_agent("ok", "ok"))

        async with connect(server) as (first, _):
            mine = (await first.new_session(cwd="/tmp")).session_id
            async with connect(server) as (second, _):
                theirs = (await second.new_session(cwd="/tmp")).session_id

                assert mine != theirs
                # The older connection still owns its own session.
                response = await first.prompt(session_id=mine, prompt=[acp.text_block("hi")])

        assert response.stop_reason == "end_turn"
