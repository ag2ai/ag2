# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Auth wiring — ``ApiKeyAuth`` direct tests + the ``HelloFrame``
validation path through ``Hub._dispatch_frame``.

The wire path is exercised by sending a ``HelloFrame`` directly down a
``LocalLink`` (bypassing ``HubClient.register``, whose in-process path
skips the frame layer). Any wire-transport reconnect hits exactly this
code.
"""

import asyncio

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    ApiKeyAuth,
    AuthBlock,
    AuthError,
    AuthRegistry,
    ErrorFrame,
    HelloFrame,
    Hub,
    LocalLink,
    NoAuth,
    Passport,
    Resume,
    WelcomeFrame,
)


async def _first_frame(client, timeout: float = 1.0):
    """Read one frame off a ``LocalLinkClient`` with a timeout."""
    async for frame in client.frames():
        return frame
    raise AssertionError("link closed before any frame arrived")


async def _read_frame(client, timeout: float = 1.0):
    return await asyncio.wait_for(_first_frame(client, timeout), timeout)


class TestApiKeyAuthValidate:
    """Direct unit tests for the validate() contract."""

    @pytest.mark.asyncio
    async def test_accepts_matching_static_key(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="alice")
        await auth.validate(passport, {"token": "k-alice"})

    @pytest.mark.asyncio
    async def test_rejects_mismatched_token(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="alice")
        with pytest.raises(AuthError, match="mismatch"):
            await auth.validate(passport, {"token": "k-wrong"})

    @pytest.mark.asyncio
    async def test_rejects_unknown_name(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="bob")
        with pytest.raises(AuthError, match="no api_key registered"):
            await auth.validate(passport, {"token": "k-alice"})

    @pytest.mark.asyncio
    async def test_rejects_missing_token_field(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="alice")
        with pytest.raises(AuthError, match="missing required string field 'token'"):
            await auth.validate(passport, {})

    @pytest.mark.asyncio
    async def test_rejects_empty_token(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="alice")
        with pytest.raises(AuthError, match="missing required string field 'token'"):
            await auth.validate(passport, {"token": ""})

    @pytest.mark.asyncio
    async def test_rejects_non_string_token(self) -> None:
        auth = ApiKeyAuth(keys={"alice": "k-alice"})
        passport = Passport(name="alice")
        with pytest.raises(AuthError, match="missing required string field 'token'"):
            await auth.validate(passport, {"token": 12345})

    @pytest.mark.asyncio
    async def test_resolver_supplies_dynamic_token(self) -> None:
        lookups: list[str] = []

        async def resolver(name: str) -> str | None:
            lookups.append(name)
            return {"alice": "k-alice"}.get(name)

        auth = ApiKeyAuth(resolver=resolver)
        await auth.validate(Passport(name="alice"), {"token": "k-alice"})

        assert lookups == ["alice"]

        with pytest.raises(AuthError, match="no api_key registered"):
            await auth.validate(Passport(name="bob"), {"token": "irrelevant"})
        assert lookups == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_static_keys_take_precedence_over_resolver(self) -> None:
        async def resolver(name: str) -> str | None:
            return "resolver-token"

        auth = ApiKeyAuth(keys={"alice": "static-token"}, resolver=resolver)
        # Static map hits; resolver should not be consulted at all.
        await auth.validate(Passport(name="alice"), {"token": "static-token"})
        with pytest.raises(AuthError, match="mismatch"):
            await auth.validate(Passport(name="alice"), {"token": "resolver-token"})

    @pytest.mark.asyncio
    async def test_empty_construction_rejects_everyone(self) -> None:
        auth = ApiKeyAuth()
        with pytest.raises(AuthError, match="no api_key registered"):
            await auth.validate(Passport(name="alice"), {"token": "anything"})


class TestHelloFrameAuthWiring:
    """Wire path: send HelloFrame over LocalLink and observe the hub's response."""

    @pytest.mark.asyncio
    async def test_noauth_default_accepts_handshake(self) -> None:
        """The default ``AuthRegistry`` (NoAuth only) accepts any claim."""
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(Passport(name="alice"), Resume())

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice"))
            frame = await _read_frame(client)
            assert isinstance(frame, WelcomeFrame)
            assert frame.endpoint_id == client.endpoint_id
            # Endpoint is now bound — hub maps alice → this endpoint.
            assert hub._agent_to_endpoint[passport.agent_id] == client.endpoint_id
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_api_key_happy_path_binds_endpoint(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice", auth_scheme="api_key", auth_claim={"token": "k-alice"}))
            frame = await _read_frame(client)
            assert isinstance(frame, WelcomeFrame)
            assert hub._agent_to_endpoint[passport.agent_id] == client.endpoint_id
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_api_key_wrong_token_returns_auth_failed_and_skips_bind(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice", auth_scheme="api_key", auth_claim={"token": "k-wrong"}))
            frame = await _read_frame(client)
            assert isinstance(frame, ErrorFrame)
            assert frame.code == "auth_failed"
            assert "mismatch" in frame.message
            # Endpoint MUST NOT be bound on auth failure.
            assert passport.agent_id not in hub._agent_to_endpoint
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_api_key_missing_token_returns_auth_failed(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice", auth_scheme="api_key", auth_claim={}))
            frame = await _read_frame(client)
            assert isinstance(frame, ErrorFrame)
            assert frame.code == "auth_failed"
            assert passport.agent_id not in hub._agent_to_endpoint
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_unknown_scheme_returns_auth_failed(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(Passport(name="alice"), Resume())

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice", auth_scheme="totally_made_up", auth_claim={"token": "x"}))
            frame = await _read_frame(client)
            assert isinstance(frame, ErrorFrame)
            assert frame.code == "auth_failed"
            assert "unknown auth scheme" in frame.message
            assert passport.agent_id not in hub._agent_to_endpoint
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_unknown_name_returns_not_found_before_auth(self) -> None:
        """Unknown ``name`` short-circuits to ``not_found``; we do not
        leak whether the name exists by checking the auth claim first."""
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="ghost", auth_scheme="api_key", auth_claim={"token": "k-alice"}))
            frame = await _read_frame(client)
            assert isinstance(frame, ErrorFrame)
            assert frame.code == "not_found"
        finally:
            await client.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_resolver_path_through_helloframe(self) -> None:
        """End-to-end: ApiKeyAuth(resolver=...) over the HelloFrame path."""

        async def resolver(name: str) -> str | None:
            return {"alice": "k-alice"}.get(name)

        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(resolver=resolver)]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        passport = await hub.register_identity(
            Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
            Resume(),
        )

        link = LocalLink(hub)
        client = link.client()
        try:
            await client.send_frame(HelloFrame(name="alice", auth_scheme="api_key", auth_claim={"token": "k-alice"}))
            frame = await _read_frame(client)
            assert isinstance(frame, WelcomeFrame)
            assert hub._agent_to_endpoint[passport.agent_id] == client.endpoint_id
        finally:
            await client.close()
            await hub.close()


class TestRegisterTimeAuth:
    """Hub.register_identity() also runs the auth validator at registration time.
    Confirm ApiKeyAuth integrates with that path too."""

    @pytest.mark.asyncio
    async def test_register_rejects_bad_initial_claim(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            with pytest.raises(AuthError):
                await hub.register_identity(
                    Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "wrong"})),
                    Resume(),
                )
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_register_accepts_good_initial_claim(self) -> None:
        hub = await Hub.open(
            MemoryKnowledgeStore(),
            auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]),
            ttl_sweep_interval=0,
            expectation_sweep_interval=0,
        )
        try:
            passport = await hub.register_identity(
                Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "k-alice"})),
                Resume(),
            )
            assert passport.agent_id is not None
        finally:
            await hub.close()
