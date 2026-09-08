# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from mcp_types.version import LATEST_HANDSHAKE_VERSION

from ag2 import Agent
from ag2.mcp.errors import UnknownConversationError
from ag2.mcp.executor import AgentExecutor, _session_id
from ag2.mcp.sessions import STDIO_SESSION, SessionConfig, SessionStore
from ag2.testing import TestConfig

from ._helpers import RecordingConfig


def _request_context(session_id: str | None) -> SimpleNamespace:
    """A minimal stand-in for the transport's RequestContext (HTTP shape).

    Carries a handshake-era protocol version, the era in which an MCP session
    exists at all and so the only one in which it can key a conversation.
    """
    headers = {"mcp-session-id": session_id} if session_id is not None else {}
    return SimpleNamespace(request=SimpleNamespace(headers=headers), protocol_version=LATEST_HANDSHAKE_VERSION)


def _stdio_request_context() -> SimpleNamespace:
    return SimpleNamespace(request=None, protocol_version=LATEST_HANDSHAKE_VERSION)


@pytest.mark.asyncio
class TestMultiTurnHistory:
    async def test_same_session_accumulates_history(self) -> None:
        config = RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=SessionStore())
        rc = _request_context("sess-1")

        await executor.call("ask", message="first", request_context=rc)
        await executor.call("ask", message="second", request_context=rc)

        # Second turn sees the first turn replayed from session history.
        assert len(config.calls) == 2
        assert len(config.calls[1]) > len(config.calls[0])

    async def test_different_sessions_are_isolated(self) -> None:
        config = RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=SessionStore())

        await executor.call("ask", message="first", request_context=_request_context("sess-1"))
        await executor.call("ask", message="hello", request_context=_request_context("sess-2"))

        # A brand-new session starts from an empty history, like the first turn.
        assert len(config.calls[1]) == len(config.calls[0])

    async def test_stateless_when_sessions_disabled(self) -> None:
        config = RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=None)
        rc = _request_context("sess-1")

        await executor.call("ask", message="first", request_context=rc)
        await executor.call("ask", message="second", request_context=rc)

        # No session store -> fresh stream each call -> no accumulation.
        assert len(config.calls[1]) == len(config.calls[0])

    async def test_stateless_http_without_session_id(self) -> None:
        config = RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=SessionStore())

        # HTTP request but no server-issued mcp-session-id (stateless transport).
        await executor.call("ask", message="first", request_context=_request_context(None))
        await executor.call("ask", message="second", request_context=_request_context(None))

        assert len(config.calls[1]) == len(config.calls[0])


@pytest.mark.asyncio
class TestSessionStore:
    async def test_same_session_reuses_stream_id(self) -> None:
        store = SessionStore()

        first = await store.acquire("s")
        second = await store.acquire("s")

        assert first.id == second.id

    async def test_distinct_sessions_get_distinct_streams(self) -> None:
        store = SessionStore()

        assert (await store.acquire("a")).id != (await store.acquire("b")).id

    async def test_lru_eviction_resets_history(self) -> None:
        store = SessionStore(max_sessions=1)

        original = (await store.acquire("a")).id
        await store.acquire("b")  # evicts "a"
        revived = (await store.acquire("a")).id

        # "a" was evicted, so it comes back with a fresh (empty) stream id.
        assert revived != original

    async def test_ttl_expiry_resets_history(self) -> None:
        clock = {"t": 0.0}
        store = SessionStore(ttl=10.0, clock=lambda: clock["t"])

        original = (await store.acquire("a")).id
        clock["t"] = 20.0
        revived = (await store.acquire("a")).id

        assert revived != original

    async def test_session_serializes_concurrent_turns(self) -> None:
        store = SessionStore()
        order: list[str] = []

        async def hold(tag: str) -> None:
            async with store.session("s"):
                order.append(f"{tag}-enter")
                await asyncio.sleep(0.01)
                order.append(f"{tag}-exit")

        await asyncio.gather(hold("a"), hold("b"))

        # The per-session turn lock prevents the two turns from interleaving.
        assert order in (
            ["a-enter", "a-exit", "b-enter", "b-exit"],
            ["b-enter", "b-exit", "a-enter", "a-exit"],
        )

    async def test_ttl_kept_within_window(self) -> None:
        clock = {"t": 0.0}
        store = SessionStore(ttl=10.0, clock=lambda: clock["t"])

        original = (await store.acquire("a")).id
        clock["t"] = 5.0
        assert (await store.acquire("a")).id == original


@pytest.mark.asyncio
class TestHandleNamedConversations:
    """A handle is just another key, so the registry's guarantees carry over."""

    async def test_fresh_mints_a_handle_that_resolves_to_the_same_stream(self) -> None:
        store = SessionStore()

        async with store.fresh() as minted:
            assert minted.handle is not None
            handle = minted.handle
            stream_id = minted.stream.id

        async with store.by_handle(handle) as continued:
            assert continued.stream.id == stream_id

    async def test_two_fresh_conversations_are_isolated(self) -> None:
        store = SessionStore()

        async with store.fresh() as one, store.fresh() as two:
            assert one.handle != two.handle
            assert one.stream.id != two.stream.id

    async def test_a_handle_the_store_never_minted_is_rejected(self) -> None:
        store = SessionStore()

        with pytest.raises(UnknownConversationError):
            async with store.by_handle(str(uuid4())):
                pass

    async def test_a_session_named_conversation_also_has_a_handle(self) -> None:
        store = SessionStore()

        async with store.session("sess-1") as named:
            assert named.handle is not None
            handle, stream_id = named.handle, named.stream.id

        # The handle names the same conversation the MCP session does, so a
        # handshake-era caller can migrate to explicit handles without losing it.
        async with store.by_handle(handle) as continued:
            assert continued.stream.id == stream_id

    async def test_idle_expiry_drops_a_handle_named_conversation(self) -> None:
        clock = {"t": 0.0}
        store = SessionStore(ttl=10.0, clock=lambda: clock["t"])

        async with store.fresh() as minted:
            handle = minted.handle
        assert handle is not None
        clock["t"] = 20.0

        with pytest.raises(UnknownConversationError):
            async with store.by_handle(handle):
                pass

    async def test_lru_overflow_drops_a_handle_named_conversation(self) -> None:
        store = SessionStore(max_sessions=1)

        async with store.fresh() as first:
            handle = first.handle
        assert handle is not None
        async with store.fresh():  # evicts the first
            pass

        with pytest.raises(UnknownConversationError):
            async with store.by_handle(handle):
                pass

    async def test_a_conversation_is_bound_to_the_principal_that_created_it(self) -> None:
        store = SessionStore()

        async with store.fresh(principal="alice") as minted:
            handle = minted.handle
        assert handle is not None

        async with store.by_handle(handle, principal="alice") as continued:
            assert continued.handle == handle
        with pytest.raises(UnknownConversationError):
            async with store.by_handle(handle, principal="bob"):
                pass

    async def test_acquire_records_the_principal_of_the_handle_it_mints(self) -> None:
        """A conversation first created through ``acquire`` is reachable by its creator.

        ``acquire`` mints a handle like every other entry point does; recording
        no principal for it would leave that handle nameable by nobody at all
        once authentication is configured.
        """
        store = SessionStore()

        await store.acquire("s", principal="alice")
        async with store.session("s", principal="alice") as conversation:
            handle = conversation.handle
        assert handle is not None

        async with store.by_handle(handle, principal="alice") as continued:
            assert continued.handle == handle
        with pytest.raises(UnknownConversationError):
            async with store.by_handle(handle, principal="bob"):
                pass


def test_session_store_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        SessionStore(max_sessions=0)
    with pytest.raises(ValueError):
        SessionStore(ttl=0.0)


class TestSessionId:
    def test_reads_header(self) -> None:
        assert _session_id(_request_context("abc")) == "abc"

    def test_stateless_http_returns_none(self) -> None:
        assert _session_id(_request_context(None)) is None

    def test_stdio_uses_process_sentinel(self) -> None:
        assert _session_id(_stdio_request_context()) == STDIO_SESSION


def test_session_config_defaults() -> None:
    cfg = SessionConfig()

    assert cfg.max_sessions == 1024
    assert cfg.ttl is None
    assert cfg.storage is None
