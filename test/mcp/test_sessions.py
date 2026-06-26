# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelResponse
from ag2.mcp.executor import AgentExecutor, _session_id
from ag2.mcp.sessions import STDIO_SESSION, SessionConfig, SessionStore
from ag2.testing import TestConfig


class _RecordingClient(LLMClient):
    """Wraps another client, capturing the full message list sent on each call."""

    def __init__(self, client: LLMClient, sink: list[list[BaseEvent]]) -> None:
        self.client = client
        self.sink = sink

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.sink.append(list(messages))
        return await self.client(messages, context=context, **kwargs)


class _RecordingConfig(ModelConfig):
    """Records the messages the framework sends to the LLM across every turn."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.calls: list[list[BaseEvent]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> _RecordingClient:
        return _RecordingClient(self.config.create(), self.calls)

    def create_files_client(self) -> None:
        raise NotImplementedError


def _request_context(session_id: str | None) -> SimpleNamespace:
    """A minimal stand-in for the transport's RequestContext (HTTP shape)."""
    headers = {"mcp-session-id": session_id} if session_id is not None else {}
    return SimpleNamespace(request=SimpleNamespace(headers=headers))


def _stdio_request_context() -> SimpleNamespace:
    return SimpleNamespace(request=None)


@pytest.mark.asyncio
class TestMultiTurnHistory:
    async def test_same_session_accumulates_history(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=SessionStore())
        rc = _request_context("sess-1")

        await executor.call("ask", message="first", request_context=rc)
        await executor.call("ask", message="second", request_context=rc)

        # Second turn sees the first turn replayed from session history.
        assert len(config.calls) == 2
        assert len(config.calls[1]) > len(config.calls[0])

    async def test_different_sessions_are_isolated(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=SessionStore())

        await executor.call("ask", message="first", request_context=_request_context("sess-1"))
        await executor.call("ask", message="hello", request_context=_request_context("sess-2"))

        # A brand-new session starts from an empty history, like the first turn.
        assert len(config.calls[1]) == len(config.calls[0])

    async def test_stateless_when_sessions_disabled(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
        executor = AgentExecutor(Agent("a", config=config), stream_progress=False, session_store=None)
        rc = _request_context("sess-1")

        await executor.call("ask", message="first", request_context=rc)
        await executor.call("ask", message="second", request_context=rc)

        # No session store -> fresh stream each call -> no accumulation.
        assert len(config.calls[1]) == len(config.calls[0])

    async def test_stateless_http_without_session_id(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
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
