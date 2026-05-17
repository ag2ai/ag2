# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming-chunk propagation tests.

Covers:

* ``ChunkFrame`` structure, encoding, and decoding.
* ``Hub.notify_chunk`` fan-out to all channel participants.
* ``AgentClient.on_chunk`` callback registration and delivery.
* End-to-end: LLM streaming tokens reach an observer's ``on_chunk``
  callback before the final reply envelope is posted.
"""

import asyncio
import typing
from collections.abc import Sequence
from typing import Any

import pytest

from autogen.beta import Agent, Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import EV_TEXT, Hub, HubClient, LocalLink, Passport, Resume
from autogen.beta.network.transport.frames import (
    ChunkFrame,
    Frame,
    decode_frame,
    encode_frame,
)

from ._helpers import ScriptedConfig

# ── Streaming LLM fixture ─────────────────────────────────────────────────────


class StreamingConfig(ModelConfig):
    """Test config that emits each token as a ``ModelMessageChunk`` before
    the final ``ModelMessage``.

    Pass the tokens to stream as positional args; their concatenation is
    used as the final ``ModelMessage`` body.
    """

    def __init__(self, *tokens: str) -> None:
        self._tokens: list[str] = list(tokens)

    def copy(self) -> "StreamingConfig":
        return self

    def create(self) -> "_StreamingClient":
        return _StreamingClient(self._tokens)

    def create_files_client(self) -> None:
        raise NotImplementedError("StreamingConfig has no Files API")


class _StreamingClient(LLMClient):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        for token in self._tokens:
            await context.send(ModelMessageChunk(token))
        full = "".join(self._tokens)
        message = ModelMessage(full)
        await context.send(message)
        return ModelResponse(message)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _agent_scripted(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


def _agent_streaming(name: str, *tokens: str) -> Agent:
    return Agent(name=name, config=StreamingConfig(*tokens))


# ── ChunkFrame unit tests ─────────────────────────────────────────────────────


class TestChunkFrame:
    def test_kind_is_chunk(self) -> None:
        f = ChunkFrame(channel_id="c1", sender_id="a1", content="hi", parent_envelope_id="e1")
        assert f.kind == "chunk"

    def test_in_all(self) -> None:
        from autogen.beta.network.transport.frames import __all__

        assert "ChunkFrame" in __all__

    def test_in_frame_type_alias(self) -> None:
        args = typing.get_args(Frame)
        assert ChunkFrame in args

    def test_default_recipient_id_empty(self) -> None:
        f = ChunkFrame(channel_id="c1", sender_id="a1", content="hi", parent_envelope_id="e1")
        assert f.recipient_id == ""

    def test_encode_includes_kind_discriminator(self) -> None:
        f = ChunkFrame(
            channel_id="ch-1",
            sender_id="ag-sender",
            content="hello",
            parent_envelope_id="env-123",
            recipient_id="ag-recip",
        )
        data = encode_frame(f)
        assert data["kind"] == "chunk"
        assert data["channel_id"] == "ch-1"
        assert data["content"] == "hello"
        assert data["parent_envelope_id"] == "env-123"
        assert data["recipient_id"] == "ag-recip"

    def test_decode_roundtrip(self) -> None:
        f = ChunkFrame(
            channel_id="ch-1",
            sender_id="ag-sender",
            content="world",
            parent_envelope_id="env-456",
            recipient_id="ag-recip",
        )
        decoded = decode_frame(encode_frame(f))
        assert isinstance(decoded, ChunkFrame)
        assert decoded.channel_id == f.channel_id
        assert decoded.sender_id == f.sender_id
        assert decoded.content == f.content
        assert decoded.parent_envelope_id == f.parent_envelope_id
        assert decoded.recipient_id == f.recipient_id

    def test_decode_from_raw_dict(self) -> None:
        data = {
            "kind": "chunk",
            "channel_id": "ch-99",
            "sender_id": "ag-1",
            "content": "token",
            "parent_envelope_id": "env-99",
            "recipient_id": "ag-2",
        }
        f = decode_frame(data)
        assert isinstance(f, ChunkFrame)
        assert f.content == "token"


# ── Hub.notify_chunk tests ────────────────────────────────────────────────────


class TestHubNotifyChunk:
    @pytest.mark.asyncio
    async def test_notify_chunk_fans_to_all_participants(self) -> None:
        """notify_chunk delivers ChunkFrame to every participant in the channel."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        alice = await hc.register(_agent_scripted("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent_scripted("bob"), Passport(name="bob"), Resume())

        received_alice: list[tuple[str, str, str]] = []
        received_bob: list[tuple[str, str, str]] = []

        async def cap_alice(content: str, *, channel_id: str, parent_envelope_id: str) -> None:
            received_alice.append((content, channel_id, parent_envelope_id))

        async def cap_bob(content: str, *, channel_id: str, parent_envelope_id: str) -> None:
            received_bob.append((content, channel_id, parent_envelope_id))

        alice.on_chunk(cap_alice)
        bob.on_chunk(cap_bob)

        channel = await alice.open(type="conversation", target="bob")
        cid = channel.channel_id

        await hub.notify_chunk(cid, bob.agent_id, "hello", "env-1")
        await asyncio.sleep(0.05)

        # Both alice and bob receive the chunk (sender is included in fan-out)
        assert received_alice == [("hello", cid, "env-1")]
        assert received_bob == [("hello", cid, "env-1")]

        await hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_notify_chunk_unknown_channel_is_noop(self) -> None:
        """notify_chunk for a non-existent channel must not raise."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        # Should not raise at all
        await hub.notify_chunk("nonexistent-channel", "ag-1", "hello", "env-1")
        await hub.close()

    @pytest.mark.asyncio
    async def test_notify_chunk_delivers_multiple_chunks_in_order(self) -> None:
        """Chunks are delivered to participants in the order they are posted."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        alice = await hc.register(_agent_scripted("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent_scripted("bob"), Passport(name="bob"), Resume())

        received: list[str] = []

        async def capture(content: str, *, channel_id: str, parent_envelope_id: str) -> None:
            received.append(content)

        alice.on_chunk(capture)

        channel = await alice.open(type="conversation", target="bob")
        cid = channel.channel_id

        tokens = ["The", " answer", " is", " 42"]
        for t in tokens:
            await hub.notify_chunk(cid, bob.agent_id, t, "env-q")

        await asyncio.sleep(0.05)
        assert received == tokens

        await hc.close()
        await hub.close()


# ── AgentClient.on_chunk hook tests ──────────────────────────────────────────


class TestAgentClientChunkHook:
    @pytest.mark.asyncio
    async def test_on_chunk_not_called_without_registration(self) -> None:
        """receive_chunk is a no-op when no callback is registered."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        alice = await hc.register(_agent_scripted("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent_scripted("bob"), Passport(name="bob"), Resume())

        channel = await alice.open(type="conversation", target="bob")

        # alice has no on_chunk callback — notify_chunk must not raise
        await hub.notify_chunk(channel.channel_id, bob.agent_id, "hi", "env-1")
        await asyncio.sleep(0.05)

        await hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_on_chunk_none_removes_callback(self) -> None:
        """Passing None to on_chunk deregisters the callback."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        alice = await hc.register(_agent_scripted("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent_scripted("bob"), Passport(name="bob"), Resume())

        received: list[str] = []

        async def capture(content: str, **kw: object) -> None:
            received.append(content)

        alice.on_chunk(capture)
        channel = await alice.open(type="conversation", target="bob")

        await hub.notify_chunk(channel.channel_id, bob.agent_id, "first", "env-1")
        await asyncio.sleep(0.05)
        assert received == ["first"]

        # Remove callback
        alice.on_chunk(None)
        await hub.notify_chunk(channel.channel_id, bob.agent_id, "second", "env-2")
        await asyncio.sleep(0.05)
        assert received == ["first"]  # "second" must not appear

        await hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_receive_chunk_calls_on_chunk_with_correct_args(self) -> None:
        """AgentClient.receive_chunk forwards all kwargs to the callback."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        alice = await hc.register(_agent_scripted("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent_scripted("bob"), Passport(name="bob"), Resume())

        calls: list[dict[str, object]] = []

        async def capture(content: str, *, channel_id: str, parent_envelope_id: str) -> None:
            calls.append({"content": content, "channel_id": channel_id, "parent_envelope_id": parent_envelope_id})

        alice.on_chunk(capture)
        channel = await alice.open(type="conversation", target="bob")

        await hub.notify_chunk(channel.channel_id, bob.agent_id, "tok", "parent-env")
        await asyncio.sleep(0.05)

        assert len(calls) == 1
        assert calls[0]["content"] == "tok"
        assert calls[0]["channel_id"] == channel.channel_id
        assert calls[0]["parent_envelope_id"] == "parent-env"

        await hc.close()
        await hub.close()


# ── End-to-end streaming tests ───────────────────────────────────────────────


class TestEndToEndStreaming:
    @pytest.mark.asyncio
    async def test_streaming_llm_chunks_reach_observer_on_chunk(self) -> None:
        """When a responding agent's LLM emits ``ModelMessageChunk`` events
        during ``agent.ask``, the initiating participant's ``on_chunk``
        callback receives each token before the final reply envelope."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)

        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)

        # alice: simple scripted (sends no reply — she's the initiator)
        alice = await alice_hc.register(
            _agent_scripted("alice"),
            Passport(name="alice"),
            Resume(),
        )
        # bob: streaming LLM — emits "The", " answer" as chunks then "The answer"
        await bob_hc.register(
            _agent_streaming("bob", "The", " answer"),
            Passport(name="bob"),
            Resume(),
        )

        chunks_received: list[str] = []
        chunk_channel_ids: list[str] = []

        async def capture_chunk(content: str, *, channel_id: str, parent_envelope_id: str) -> None:
            chunks_received.append(content)
            chunk_channel_ids.append(channel_id)

        alice.on_chunk(capture_chunk)

        channel = await alice.open(type="consulting", target="bob")
        await channel.send("What is the answer?")

        # Wait for bob's handler to process, stream, and post the final reply
        await asyncio.sleep(0.2)

        # Alice must have received both streaming tokens
        assert chunks_received == ["The", " answer"]

        # Chunks must carry the correct channel_id
        assert all(cid == channel.channel_id for cid in chunk_channel_ids)

        # The final EV_TEXT reply must also be in the WAL
        wal = await hub.read_wal(channel.channel_id)
        text_envs = [e for e in wal if e.event_type == EV_TEXT]
        assert len(text_envs) >= 1
        alice_q = next((e for e in text_envs if e.event_data.get("text") == "What is the answer?"), None)
        bob_reply = next((e for e in text_envs if e.event_data.get("text") == "The answer"), None)
        assert alice_q is not None, "alice's question must be in WAL"
        assert bob_reply is not None, "bob's assembled reply must be in WAL"

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_streaming_chunks_arrive_before_final_envelope(self) -> None:
        """Streaming tokens are received while the final envelope has not
        yet been posted (observable ordering of chunks-then-envelope)."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)

        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)

        alice = await alice_hc.register(
            _agent_scripted("alice"),
            Passport(name="alice"),
            Resume(),
        )
        await bob_hc.register(
            _agent_streaming("bob", "A", "B", "C"),
            Passport(name="bob"),
            Resume(),
        )

        events: list[str] = []  # "chunk:<tok>" or "envelope"

        async def capture_chunk(content: str, **kw: object) -> None:
            events.append(f"chunk:{content}")

        async def capture_envelope(env: object) -> None:
            from autogen.beta.network import EV_TEXT  # type: ignore[attr-defined]

            if (
                hasattr(env, "event_type")
                and env.event_type == EV_TEXT
                and hasattr(env, "event_data")
                and env.event_data.get("text") == "ABC"
            ):
                events.append("envelope")

        alice.on_chunk(capture_chunk)
        orig = alice._on_envelope

        async def combined_envelope(env: object) -> None:
            await capture_envelope(env)
            if orig is not None:
                await orig(env)

        alice.on_envelope(combined_envelope)

        channel = await alice.open(type="consulting", target="bob")
        await channel.send("go")
        await asyncio.sleep(0.2)

        # Chunks must appear before the envelope in the event log
        assert "chunk:A" in events
        assert "chunk:B" in events
        assert "chunk:C" in events
        assert "envelope" in events

        chunk_indices = [i for i, e in enumerate(events) if e.startswith("chunk:")]
        envelope_index = events.index("envelope")
        assert all(i < envelope_index for i in chunk_indices), f"all chunks must precede the final envelope: {events}"

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_streaming_does_not_break_channel_when_no_on_chunk(self) -> None:
        """When no on_chunk callback is registered, streaming LLM still
        completes and posts the final reply normally."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        link = LocalLink(hub)

        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)

        alice = await alice_hc.register(
            _agent_scripted("alice"),
            Passport(name="alice"),
            Resume(),
        )
        await bob_hc.register(
            _agent_streaming("bob", "X", "Y"),
            Passport(name="bob"),
            Resume(),
        )

        # No on_chunk registered — must not raise
        channel = await alice.open(type="consulting", target="bob")
        await channel.send("question")
        await asyncio.sleep(0.2)

        wal = await hub.read_wal(channel.channel_id)
        assert any(e.event_type == EV_TEXT and e.event_data.get("text") == "XY" for e in wal)

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()


# ── Module surface tests ──────────────────────────────────────────────────────


class TestModuleSurface:
    def test_chunk_frame_importable_from_transport(self) -> None:
        from autogen.beta.network.transport.frames import ChunkFrame as ChunkFrameAlias

        assert ChunkFrameAlias is ChunkFrame

    def test_hub_has_notify_chunk_method(self) -> None:
        assert callable(getattr(Hub, "notify_chunk", None))

    def test_agent_client_has_receive_chunk_method(self) -> None:
        from autogen.beta.network.client.agent_client import AgentClient

        assert callable(getattr(AgentClient, "receive_chunk", None))

    def test_agent_client_has_on_chunk_method(self) -> None:
        from autogen.beta.network.client.agent_client import AgentClient

        assert callable(getattr(AgentClient, "on_chunk", None))

    def test_human_client_has_receive_chunk_method(self) -> None:
        from autogen.beta.network.client.human_client import HumanClient

        assert callable(getattr(HumanClient, "receive_chunk", None))
