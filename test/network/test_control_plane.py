# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-process control plane over a real ``serve_ws`` loopback.

These exercises drive the **public** ``HubClient`` / ``AgentClient``
API with no in-process hub reference (``HubClient(WsLink(url))``), so
every control-plane operation — register, discovery, channel creation,
posting an envelope, reconnect-by-name, task checkpoints — travels the
wire as a ``RequestFrame`` / ``ResponseFrame`` RPC. They use scripted
(no-LLM) agents so the default notify handler runs deterministically;
the point is to prove the API works identically across a process
boundary, not to test model behaviour.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    ApiKeyAuth,
    AuthBlock,
    AuthError,
    AuthRegistry,
    Hub,
    HubClient,
    NoAuth,
    NotFoundError,
    Passport,
    Resume,
    WsLink,
    serve_ws,
)

from ._helpers import ScriptedConfig, wait_for_text_count


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig(*replies))


async def _new_hub(auth: AuthRegistry | None = None) -> Hub:
    return await Hub.open(
        MemoryKnowledgeStore(),
        auth=auth,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )


def _bound_port(server) -> int:
    return server.sockets[0].getsockname()[1]


def _url(server) -> str:
    return f"ws://127.0.0.1:{_bound_port(server)}"


class TestRegistrationAndDiscoveryOverWire:
    @pytest.mark.asyncio
    async def test_register_then_discover_across_connections(self) -> None:
        """A remote client registers an identity; a second remote client
        discovers it — both purely over the wire, no shared hub ref."""
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            url = _url(server)
            alice_hc = HubClient(WsLink(url))  # no hub= → remote mode
            bob_hc = HubClient(WsLink(url))
            assert alice_hc.remote is True
            try:
                alice = await alice_hc.register(
                    _agent("alice"), Passport(name="alice"), Resume(claimed_capabilities=["policy"])
                )
                assert alice.agent_id is not None

                # Second connection sees alice via discovery RPCs.
                found = await bob_hc.get_agent("alice")
                assert found.agent_id == alice.agent_id
                listed = await bob_hc.list_agents(capability="policy")
                assert [p.name for p in listed] == ["alice"]
            finally:
                await alice_hc.close()
                await bob_hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_get_unknown_agent_raises_not_found_over_wire(self) -> None:
        """A hub-side ``NotFoundError`` propagates back through the
        ``ResponseFrame`` and re-raises as the same type client-side."""
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            hc = HubClient(WsLink(_url(server)))
            try:
                with pytest.raises(NotFoundError):
                    await hc.get_agent("ghost")
            finally:
                await hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_register_with_bad_api_key_raises_auth_error_over_wire(self) -> None:
        """Auth validation runs hub-side inside the register op; a bad
        claim surfaces as ``AuthError`` on the calling client."""
        hub = await _new_hub(auth=AuthRegistry([NoAuth(), ApiKeyAuth(keys={"alice": "k-alice"})]))
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            hc = HubClient(WsLink(_url(server)))
            try:
                with pytest.raises(AuthError):
                    await hc.register(
                        _agent("alice"),
                        Passport(name="alice", auth=AuthBlock(scheme="api_key", claim={"token": "wrong"})),
                        Resume(),
                    )
            finally:
                await hc.close()
        await hub.close()


class TestChannelAndDeliveryOverWire:
    @pytest.mark.asyncio
    async def test_discussion_round_trip_two_remote_processes(self) -> None:
        """Two independent remote connections drive a discussion end to
        end: ``open`` + ``send`` cross the wire, the hub dispatches a
        ``NotifyFrame`` to the remote respondent, whose default handler
        folds adapter state locally, runs its (scripted) turn, and posts
        the reply back — all through ``RequestFrame`` RPC."""
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            url = _url(server)
            alice_hc = HubClient(WsLink(url))
            bob_hc = HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
                await bob_hc.register(_agent("bob", "Hi Alice, good to meet you."), Passport(name="bob"), Resume())

                channel = await alice.open(type="discussion", target=["bob"], knobs={"ordering": "round_robin"})
                await channel.send("Hello Bob, introduce yourself.")

                # alice's opening line + bob's scripted reply land in the WAL.
                await wait_for_text_count(hub, channel.channel_id, 2, timeout=5.0)
                wal = await hub.read_wal(channel.channel_id)
                texts = [e for e in wal if e.event_type == EV_TEXT]
                senders = {hub.name_for(e.sender_id) for e in texts}
                assert senders == {"alice", "bob"}
            finally:
                await alice_hc.close()
                await bob_hc.close()
        await hub.close()

    @pytest.mark.asyncio
    async def test_reconnect_by_name_resumes_pending_turn_over_wire(self) -> None:
        """Bob connects, is handed a turn, then drops before answering.
        A fresh connection ``attach``es to the same identity over the
        wire (HelloFrame handshake) and ``resume_pending_turns`` re-fires
        the turn against a remote hub — the reply lands without a
        duplicate."""
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            url = _url(server)
            alice_hc = HubClient(WsLink(url))
            # First bob connection auto-acks the invite (default handler)
            # but its empty script posts no reply, so the turn stays
            # pending on the hub even though the delivery was acked.
            bob_hc1 = HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
                bob1 = await bob_hc1.register(_agent("bob"), Passport(name="bob"), Resume())
                bob_id = bob1.agent_id

                channel = await alice.open(type="discussion", target=["bob"], knobs={"ordering": "round_robin"})
                await channel.send("Bob, please weigh in.")
                await wait_for_text_count(hub, channel.channel_id, 1, timeout=5.0)
                await asyncio.sleep(0.1)  # let bob1's empty turn + ack settle

                # Bob's process "restarts": drop the old connection, attach
                # a fresh one to the same name with a real handler.
                await bob_hc1.close()

                bob_hc2 = HubClient(WsLink(url))
                try:
                    bob2 = await bob_hc2.attach(
                        _agent("bob", "Sure — here is my take."),
                        name="bob",
                        passport=Passport(name="bob"),
                        resume=Resume(),
                    )
                    assert bob2.agent_id == bob_id  # same identity re-bound

                    resumed = await bob2.resume_pending_turns()
                    assert resumed == 1

                    await wait_for_text_count(hub, channel.channel_id, 2, timeout=5.0)
                    wal = await hub.read_wal(channel.channel_id)
                    bob_texts = [e for e in wal if e.event_type == EV_TEXT and e.sender_id == bob_id]
                    assert len(bob_texts) == 1  # resumed once, no duplicate
                finally:
                    await bob_hc2.close()
            finally:
                await alice_hc.close()
        await hub.close()


class TestTaskOpsOverWire:
    @pytest.mark.asyncio
    async def test_checkpoint_round_trip_over_wire(self) -> None:
        """``checkpoint_task`` / ``read_task_checkpoint`` round-trip the
        opaque resume blob through the hub over the wire."""
        hub = await _new_hub()
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            hc = HubClient(WsLink(_url(server)))
            try:
                await hc.register(_agent("alice"), Passport(name="alice"), Resume())
                assert await hc.read_task_checkpoint("task-1") is None
                await hc.checkpoint_task("task-1", {"step": 3, "scratch": ["a", "b"]})
                assert await hc.read_task_checkpoint("task-1") == {"step": 3, "scratch": ["a", "b"]}
            finally:
                await hc.close()
        await hub.close()
