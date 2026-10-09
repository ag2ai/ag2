# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire control-plane ops are bound to the calling connection's identities.

A connection acts only as the agents it registered or re-attached to
with an authenticated ``HelloFrame``. Each test drives a real
``serve_ws`` loopback with ``ApiKeyAuth``: mallory holds only her own
key and tries to act as, or read the private state of, bob.
"""

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

from ag2 import Agent, Context
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_REJECT,
    EV_TASK_CANCELLED,
    EV_TASK_CANCEL_REQUEST,
    EV_TEXT,
    AccessDeniedError,
    ApiKeyAuth,
    AuthBlock,
    AuthRegistry,
    BaseHubListener,
    ChannelMetadata,
    ChannelState,
    Envelope,
    ErrorFrame,
    HelloFrame,
    Hub,
    HubClient,
    LimitsBlock,
    NoAuth,
    NotFoundError,
    Passport,
    PingFrame,
    PongFrame,
    ProtocolError,
    ReceiptFrame,
    RequestFrame,
    ResponseFrame,
    Resume,
    Rule,
    WelcomeFrame,
    WsLink,
    WsLinkClient,
    serve_ws,
)
from ag2.network.hub.layout import passport_path
from ag2.network.task_mirror import TaskMirror
from ag2.network.testing import RacingKnowledgeStore
from ag2.stream import MemoryStream
from ag2.task import TaskMetadata, TaskSpec, TaskStarted, TaskState

from ._helpers import ScriptedConfig

_NAMES = ("alice", "bob", "carol", "mallory")


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


def _passport(name: str) -> Passport:
    return Passport(name=name, auth=AuthBlock(scheme="api_key", claim={"token": f"k-{name}"}))


class _InvitePosted(BaseHubListener):
    """Signals once the hub has posted a channel invite."""

    def __init__(self) -> None:
        self.posted = asyncio.Event()

    async def on_envelope_posted(self, envelope: Envelope, metadata: ChannelMetadata) -> None:
        if envelope.event_type == EV_CHANNEL_INVITE:
            self.posted.set()


class _WalReadCountingStore(MemoryKnowledgeStore):
    """Counts WAL reads; each read yields to the loop, as a real store's I/O does."""

    def __init__(self) -> None:
        super().__init__()
        self.wal_reads = 0

    async def read(self, path: str) -> str | None:
        if path.endswith("/wal.jsonl"):
            self.wal_reads += 1
        content = await super().read(path)
        await asyncio.sleep(0)  # hand the loop over after the read, as I/O completion does
        return content


async def _hub_with_live_task(store: MemoryKnowledgeStore) -> tuple[Hub, str, Envelope]:
    """In-process hub with an alice–bob channel, bob's live task in it, and
    carol's (a non-participant) well-formed cancel request for that task."""
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    carol = await hub.register(_agent("carol"))
    channel = await alice.open(type="conversation", target="bob")
    await hub.observe_task(
        TaskMetadata(
            task_id="t-bob",
            owner_id=bob.agent_id,
            spec=TaskSpec(title="t"),
            state=TaskState.RUNNING,
            channel_id=channel.channel_id,
        )
    )
    request = Envelope(
        channel_id=channel.channel_id,
        sender_id=carol.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TASK_CANCEL_REQUEST,
        event_data={"task_id": "t-bob", "reason": "wrap up"},
        task_id="t-bob",
    )
    return hub, channel.channel_id, request


class _TaskEvents(BaseHubListener):
    """Collects every task event the hub fans out."""

    def __init__(self, fired: list[tuple[str, str, dict]]) -> None:
        self.fired = fired

    async def on_task_event(self, task_id: str, kind: str, payload: dict) -> None:
        self.fired.append((task_id, kind, payload))


@asynccontextmanager
async def _serve(
    *,
    allow_no_auth: bool = False,
    allow_remote_agents: bool = False,
    store: MemoryKnowledgeStore | None = None,
) -> AsyncGenerator[tuple[Hub, str]]:
    api_key = ApiKeyAuth(keys={name: f"k-{name}" for name in _NAMES})
    hub = await Hub.open(
        store if store is not None else MemoryKnowledgeStore(),
        auth=AuthRegistry([NoAuth(), api_key] if allow_no_auth else [api_key]),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        allow_remote_agent_registration=allow_remote_agents,
    )
    try:
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            yield hub, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        await hub.close()


@pytest.mark.asyncio
class TestActingAsAnotherAgent:
    async def test_connection_acts_as_each_agent_it_registered_and_no_other(self) -> None:
        async with _serve() as (hub, url):
            shared_hc, bob_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await shared_hc.register(_agent("alice"), _passport("alice"), Resume())
                carol = await shared_hc.register(_agent("carol"), _passport("carol"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume(), skill_md="bob's skill")

                await shared_hc.set_skill(alice.agent_id, "alice's skill")
                await shared_hc.set_skill(carol.agent_id, "carol's skill")
                with pytest.raises(AccessDeniedError):
                    await shared_hc.set_skill(bob.agent_id, "overwritten")

                assert await hub.get_skill(carol.agent_id) == "carol's skill"
                assert await hub.get_skill(bob.agent_id) == "bob's skill"
            finally:
                await shared_hc.close()
                await bob_hc.close()

    async def test_post_envelope_as_another_agent_is_denied(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=bob.agent_id,
                            audience=None,
                            event_type=EV_TEXT,
                            event_data={"text": "spoofed"},
                        )
                    )

                wal = await hub.read_wal(channel.channel_id)
                assert not any(e.event_type == EV_TEXT for e in wal)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_update_of_another_agents_task_is_denied(self) -> None:
        async with _serve() as (hub, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob", owner_id=bob.agent_id, spec=TaskSpec(title="t"), state=TaskState.RUNNING
                    )
                )

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.update_task("t-bob", state=TaskState.FAILED, error="sabotaged")

                assert (await hub.get_task("t-bob")).state == TaskState.RUNNING
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_receipt_for_another_agent_does_not_advance_its_cursor(self) -> None:
        async with _serve() as (hub, url):
            link = WsLinkClient(url)
            try:
                await hub.register_identity(_passport("mallory"), Resume())
                # bob holds no connection, so nothing acks on his behalf.
                bob = await hub.register_identity(_passport("bob"), Resume())
                assert bob.agent_id is not None
                await link.open()
                frames = aiter(link.frames())
                await link.send_frame(
                    HelloFrame(name="mallory", auth_scheme="api_key", auth_claim={"token": "k-mallory"})
                )
                assert isinstance(await asyncio.wait_for(anext(frames), 2.0), WelcomeFrame)

                await link.send_frame(
                    ReceiptFrame(envelope_id="z" * 32, status="ack", recipient_id=bob.agent_id, channel_id="c-1")
                )
                # Frames on one connection are handled in order: once this
                # RPC answers, the receipt above has been processed.
                await link.send_frame(RequestFrame(request_id="r-1", op="get_agent", params={"name_or_id": "mallory"}))
                assert isinstance(await asyncio.wait_for(anext(frames), 2.0), ResponseFrame)

                assert hub.inbox_cursor(bob.agent_id, "c-1") == ""
            finally:
                await link.close()

    async def test_checkpoint_of_an_unobserved_task_belongs_to_its_first_writer(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.checkpoint_task("t-bob", {"step": 1})

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.read_task_checkpoint("t-bob")
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.checkpoint_task("t-bob", {"step": 99})

                assert await bob_hc.read_task_checkpoint("t-bob") == {"step": 1}
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_observing_a_task_id_checkpointed_by_another_agent_is_denied(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.checkpoint_task("t-bob", {"step": 1})

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.observe_task(
                        TaskMetadata(
                            task_id="t-bob",
                            owner_id=mallory.agent_id,
                            spec=TaskSpec(title="t"),
                            state=TaskState.RUNNING,
                        )
                    )

                assert await bob_hc.read_task_checkpoint("t-bob") == {"step": 1}
            finally:
                await bob_hc.close()
                await mallory_hc.close()


@pytest.mark.asyncio
class TestChannelScope:
    async def test_only_participants_read_the_channel_wal(self) -> None:
        async with _serve() as (_, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                assert await bob_hc.read_wal(channel.channel_id)
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.read_wal(channel.channel_id)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_list_channels_without_agent_lists_only_the_connections_channels(self) -> None:
        async with _serve() as (_, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                assert [m.channel_id for m in await bob_hc.list_channels()] == [channel.channel_id]
                assert await mallory_hc.list_channels() == []
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_non_participant_content_is_refused_whatever_the_adapter_accepts(self) -> None:
        # The conversation adapter accepts non-text events from anyone.
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type="app.note",
                            event_data={"note": "injected"},
                        )
                    )

                assert not any(e.event_type == "app.note" for e in await hub.read_wal(channel.channel_id))
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_non_participant_protocol_event_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type=EV_TASK_CANCELLED,
                            event_data={"text": "injected"},
                        )
                    )

                wal = await hub.read_wal(channel.channel_id)
                assert not any(e.sender_id == mallory.agent_id for e in wal)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_non_participant_invite_reject_leaves_the_channel_pending(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            # bob holds no connection, so the invite waits for his ack.
            await hub.register_identity(_passport("bob"), Resume())
            opening: asyncio.Task[object] | None = None
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                invited = _InvitePosted()
                hub.register_listener(invited)
                opening = asyncio.create_task(alice.open(type="conversation", target="bob"))
                await asyncio.wait_for(invited.posted.wait(), 2.0)
                (pending,) = await hub.list_channels()

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=pending.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type=EV_CHANNEL_INVITE_REJECT,
                            event_data={"channel_id": pending.channel_id},
                        )
                    )

                assert (await hub.get_channel(pending.channel_id)).state == ChannelState.PENDING
            finally:
                if opening is not None:
                    opening.cancel()
                    await asyncio.gather(opening, return_exceptions=True)
                await alice_hc.close()
                await mallory_hc.close()

    async def test_observing_a_task_onto_a_foreign_channel_is_denied(self) -> None:
        async with _serve() as (_, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.observe_task(
                        TaskMetadata(
                            task_id="t-mallory",
                            owner_id=mallory.agent_id,
                            spec=TaskSpec(title="t"),
                            state=TaskState.RUNNING,
                            channel_id=channel.channel_id,
                        )
                    )
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_task_id_that_is_not_a_single_path_segment_is_refused(self) -> None:
        store = MemoryKnowledgeStore()
        async with _serve(store=store) as (_, url):
            bob_hc = HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())

                with pytest.raises(ProtocolError, match="invalid task_id"):
                    await bob_hc.checkpoint_task("../agents/x", {"step": 1})

                assert await store.read("/agents/x/checkpoint.json") is None
            finally:
                await bob_hc.close()


@pytest.mark.asyncio
class TestPeerCancelRequest:
    async def test_outsider_sends_one_cancel_request_per_task(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, carol_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                carol = await carol_hc.register(_agent("carol"), _passport("carol"), Resume())
                channel = await alice.open(type="conversation", target="bob")
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob",
                        owner_id=bob.agent_id,
                        spec=TaskSpec(title="t"),
                        state=TaskState.RUNNING,
                        channel_id=channel.channel_id,
                    )
                )
                request = Envelope(
                    channel_id=channel.channel_id,
                    sender_id=carol.agent_id,
                    audience=[bob.agent_id],
                    event_type=EV_TASK_CANCEL_REQUEST,
                    event_data={"task_id": "t-bob", "reason": "wrap up"},
                    task_id="t-bob",
                )

                await carol_hc.post_envelope(request)
                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await carol_hc.post_envelope(request)

                wal = await hub.read_wal(channel.channel_id)
                assert sum(e.event_type == EV_TASK_CANCEL_REQUEST for e in wal) == 1
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await carol_hc.close()

    async def test_repeat_cancel_is_refused_without_reading_the_wal_even_after_hydrate(self) -> None:
        store = _WalReadCountingStore()
        hub, _, request = await _hub_with_live_task(store)
        try:
            await hub.post_envelope(dataclasses.replace(request))
            await hub.hydrate()
            store.wal_reads = 0

            with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                await hub.post_envelope(dataclasses.replace(request))

            assert store.wal_reads == 0
        finally:
            await hub.close()

    async def test_concurrent_duplicate_cancels_accept_exactly_one(self) -> None:
        hub, channel_id, request = await _hub_with_live_task(_WalReadCountingStore())
        try:
            results = await asyncio.gather(
                hub.post_envelope(dataclasses.replace(request)),
                hub.post_envelope(dataclasses.replace(request)),
                return_exceptions=True,
            )

            assert sorted(type(r).__name__ for r in results) == ["ProtocolError", "str"]
            wal = await hub.read_wal(channel_id)
            assert sum(e.event_type == EV_TASK_CANCEL_REQUEST for e in wal) == 1
        finally:
            await hub.close()

    async def test_outsider_cancel_request_for_the_owners_task_is_accepted(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, carol_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                carol = await carol_hc.register(_agent("carol"), _passport("carol"), Resume())
                channel = await alice.open(type="conversation", target="bob")
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob",
                        owner_id=bob.agent_id,
                        spec=TaskSpec(title="t"),
                        state=TaskState.RUNNING,
                        channel_id=channel.channel_id,
                    )
                )

                # Shaped exactly as ``tasks(action="cancel")`` sends it.
                await carol_hc.post_envelope(
                    Envelope(
                        channel_id=channel.channel_id,
                        sender_id=carol.agent_id,
                        audience=[bob.agent_id],
                        event_type=EV_TASK_CANCEL_REQUEST,
                        event_data={"task_id": "t-bob", "reason": "wrap up"},
                        task_id="t-bob",
                    )
                )

                wal = await hub.read_wal(channel.channel_id)
                assert [e.sender_id for e in wal if e.event_type == EV_TASK_CANCEL_REQUEST] == [carol.agent_id]
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await carol_hc.close()

    async def test_outsider_cancel_request_not_shaped_as_the_tasks_tool_sends_it_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob",
                        owner_id=bob.agent_id,
                        spec=TaskSpec(title="t"),
                        state=TaskState.RUNNING,
                        channel_id=channel.channel_id,
                    )
                )

                unconstrained = Envelope(
                    channel_id=channel.channel_id,
                    sender_id=mallory.agent_id,
                    audience=None,
                    event_type=EV_TASK_CANCEL_REQUEST,
                    event_data={"text": "injected"},
                    task_id="nope",
                )
                non_text_reason = Envelope(
                    channel_id=channel.channel_id,
                    sender_id=mallory.agent_id,
                    audience=[bob.agent_id],
                    event_type=EV_TASK_CANCEL_REQUEST,
                    event_data={"task_id": "t-bob", "reason": {"text": "injected"}},
                    task_id="t-bob",
                )
                for envelope in (unconstrained, non_text_reason):
                    with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                        await mallory_hc.post_envelope(envelope)

                wal = await hub.read_wal(channel.channel_id)
                assert not any(e.sender_id == mallory.agent_id for e in wal)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()


@pytest.mark.asyncio
class TestReattach:
    async def test_hello_with_a_non_object_claim_is_refused(self) -> None:
        async with _serve() as (_, url):
            bob_hc = HubClient(WsLink(url))
            link = WsLinkClient(url)
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await link.open()

                await link.send_frame(HelloFrame(name="bob", auth_scheme="api_key", auth_claim="k-bob"))  # type: ignore[arg-type]
                reply = await asyncio.wait_for(anext(aiter(link.frames())), 2.0)

                assert isinstance(reply, ErrorFrame)
                assert reply.code == "auth_failed"
            finally:
                await bob_hc.close()
                await link.close()

    async def test_reattach_moves_the_identity_to_the_new_connection(self) -> None:
        async with _serve() as (hub, url):
            old_hc, new_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await old_hc.register(_agent("bob"), _passport("bob"), Resume())
                await new_hc.attach(_agent("bob"), name="bob", passport=_passport("bob"))

                await new_hc.set_skill(bob.agent_id, "from the new connection")
                with pytest.raises(AccessDeniedError):
                    await old_hc.set_skill(bob.agent_id, "from the old connection")

                assert await hub.get_skill(bob.agent_id) == "from the new connection"
            finally:
                await old_hc.close()
                await new_hc.close()

    async def test_hello_with_a_scheme_other_than_the_registered_one_is_refused(self) -> None:
        async with _serve(allow_no_auth=True) as (_, url):
            bob_hc = HubClient(WsLink(url))
            mallory_link = WsLinkClient(url)
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_link.open()

                await mallory_link.send_frame(HelloFrame(name="bob", auth_scheme="none", auth_claim={}))
                reply = await asyncio.wait_for(anext(aiter(mallory_link.frames())), 2.0)

                assert isinstance(reply, ErrorFrame)
                assert reply.code == "auth_failed"
            finally:
                await bob_hc.close()
                await mallory_link.close()


@pytest.mark.asyncio
class TestRemoteAgent:
    async def test_wire_remote_agent_registration_is_off_by_default(self) -> None:
        async with _serve() as (_, url):
            mallory_hc, carol_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.register(
                        _agent("carol"),
                        Passport(name="carol", kind="remote_agent", auth=AuthBlock(scheme="whatever")),
                        Resume(),
                        attach_plugin=False,
                    )

                carol = await carol_hc.register(_agent("carol"), _passport("carol"), Resume())
                assert carol.agent_id is not None
            finally:
                await mallory_hc.close()
                await carol_hc.close()

    async def test_connection_without_an_agent_cannot_register_a_remote_agent(self) -> None:
        async with _serve(allow_no_auth=True, allow_remote_agents=True) as (hub, url):
            keyless_hc = HubClient(WsLink(url))
            try:
                with pytest.raises(AccessDeniedError):
                    await keyless_hc.register(
                        _agent("partner"), Passport(name="partner", kind="remote_agent"), Resume(), attach_plugin=False
                    )

                assert hub.find_agent_id("partner") is None
            finally:
                await keyless_hc.close()

    async def test_hello_as_a_remote_agent_without_its_owner_is_refused(self) -> None:
        async with _serve(allow_no_auth=True, allow_remote_agents=True) as (_, url):
            owner_hc = HubClient(WsLink(url))
            keyless_link = WsLinkClient(url)
            try:
                await owner_hc.register(_agent("alice"), _passport("alice"), Resume())
                await owner_hc.register(
                    _agent("partner"), Passport(name="partner", kind="remote_agent"), Resume(), attach_plugin=False
                )
                await keyless_link.open()

                await keyless_link.send_frame(HelloFrame(name="partner", auth_scheme="none", auth_claim={}))
                reply = await asyncio.wait_for(anext(aiter(keyless_link.frames())), 2.0)

                assert isinstance(reply, ErrorFrame)
                assert reply.code == "auth_failed"
            finally:
                await owner_hc.close()
                await keyless_link.close()

    async def test_owner_reattaches_its_remote_agent_on_a_new_connection(self) -> None:
        async with _serve(allow_no_auth=True, allow_remote_agents=True) as (hub, url):
            first_hc, second_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await first_hc.register(_agent("alice"), _passport("alice"), Resume())
                partner = await first_hc.register(
                    _agent("partner"),
                    Passport(name="partner", kind="remote_agent", auth=AuthBlock(scheme="a2a")),
                    Resume(),
                    attach_plugin=False,
                )
                await first_hc.close()
                await hub.hydrate()  # ownership is read back from the store

                await second_hc.attach(_agent("alice"), name="alice", passport=_passport("alice"))
                await second_hc.attach(_agent("partner"), name="partner", attach_plugin=False)
                await second_hc.set_skill(partner.agent_id, "federated peer")

                assert await hub.get_skill(partner.agent_id) == "federated peer"
            finally:
                await first_hc.close()
                await second_hc.close()


@pytest.mark.asyncio
class TestCredentials:
    async def test_passports_read_over_the_wire_carry_no_claim(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())

                assert bob.passport.auth.claim == {}
                assert (await mallory_hc.get_agent("bob")).auth == AuthBlock(scheme="api_key")
                assert [p.auth.claim for p in await mallory_hc.list_agents()] == [{}, {}]
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_claim_is_not_kept_in_the_store(self) -> None:
        store = MemoryKnowledgeStore()
        async with _serve(store=store) as (hub, url):
            bob_hc = HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                assert bob.agent_id is not None
                assert "k-bob" not in (await store.read(passport_path(bob.agent_id)) or "")

                # A store written before claims were dropped is cleaned on hydrate.
                legacy = await hub.get_agent(bob.agent_id)
                legacy_dict = legacy.to_dict()
                legacy_dict["auth"]["claim"] = {"token": "k-bob"}
                await store.write(passport_path(bob.agent_id), json.dumps(legacy_dict))
                await hub.hydrate()

                assert (await hub.get_agent(bob.agent_id)).auth.claim == {}
                assert "k-bob" not in (await store.read(passport_path(bob.agent_id)) or "")
            finally:
                await bob_hc.close()


@pytest.mark.asyncio
async def test_participant_cannot_post_a_hub_only_channel_event() -> None:
    async with _serve() as (hub, url):
        alice_hc, bob_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
        try:
            alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
            bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
            channel = await alice.open(type="conversation", target="bob")

            with pytest.raises(ProtocolError, match="emitted only by the hub"):
                await bob_hc.post_envelope(
                    Envelope(
                        channel_id=channel.channel_id,
                        sender_id=bob.agent_id,
                        audience=None,
                        event_type=EV_CHANNEL_CLOSED,
                        event_data={"channel_id": channel.channel_id, "reason": "fake"},
                    )
                )

            assert (await hub.get_channel(channel.channel_id)).state == ChannelState.ACTIVE
        finally:
            await alice_hc.close()
            await bob_hc.close()


@pytest.mark.asyncio
class TestTaskEvents:
    async def test_connection_without_an_agent_cannot_fire_task_events(self) -> None:
        async with _serve() as (hub, url):
            bob_hc, unbound_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())

                with pytest.raises(AccessDeniedError):
                    await unbound_hc.fire_task_event(
                        "t-forged", "completed", {"owner_id": bob.agent_id, "capability": "payments"}
                    )

                assert not any(r.get("task_id") == "t-forged" for r in await hub.audit_log.read_all())
            finally:
                await bob_hc.close()
                await unbound_hc.close()

    async def test_mirror_failed_for_an_id_checkpointed_by_another_agent_is_denied(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.checkpoint_task("t-bob", {"step": 1})

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.fire_task_event("t-bob", "mirror_failed", {"owner_id": mallory.agent_id})
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_concurrent_first_checkpoints_leave_one_writer(self) -> None:
        async with _serve(store=RacingKnowledgeStore("/checkpoint_writers.json")) as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())

                results = await asyncio.gather(
                    bob_hc.checkpoint_task("t-race", {"by": "bob"}),
                    mallory_hc.checkpoint_task("t-race", {"by": "mallory"}),
                    return_exceptions=True,
                )

                assert sum(isinstance(r, AccessDeniedError) for r in results) == 1
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_agent_cannot_forge_task_events_for_unobserved_tasks_or_other_owners(self) -> None:
        async with _serve() as (hub, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.fire_task_event("t-forged", "completed", {"owner_id": mallory.agent_id})
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.fire_task_event("t-forged", "mirror_failed", {"owner_id": bob.agent_id})
                await mallory_hc.fire_task_event("t-own", "mirror_failed", {"owner_id": mallory.agent_id})

                assert not any(r.get("task_id") == "t-forged" for r in await hub.audit_log.read_all())
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    async def test_mirror_failure_naming_a_foreign_channel_or_extra_fields_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError):
                    await mallory_hc.fire_task_event(
                        "x-1",
                        "mirror_failed",
                        {"channel_id": channel.channel_id, "capability": "payments", "outcome": "completed"},
                    )
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.fire_task_event(
                        "x-1", "mirror_failed", {"owner_id": mallory.agent_id, "channel_id": channel.channel_id}
                    )
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_terminal_task_event_for_an_observed_task_is_refused(self) -> None:
        async with _serve() as (hub, url):
            mallory_hc = HubClient(WsLink(url))
            try:
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await mallory_hc.observe_task(
                    TaskMetadata(
                        task_id="m-1", owner_id=mallory.agent_id, spec=TaskSpec(title="t"), state=TaskState.RUNNING
                    )
                )

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.fire_task_event("m-1", "completed", {"reason": "forged"})

                assert not any(r.get("task_id") == "m-1" for r in await hub.audit_log.read_all())
                assert (await hub.get_task("m-1")).state == TaskState.RUNNING
            finally:
                await mallory_hc.close()

    async def test_task_mirror_reports_its_failure_over_the_wire(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            fired: list[tuple[str, str, dict]] = []
            listener = _TaskEvents(fired)
            hub.register_listener(listener)
            try:
                alice = await alice_hc.register(
                    _agent("alice"),
                    _passport("alice"),
                    Resume(),
                    rule=Rule(limits=LimitsBlock(max_concurrent_tasks=1)),
                )
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                channel = await alice.open(type="conversation", target="bob")
                stream = MemoryStream()
                TaskMirror(hub_client=alice_hc, owner_id=alice.agent_id, channel_id=channel.channel_id).attach(stream)

                # The second task exceeds alice's cap, so the hub refuses to observe it.
                await stream.send(TaskStarted(task_id="t-1", objective="first"), Context(stream=stream))
                await stream.send(TaskStarted(task_id="t-2", objective="second"), Context(stream=stream))

                assert [(t, k, p["channel_id"]) for t, k, p in fired] == [("t-2", "mirror_failed", channel.channel_id)]
            finally:
                await alice_hc.close()
                await bob_hc.close()

    async def test_observation_recorded_by_another_agent_does_not_suppress_the_owners(self) -> None:
        async with _serve() as (hub, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await mallory_hc.record_observation(
                    owner_id=mallory.agent_id, capability="payments", outcome=TaskState.COMPLETED, task_id="t-bob"
                )
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob", owner_id=bob.agent_id, spec=TaskSpec(title="t"), state=TaskState.RUNNING
                    )
                )

                await bob_hc.record_observation(
                    owner_id=bob.agent_id, capability="payments", outcome=TaskState.COMPLETED, task_id="t-bob"
                )

                assert (await hub.get_resume(bob.agent_id)).observed["payments"].completed == 1
            finally:
                await bob_hc.close()
                await mallory_hc.close()


@pytest.mark.asyncio
async def test_a_name_cannot_shadow_an_agent_id() -> None:
    async with _serve(allow_no_auth=True) as (hub, url):
        bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
        try:
            bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
            assert bob.agent_id is not None

            with pytest.raises(ProtocolError, match="agent id"):
                await mallory_hc.register(_agent("shadow"), Passport(name=bob.agent_id), Resume())

            assert (await mallory_hc.get_agent(bob.agent_id)).name == "bob"
        finally:
            await bob_hc.close()
            await mallory_hc.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        HelloFrame(name=["bob"]),  # type: ignore[arg-type]
        HelloFrame(name="bob", since_envelope_id=5),  # type: ignore[arg-type]
        ReceiptFrame(envelope_id="e-1", status="ack", recipient_id=["bob"], channel_id="c-1"),  # type: ignore[arg-type]
    ],
    ids=["hello-name-list", "hello-since-int", "receipt-recipient-list"],
)
async def test_malformed_frame_gets_an_error_and_keeps_the_connection(frame: object) -> None:
    async with _serve(allow_no_auth=True) as (hub, url):
        await hub.register_identity(Passport(name="bob"), Resume())
        link = WsLinkClient(url)
        try:
            await link.open()
            frames = aiter(link.frames())

            await link.send_frame(frame)  # type: ignore[arg-type]
            reply = await asyncio.wait_for(anext(frames), 2.0)
            await link.send_frame(PingFrame())
            pong = await asyncio.wait_for(anext(frames), 2.0)

            assert isinstance(reply, ErrorFrame)
            assert reply.code == "protocol_error"
            assert isinstance(pong, PongFrame)
        finally:
            await link.close()


@pytest.mark.asyncio
class TestResidualExposure:
    """Reads and writes a participant-free or channel-free caller used to reach."""

    async def test_task_is_visible_to_its_owner_and_channel_participants_only(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob",
                        owner_id=bob.agent_id,
                        spec=TaskSpec(title="secret"),
                        state=TaskState.RUNNING,
                        channel_id=channel.channel_id,
                    )
                )

                assert (await bob_hc.get_task("t-bob")).task_id == "t-bob"
                assert (await alice_hc.get_task("t-bob")).spec.title == "secret"
                assert [t.task_id for t in await alice_hc.list_tasks()] == ["t-bob"]
                with pytest.raises(NotFoundError):
                    await mallory_hc.get_task("t-bob")
                assert await mallory_hc.list_tasks() == []
                assert await mallory_hc.list_tasks(channel_id=channel.channel_id) == []
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_envelope_addressed_to_a_non_participant_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, carol_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                carol = await carol_hc.register(_agent("carol"), _passport("carol"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError):
                    await alice_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=alice.agent_id,
                            audience=[carol.agent_id],
                            event_type=EV_TEXT,
                            event_data={"text": "hi carol"},
                        )
                    )
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await carol_hc.close()

    async def test_turn_failure_for_a_channel_the_agent_is_not_in_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.report_turn_failure(
                        channel_id=channel.channel_id,
                        agent_id=mallory.agent_id,
                        envelope_id="e-1",
                        exc=RuntimeError("forged"),
                    )
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    async def test_peer_cancel_reason_over_the_length_limit_is_refused(self) -> None:
        hub, _, request = await _hub_with_live_task(MemoryKnowledgeStore())
        try:
            request.event_data["reason"] = "x" * 501
            with pytest.raises(ProtocolError):
                await hub.post_envelope(request)
        finally:
            await hub.close()

    async def test_mirror_failure_text_is_clipped(self) -> None:
        async with _serve() as (hub, url):
            alice_hc = HubClient(WsLink(url))
            fired: list[tuple[str, str, dict]] = []
            hub.register_listener(_TaskEvents(fired))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await alice_hc.fire_task_event(
                    "t-alice", "mirror_failed", {"owner_id": alice.agent_id, "exc_message": "x" * 5000}
                )

                assert fired == [("t-alice", "mirror_failed", {"owner_id": alice.agent_id, "exc_message": "x" * 500})]
            finally:
                await alice_hc.close()

    async def test_serving_no_auth_beyond_loopback_is_refused_unless_allowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
        try:
            with pytest.raises(ValueError, match="'none' auth scheme"):
                async with serve_ws(hub, "0.0.0.0", 0):
                    pass

            with caplog.at_level(logging.WARNING, logger="ag2.network.transport.ws"):
                async with serve_ws(hub, "0.0.0.0", 0, allow_unauthenticated=True):
                    pass
            assert "accepts the 'none' auth scheme" in caplog.text

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="ag2.network.transport.ws"):
                async with serve_ws(hub, "127.0.0.1", 0):
                    pass
            assert caplog.text == ""
        finally:
            await hub.close()
