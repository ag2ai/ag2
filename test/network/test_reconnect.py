# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Reconnect-by-name: ``HubClient.attach``, ``Hub.pending_turns_for``,
and ``AgentClient.resume_pending_turns``.

``attach(name=...)`` is the reconnect-aware companion to ``register``.
If the named identity exists, the existing ``agent_id`` is re-bound to
the new connection's endpoint and the prior endpoint mapping is
evicted (the endpoint stays alive for any other agents on it).
``pending_turns_for(agent_id)`` walks active channels asking each
adapter who is expected next; ``resume_pending_turns`` re-fires the
notify handler against each returned turn's triggering envelope.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    PendingTurn,
    Resume,
)

from ._helpers import ScriptedConfig, wait_for_text_count


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig(*replies))


async def _new_hub() -> Hub:
    return await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )


class TestAttachRegisterFallback:
    @pytest.mark.asyncio
    async def test_attach_falls_back_to_register_when_name_is_new(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)

        try:
            client = await hc.attach(
                _agent("alice"),
                "alice",
                passport=Passport(name="alice"),
                resume=Resume(claimed_capabilities=["debate"]),
            )
            assert client.agent_id is not None
            assert hub._name_to_id["alice"] == client.agent_id
            # The endpoint binding goes through register's path.
            assert hub._agent_to_endpoint[client.agent_id] == hc._client_link.endpoint_id
            # Resume captured.
            assert "debate" in hub._resumes[client.agent_id].claimed_capabilities
        finally:
            await hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_attach_without_passport_when_name_is_new_raises(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)
        try:
            with pytest.raises(ValueError, match="not registered"):
                await hc.attach(_agent("ghost"), "ghost")
        finally:
            await hc.close()
            await hub.close()


class TestAttachRebind:
    @pytest.mark.asyncio
    async def test_attach_reuses_existing_agent_id(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        first_hc = HubClient(link, hub=hub)
        bob_first = await first_hc.register(_agent("bob"), Passport(name="bob"), Resume())
        original_agent_id = bob_first.agent_id
        await first_hc.close()

        second_hc = HubClient(link, hub=hub)
        try:
            bob_second = await second_hc.attach(_agent("bob"), "bob")
            assert bob_second.agent_id == original_agent_id
            # Persisted passport/resume came back from the hub, not from
            # an attach-time argument.
            assert bob_second.passport.name == "bob"
        finally:
            await second_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_attach_evicts_prior_endpoint_binding(self) -> None:
        """The hub stops routing notifies to the previously-bound
        endpoint but keeps that endpoint alive — useful when several
        agents shared the prior connection."""
        hub = await _new_hub()
        link = LocalLink(hub)

        alice_hc = HubClient(link, hub=hub)
        bob_hc_v1 = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob_v1 = await bob_hc_v1.register(_agent("bob"), Passport(name="bob"), Resume())
        bob_id = bob_v1.agent_id
        v1_endpoint_id = bob_hc_v1._client_link.endpoint_id
        assert hub._agent_to_endpoint[bob_id] == v1_endpoint_id

        try:
            # Open a channel so there's something to dispatch later.
            channel = await alice.open(type="discussion", target=["bob"])

            # Bob re-attaches via a different HubClient — same name.
            bob_hc_v2 = HubClient(link, hub=hub)
            bob_v2 = await bob_hc_v2.attach(_agent("bob"), "bob")
            v2_endpoint_id = bob_hc_v2._client_link.endpoint_id

            try:
                # The hub now routes bob's notifies to v2.
                assert hub._agent_to_endpoint[bob_id] == v2_endpoint_id
                # v1 endpoint is still attached to the hub (other agents
                # could be on it) but no longer maps to bob.
                assert v1_endpoint_id in hub._endpoints_by_id
                assert bob_id not in hub._endpoint_to_agents.get(v1_endpoint_id, set())

                # Send a message; bob_v2's handler should receive it.
                received_v2: list[Envelope] = []
                orig = bob_v2._on_envelope

                async def cap(env):
                    received_v2.append(env)
                    if orig is not None:
                        await orig(env)

                bob_v2.on_envelope(cap)
                await channel.send("after-reattach")
                await wait_for_text_count(hub, channel.channel_id, 1)
                await asyncio.sleep(0.05)

                assert any(
                    e.event_type == EV_TEXT and e.event_data.get("text") == "after-reattach" for e in received_v2
                )
            finally:
                await bob_hc_v2.close()
        finally:
            await bob_hc_v1.close()
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_attach_does_not_close_shared_endpoint(self) -> None:
        """When multiple agents share an endpoint, attaching one of
        them via a different connection must not break the others."""
        hub = await _new_hub()
        link = LocalLink(hub)

        shared_hc = HubClient(link, hub=hub)
        await shared_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await shared_hc.register(_agent("bob"), Passport(name="bob"), Resume())
        shared_endpoint_id = shared_hc._client_link.endpoint_id
        # Both bound to the same endpoint.
        assert hub._endpoint_to_agents[shared_endpoint_id] >= {bob.agent_id}

        try:
            other_hc = HubClient(link, hub=hub)
            try:
                await other_hc.attach(_agent("bob"), "bob")
                # The shared endpoint stays attached (alice still uses it).
                assert shared_endpoint_id in hub._endpoints_by_id
                # Alice's binding survives.
                alice_id = hub._name_to_id["alice"]
                assert hub._agent_to_endpoint[alice_id] == shared_endpoint_id
            finally:
                await other_hc.close()
        finally:
            await shared_hc.close()
            await hub.close()


class TestPendingTurnsFor:
    @pytest.mark.asyncio
    async def test_returns_turn_in_discussion_when_agent_is_expected(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("alice speaks")
            await wait_for_text_count(hub, channel.channel_id, 1)

            wal = await hub.read_wal(channel.channel_id)
            text_env = next(e for e in wal if e.event_type == EV_TEXT)

            pending = await hub.pending_turns_for(bob.agent_id)
            assert len(pending) == 1
            turn = pending[0]
            assert isinstance(turn, PendingTurn)
            assert turn.channel_id == channel.channel_id
            assert turn.triggering_envelope_id == text_env.envelope_id
            assert turn.expected_at == text_env.created_at
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_returns_empty_when_agent_not_expected(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("alice speaks")
            await wait_for_text_count(hub, channel.channel_id, 1)

            # Bob is expected next; alice is not.
            assert await hub.pending_turns_for(alice.agent_id) == []
            # Sanity: bob has the pending turn.
            assert len(await hub.pending_turns_for(bob.agent_id)) == 1
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_returns_empty_in_free_form_channel(self) -> None:
        """Conversation has no turn ordering — pending_turns should
        always be empty regardless of who last spoke."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="conversation", target="bob")
            await channel.send("alice speaks")
            await wait_for_text_count(hub, channel.channel_id, 1)

            assert await hub.pending_turns_for(bob.agent_id) == []
            assert await hub.pending_turns_for(alice.agent_id) == []
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_returns_empty_for_unregistered_agent(self) -> None:
        hub = await _new_hub()
        try:
            assert await hub.pending_turns_for("does-not-exist") == []
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_returns_creator_with_no_trigger_on_fresh_channel(self) -> None:
        """A discussion just opened expects its creator to speak first
        with no triggering envelope yet."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            await alice.open(type="discussion", target=["bob"])
            pending = await hub.pending_turns_for(alice.agent_id)
            assert len(pending) == 1
            assert pending[0].triggering_envelope_id is None
            # expected_at falls back to the hub clock (no trigger envelope).
            assert pending[0].expected_at != ""
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()


class TestResumePendingTurns:
    @pytest.mark.asyncio
    async def test_resume_refires_handler_on_triggering_envelope(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("ask bob")
            await wait_for_text_count(hub, channel.channel_id, 1)
            await asyncio.sleep(0.05)

            wal = await hub.read_wal(channel.channel_id)
            ask = next(e for e in wal if e.event_type == EV_TEXT)

            captured: list[Envelope] = []
            orig = bob._on_envelope

            async def cap(env):
                captured.append(env)
                if orig is not None:
                    await orig(env)

            bob.on_envelope(cap)

            n = await bob.resume_pending_turns()
            assert n == 1
            assert any(e.envelope_id == ask.envelope_id for e in captured)
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_resume_returns_zero_when_no_pending_turns(self) -> None:
        hub = await _new_hub()
        link = LocalLink(hub)
        bob_hc = HubClient(link, hub=hub)
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
        try:
            assert await bob.resume_pending_turns() == 0
        finally:
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_resume_skips_turns_with_no_trigger(self) -> None:
        """A freshly opened channel's expected turn has no triggering
        envelope; resume_pending_turns must skip it."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            await alice.open(type="discussion", target=["bob"])
            # Alice is the expected creator but no trigger envelope.
            assert await alice.resume_pending_turns() == 0
        finally:
            await alice_hc.close()
            await bob_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_reconnect_cycle_attach_and_resume(self) -> None:
        """End-to-end: bob disconnects mid-turn, reattaches via a fresh
        HubClient, then resume_pending_turns re-fires his handler against
        the unanswered envelope."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc_v1 = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob_v1 = await bob_hc_v1.register(_agent("bob"), Passport(name="bob"), Resume())

        try:
            channel = await alice.open(type="discussion", target=["bob"])
            await channel.send("ask bob")
            await wait_for_text_count(hub, channel.channel_id, 1)
            await asyncio.sleep(0.05)

            wal = await hub.read_wal(channel.channel_id)
            ask = next(e for e in wal if e.event_type == EV_TEXT)
            bob_id = bob_v1.agent_id

            # bob crashes — close his connection.
            await bob_hc_v1.close()

            # Reconnect via attach. Same name, new connection.
            bob_hc_v2 = HubClient(link, hub=hub)
            try:
                bob_v2 = await bob_hc_v2.attach(_agent("bob"), "bob")
                assert bob_v2.agent_id == bob_id

                captured: list[Envelope] = []
                orig = bob_v2._on_envelope

                async def cap(env):
                    captured.append(env)
                    if orig is not None:
                        await orig(env)

                bob_v2.on_envelope(cap)

                n = await bob_v2.resume_pending_turns()
                assert n == 1
                assert any(e.envelope_id == ask.envelope_id for e in captured)
            finally:
                await bob_hc_v2.close()
        finally:
            await alice_hc.close()
            await hub.close()
