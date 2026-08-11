# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Hub.register(agent)`` convenience — hub-owned client lifecycle.

``Hub.register`` lifts ``HubClient.register`` onto the hub: each call mints
a dedicated ``HubClient`` (on a default ``LocalLink`` or a supplied
``link``) and returns the ``AgentClient`` handle. The handle owns that
connection — ``AgentClient.close()`` unregisters the agent and closes it,
and ``Hub.close()`` closes any handles left open. ``Hub.register_identity``
remains the low-level identity primitive underneath, and the explicit
``HubClient(link, hub=hub)`` flow (where one connection may host many
agents) is untouched.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.adapters.conversation import CONVERSATION_TYPE
from ag2.network.channel import ChannelState
from ag2.network.errors import NotFoundError
from ag2.testing import TestConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


@pytest.mark.asyncio
class TestHubRegisterAgent:
    async def test_returns_agent_client_registered_in_hub(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))

        # Discoverable in the hub registry under the agent's name.
        passport = await hub.get_agent("alice")
        assert passport.agent_id == alice.agent_id
        assert passport.name == "alice"

        await hub.close()

    async def test_defaults_passport_and_resume_from_agent(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        # No passport/resume passed — derived from agent.name.
        alice = await hub.register(_agent("alice"))
        assert alice.passport.name == "alice"

        await hub.close()

    async def test_two_agents_get_distinct_clients_and_converse(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        # Each call mints its own hub-owned HubClient.
        assert alice._hub_client is not bob._hub_client

        # A conversation between them still reaches ACTIVE.
        channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
        assert channel.state == ChannelState.ACTIVE

        await hub.close()

    async def test_hub_close_closes_owned_client(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        client = alice._hub_client
        assert not client._closed

        await hub.close()

        # The caller never saw the HubClient; hub.close() drained it.
        assert client._closed

    async def test_agent_close_unregisters_and_closes_its_client(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        client = alice._hub_client

        await alice.close()

        # Agent removed from the registry and its dedicated client closed.
        with pytest.raises(NotFoundError):
            await hub.get_agent(alice.agent_id)
        assert client._closed

        await hub.close()

    async def test_agent_close_isolates_siblings(self) -> None:
        """Closing one agent leaves another's connection untouched."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        await alice.close()

        # bob's connection is unaffected — it can still open a channel.
        await hub.register(_agent("carol"))
        channel = await bob.open(type=CONVERSATION_TYPE, target="carol")
        assert channel.state == ChannelState.ACTIVE
        assert not bob._hub_client._closed

        await hub.close()

    async def test_agent_close_is_idempotent(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))

        await alice.close()
        await alice.close()  # second call is a no-op, not an error

        await hub.close()

    async def test_hub_close_skips_already_closed_clients(self) -> None:
        """hub.close() closes whatever the caller left open; closed ones no-op."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        await alice.close()  # alice's client closed early
        assert alice._hub_client._closed
        assert not bob._hub_client._closed

        await hub.close()  # closes bob's, no-ops on alice's

        assert bob._hub_client._closed

    async def test_explicit_link_builds_separate_tracked_client(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        a = await hub.register(_agent("alice"))
        b = await hub.register(_agent("bob"), link=LocalLink(hub))

        # Distinct clients; both tracked and drained on hub.close().
        assert a._hub_client is not b._hub_client

        await hub.close()
        assert a._hub_client._closed
        assert b._hub_client._closed

    async def test_rejects_human_passport_without_leaking_client(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        with pytest.raises(ValueError, match="register_human"):
            await hub.register(_agent("h"), Passport(name="h", kind="human"))

        # The failed registration's client was closed + untracked, not leaked.
        assert hub._owned_clients == []

        await hub.close()


@pytest.mark.asyncio
class TestExplicitFlowClose:
    """In the explicit HubClient flow the connection is shared/caller-owned,
    so AgentClient.close() only unregisters — it must not close the transport."""

    async def test_close_on_shared_client_only_unregisters(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        hc = HubClient(LocalLink(hub), hub=hub)

        # Two agents on ONE caller-owned connection.
        alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

        await alice.close()

        # alice is unregistered, but the shared connection stays up for bob.
        with pytest.raises(NotFoundError):
            await hub.get_agent(alice.agent_id)
        assert not hc._closed
        assert (await hub.get_agent(bob.agent_id)).name == "bob"

        await hc.close()
        await hub.close()


@pytest.mark.asyncio
class TestRegisterIdentity:
    """The renamed low-level identity primitive still stamps + persists."""

    async def test_low_level_identity_stamps_passport(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)

        stamped = await hub.register_identity(Passport(name="echo"), Resume())

        assert stamped.agent_id is not None
        assert stamped.name == "echo"
        # Persisted in the registry, discoverable by name.
        assert (await hub.get_agent("echo")).agent_id == stamped.agent_id

        await hub.close()
