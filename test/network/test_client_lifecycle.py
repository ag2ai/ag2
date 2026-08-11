# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Async-context-manager lifecycle for ``HubClient`` and ``AgentClient``.

Both classes implement ``__aenter__`` / ``__aexit__`` so an ``async
with`` block runs the right cleanup on exit:

* ``HubClient.__aexit__`` calls ``self.close()`` — cancels the link's
  receive task and closes the transport.
* ``AgentClient.__aexit__`` calls ``self.unregister()`` — round-trip
  to the hub to remove the agent from its registry, plus marks the
  client locally disconnected.

Cleanup must run on both clean exit and exception paths.
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
from ag2.network.errors import NotFoundError
from ag2.testing import TestConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


@pytest.mark.asyncio
class TestClientLifecycle:
    """async-with on HubClient and AgentClient runs the right cleanup."""

    async def test_hub_client_close_runs_on_exit(self) -> None:
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)

        async with HubClient(link, hub=hub) as hc:
            assert not hc._closed

        assert hc._closed

        await hub.close()

    async def test_agent_client_unregister_runs_on_exit(self) -> None:
        """On block exit, the agent is unregistered from the hub registry
        and marked locally disconnected."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)

        async with HubClient(link, hub=hub) as hc:
            ac = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
            agent_id = ac.agent_id
            assert not ac._disconnected

            async with ac:
                # Hub knows about alice while the block is open.
                passport = await hub.get_agent(agent_id)
                assert passport.name == "alice"

            # On block exit: unregister() called → marked disconnected
            # locally and removed from hub registry.
            assert ac._disconnected
            with pytest.raises(NotFoundError):
                await hub.get_agent(agent_id)

        await hub.close()

    async def test_cleanup_runs_on_exception(self) -> None:
        """If the body raises, both __aexit__ paths still run."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)

        boom = RuntimeError("boom")

        try:
            async with HubClient(link, hub=hub) as hc:
                ac = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
                agent_id = ac.agent_id
                async with ac:
                    raise boom
        except RuntimeError as caught:
            assert caught is boom

        # AgentClient.__aexit__ ran despite the raise: locally disconnected
        # and removed from the hub registry.
        assert ac._disconnected
        with pytest.raises(NotFoundError):
            await hub.get_agent(agent_id)

        # HubClient.__aexit__ also ran: transport closed.
        assert hc._closed

        await hub.close()
