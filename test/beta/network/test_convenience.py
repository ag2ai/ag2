# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.network.convenience import Network
from autogen.beta.network.hub import RegistrationHandle


class _MockAgent:
    """Minimal mock agent for Network tests."""

    def __init__(self, name: str):
        self.name = name


class TestNetworkContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Network should support async with for start/stop."""
        network = Network()
        await network.register(_MockAgent("a"))

        async with network:
            assert "a" in network.hub.agents

    @pytest.mark.asyncio
    async def test_aexit_cleans_up_hub(self) -> None:
        """Network.__aexit__ should call hub.close() (plugins + channel)."""
        from autogen.beta.network.topology import BasePlugin, Pipeline

        class _TrackingPlugin(BasePlugin):
            def __init__(self):
                self.uninstalled = False

            def uninstall(self):
                self.uninstalled = True

        plugin = _TrackingPlugin()
        network = Network(topology=Pipeline(plugin))
        await network.register(_MockAgent("a"))

        async with network:
            assert "a" in network.hub.agents

        # After exiting context, plugin should be uninstalled
        assert plugin.uninstalled

    @pytest.mark.asyncio
    async def test_register_returns_handle(self) -> None:
        network = Network()
        handle = await network.register(_MockAgent("a"), capabilities=["x"])
        assert isinstance(handle, RegistrationHandle)
        assert handle.name == "a"

    @pytest.mark.asyncio
    async def test_handle_unregister(self) -> None:
        network = Network()
        handle = await network.register(_MockAgent("a"), capabilities=["x"])
        await handle.unregister()
        assert "a" not in network.hub.agents
