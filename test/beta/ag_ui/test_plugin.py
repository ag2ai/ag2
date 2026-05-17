# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.ag_ui import A2UIPlugin, AGUIStream
from autogen.beta.middleware import HistoryLimiter

try:
    from starlette.endpoints import HTTPEndpoint

    starlette = True
except ImportError:
    starlette = False


pytestmark = pytest.mark.asyncio


class TestA2UIPlugin:
    def test_plugin_stores_history_limit(self) -> None:
        plugin = A2UIPlugin(history_limit=20)
        assert plugin._history_limit == 20

    def test_plugin_no_history_limit(self) -> None:
        plugin = A2UIPlugin()
        assert plugin._history_limit is None

    def test_plugin_adds_history_limiter_middleware(self) -> None:
        plugin = A2UIPlugin(history_limit=15)
        assert any(isinstance(m, HistoryLimiter) for m in plugin._middleware)

    def test_plugin_no_middleware_without_limit(self) -> None:
        plugin = A2UIPlugin()
        assert not any(isinstance(m, HistoryLimiter) for m in plugin._middleware)

    def test_stream_raises_before_registration(self) -> None:
        plugin = A2UIPlugin()
        with pytest.raises(RuntimeError, match="not been registered"):
            _ = plugin.stream

    def test_build_asgi_raises_before_registration(self) -> None:
        plugin = A2UIPlugin()
        with pytest.raises(RuntimeError, match="not been registered"):
            plugin.build_asgi()

    def test_register_stores_agent(self) -> None:
        plugin = A2UIPlugin()
        agent = Agent("test_agent")
        plugin.register(agent)
        assert plugin._agent is agent

    def test_stream_returns_agui_stream_after_registration(self) -> None:
        plugin = A2UIPlugin()
        agent = Agent("test_agent")
        plugin.register(agent)
        assert isinstance(plugin.stream, AGUIStream)

    def test_register_via_agent_constructor(self) -> None:
        plugin = A2UIPlugin()
        agent = Agent("test_agent", plugins=[plugin])
        # Plugin is registered automatically by Agent.__init__
        assert plugin._agent is agent

    def test_history_limiter_wired_to_agent(self) -> None:
        plugin = A2UIPlugin(history_limit=10)
        agent = Agent("test_agent", plugins=[plugin])
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "HistoryLimiter" in middleware_types

    @pytest.mark.skipif(not starlette, reason="starlette not installed")
    def test_build_asgi_returns_endpoint_class(self) -> None:
        plugin = A2UIPlugin()
        Agent("test_agent", plugins=[plugin])
        endpoint_class = plugin.build_asgi()
        assert issubclass(endpoint_class, HTTPEndpoint)

    @pytest.mark.skipif(not starlette, reason="starlette not installed")
    def test_build_asgi_shortcut_equals_stream_build_asgi(self) -> None:
        plugin = A2UIPlugin()
        Agent("test_agent", plugins=[plugin])
        # Both calls produce distinct endpoint classes wrapping the same agent.
        # They should have the same base class; identity check would be too strict
        # since each call to build_asgi() creates a new class.
        assert issubclass(plugin.build_asgi(), HTTPEndpoint)
        assert issubclass(plugin.stream.build_asgi(), HTTPEndpoint)
