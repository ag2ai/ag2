# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.middleware import HistoryLimiter
from autogen.beta.nlip import NLIPPlugin

try:
    from nlip_server.server import NLIP_Application

    nlip_server = True
except ImportError:
    nlip_server = False


class TestNLIPPlugin:
    def test_plugin_stores_history_limit(self) -> None:
        plugin = NLIPPlugin(history_limit=20)
        assert plugin._history_limit == 20

    def test_plugin_no_history_limit(self) -> None:
        plugin = NLIPPlugin()
        assert plugin._history_limit is None

    def test_plugin_adds_history_limiter_middleware(self) -> None:
        plugin = NLIPPlugin(history_limit=15)
        assert any(isinstance(m, HistoryLimiter) for m in plugin._middleware)

    def test_plugin_no_middleware_without_limit(self) -> None:
        plugin = NLIPPlugin()
        assert not any(isinstance(m, HistoryLimiter) for m in plugin._middleware)

    def test_build_asgi_raises_before_registration(self) -> None:
        plugin = NLIPPlugin()
        with pytest.raises(RuntimeError, match="not been registered"):
            plugin.build_asgi()

    def test_register_stores_agent(self) -> None:
        plugin = NLIPPlugin()
        agent = Agent("test_agent")
        plugin.register(agent)
        assert plugin._agent is agent

    def test_register_via_agent_constructor(self) -> None:
        plugin = NLIPPlugin()
        agent = Agent("test_agent", plugins=[plugin])
        assert plugin._agent is agent

    def test_history_limiter_wired_to_agent(self) -> None:
        plugin = NLIPPlugin(history_limit=10)
        agent = Agent("test_agent", plugins=[plugin])
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "HistoryLimiter" in middleware_types

    @pytest.mark.skipif(not nlip_server, reason="nlip-server not installed")
    def test_build_asgi_returns_nlip_application(self) -> None:
        plugin = NLIPPlugin()
        Agent("test_agent", plugins=[plugin])
        app = plugin.build_asgi()
        assert isinstance(app, NLIP_Application)

    @pytest.mark.skipif(not nlip_server, reason="nlip-server not installed")
    def test_build_asgi_is_asgi_callable(self) -> None:
        plugin = NLIPPlugin()
        Agent("test_agent", plugins=[plugin])
        app = plugin.build_asgi()
        assert callable(app)

    @pytest.mark.skipif(not nlip_server, reason="nlip-server not installed")
    def test_each_build_asgi_call_produces_new_application(self) -> None:
        plugin = NLIPPlugin()
        Agent("test_agent", plugins=[plugin])
        app1 = plugin.build_asgi()
        app2 = plugin.build_asgi()
        # Each call creates a fresh BetaNlipApplication wrapping the same agent.
        assert app1 is not app2
        assert isinstance(app1, NLIP_Application)
        assert isinstance(app2, NLIP_Application)
