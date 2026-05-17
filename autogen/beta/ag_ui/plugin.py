# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``A2UIPlugin`` — attach AG-UI protocol support to an Agent via the Plugin API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autogen.beta.agent import Plugin
from autogen.beta.middleware import HistoryLimiter

from .stream import AGUIStream

if TYPE_CHECKING:
    from autogen.beta.agent import Agent

__all__ = ("A2UIPlugin",)


class A2UIPlugin(Plugin):
    """Expose an Agent over the AG-UI protocol.

    Registering this plugin with an agent records a reference to that
    agent so that :meth:`stream` and :meth:`build_asgi` can be called
    directly on the plugin without passing the agent each time.

    Args:
        history_limit: Optional maximum number of conversation events
            the LLM sees per turn.  When set, a
            :class:`~autogen.beta.middleware.HistoryLimiter` is added to
            the agent's middleware stack automatically.  ``None`` (the
            default) leaves history unbounded.

    Example::

        from autogen.beta import Agent
        from autogen.beta.ag_ui import A2UIPlugin

        plugin = A2UIPlugin(history_limit=40)
        agent = Agent("assistant", plugins=[plugin])

        # Serve the agent:
        asgi_endpoint = plugin.build_asgi()

        # Or stream events directly:
        async for chunk in plugin.stream.dispatch(run_input):
            ...
    """

    def __init__(self, *, history_limit: int | None = None) -> None:
        middleware = [HistoryLimiter(history_limit)] if history_limit is not None else []
        super().__init__(middleware=middleware)
        self._history_limit = history_limit
        self._agent: Agent | None = None

    def register(self, agent: Agent[Any]) -> None:
        """Apply middleware to the agent and record the reference."""
        super().register(agent)
        self._agent = agent

    @property
    def stream(self) -> AGUIStream:
        """Return an :class:`~autogen.beta.ag_ui.AGUIStream` bound to the registered agent.

        Raises:
            RuntimeError: If the plugin has not been registered with an agent yet.
        """
        if self._agent is None:
            raise RuntimeError(
                "A2UIPlugin has not been registered with an agent. "
                "Pass this plugin to Agent(..., plugins=[plugin]) before calling .stream."
            )
        return AGUIStream(self._agent)

    def build_asgi(self) -> Any:
        """Build and return the ASGI endpoint class for the registered agent.

        Shortcut for ``plugin.stream.build_asgi()``.

        Returns:
            An ASGI-compatible ``HTTPEndpoint`` class (Starlette).

        Raises:
            RuntimeError: If the plugin has not been registered with an agent yet.
        """
        return self.stream.build_asgi()
