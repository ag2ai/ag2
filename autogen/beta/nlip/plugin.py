# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``NLIPPlugin`` — attach NLIP protocol support to a beta Agent via the Plugin API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent import Plugin
from ..middleware import HistoryLimiter

if TYPE_CHECKING:
    from ..agent import Agent
    from .app import BetaNlipApplication

__all__ = ("NLIPPlugin",)


class NLIPPlugin(Plugin):
    """Expose a beta Agent over the NLIP (Natural Language Interaction Protocol).

    Registering this plugin with an agent records a reference to that
    agent so that :meth:`build_asgi` can be called directly on the plugin
    without passing the agent each time.

    Args:
        history_limit: Optional maximum number of conversation events
            the LLM sees per turn.  When set, a
            :class:`~autogen.beta.middleware.HistoryLimiter` is added to
            the agent's middleware stack automatically.  ``None`` (the
            default) leaves history unbounded.

    Example::

        import uvicorn
        from autogen.beta import Agent
        from autogen.beta.nlip import NLIPPlugin

        plugin = NLIPPlugin(history_limit=40)
        agent = Agent("assistant", plugins=[plugin])

        # Serve the agent:
        uvicorn.run(plugin.build_asgi(), host="0.0.0.0", port=8000)
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

    def build_asgi(self) -> BetaNlipApplication:
        """Build and return the ASGI-callable NLIP application for the registered agent.

        Returns:
            A :class:`BetaNlipApplication` instance (both ``NLIP_Application``
            and ASGI callable).

        Raises:
            RuntimeError: If the plugin has not been registered with an agent yet.
        """
        if self._agent is None:
            raise RuntimeError(
                "NLIPPlugin has not been registered with an agent. "
                "Pass this plugin to Agent(..., plugins=[plugin]) before calling .build_asgi()."
            )
        from .app import BetaNlipApplication

        return BetaNlipApplication(self._agent)
