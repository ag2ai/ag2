# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``NetworkPlugin`` — attaches the network tool surface to an ``Agent``.

Two streams of LLM tools compose into an agent's tool list per turn:

* **Identity-level** (attached by ``NetworkPlugin`` once at registration,
  always available): ``peers`` / ``channels`` / ``tasks`` / ``context`` /
  ``delegate``. Cross-cutting verbs that work in any channel — discovery,
  channel lifecycle, task observation, and the one-shot ``delegate``
  convenience that opens its own consulting channel.

* **Channel-level** (resolved per turn by the default notify handler
  via ``adapter.tools_for(...)``): ``say`` for adapters that accept
  free-form text (consulting / conversation / discussion), user-authored
  handoff tools for workflow. Adapter-provided tools merge into the
  ``tools=`` override passed to ``agent.ask``.

The plugin appends ``NetworkContextPolicy`` to the agent's assembly
chain so every LLM call sees a "you are <name>" prefix.
"""

from typing import TYPE_CHECKING

from autogen.beta.agent import Plugin
from autogen.beta.events import BaseEvent

from .tools import (
    make_channels_tool,
    make_context_tool,
    make_delegate_tool,
    make_peers_tool,
    make_tasks_tool,
)

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext as Context

    from .agent_client import AgentClient

__all__ = ("NetworkContextPolicy", "NetworkPlugin")


class NetworkContextPolicy:
    """Assembly policy: prepends a network-aware prefix to every LLM call.

    Names the agent. The per-turn tool list is dynamic (identity-level
    set + adapter-provided set merged by the handler) so the prefix
    doesn't enumerate tools — the LLM sees them in its tools array.
    """

    name = "network_context"

    def __init__(self, client: "AgentClient") -> None:
        # __init__ stores params; no side effects.
        self._client = client

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: "Context",
    ) -> tuple[list[str], list[BaseEvent]]:
        prefix = f"You are {self._client.passport.name} (agent_id: {self._client.agent_id})."
        return [prefix, *prompts], events


class NetworkPlugin(Plugin):
    """Attaches an Agent to a network.

    Adds the identity-level cross-cutting tools to ``agent.tools``:
    ``peers`` / ``channels`` / ``tasks`` / ``context`` / ``delegate``.
    These are stable for the life of the registration and work in any
    channel context.

    Channel-level tools (``say``, workflow handoff tools) are NOT
    attached here — they come from ``adapter.tools_for(...)``, resolved
    per turn by the default notify handler and merged into
    ``agent.ask(tools=...)``. This keeps protocol-specific verbs out of
    an agent's tool list when they don't apply (e.g. ``say`` is hidden
    from workflow participants who must use handoff tools).
    """

    def __init__(self, client: "AgentClient") -> None:
        super().__init__(
            tools=[
                make_delegate_tool(client),
                make_peers_tool(client),
                make_channels_tool(client),
                make_tasks_tool(client),
                make_context_tool(client),
            ],
        )
        self._client = client
        self.add_policy(NetworkContextPolicy(client))
