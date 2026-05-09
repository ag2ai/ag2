# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``NetworkPlugin`` — attaches the network tool surface to an ``Agent``.

* Adds the network tools (``say``, ``delegate``, ``peers``,
  ``sessions``, ``tasks``, ``context``) to ``agent.tools``.
* Appends ``NetworkContextPolicy`` to the agent's assembly chain so
  every LLM call sees a "you are <name>" prefix plus the available
  tool names.

Plugins are first-class in beta (``autogen/beta/agent.py`` ``Plugin``
class). The network plugin uses the existing slot.
"""

from typing import TYPE_CHECKING

from autogen.beta.agent import Plugin
from autogen.beta.assembly import AssemblyPolicy
from autogen.beta.events import BaseEvent

from ..transitions import AgentTarget, ToolCalled, TransitionGraph
from .tools import (
    make_context_tool,
    make_delegate_tool,
    make_handoff_tools,
    make_peers_tool,
    make_say_tool,
    make_sessions_tool,
    make_tasks_tool,
)

if TYPE_CHECKING:
    from autogen.beta.agent import Agent
    from autogen.beta.context import ConversationContext as Context

    from .agent_client import AgentClient

__all__ = ("NetworkContextPolicy", "NetworkPlugin")


class NetworkContextPolicy:
    """Assembly policy: prepends a network-aware prefix to every LLM call.

    Names the agent and lists its network tools.
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
        prefix = (
            f"You are {self._client.passport.name} "
            f"(agent_id: {self._client.agent_id}).\n"
            "Network tools: say, delegate, peers, sessions, tasks, context."
        )
        return [prefix, *prompts], events


class NetworkPlugin(Plugin):
    """Attaches an Agent to a network.

    Adds ``say`` and ``delegate`` to ``agent.tools`` so the LLM sees
    them on every turn — the verbs are stable for the life of the
    registration. Also appends ``NetworkContextPolicy`` to the agent's
    assembly chain.
    """

    def __init__(self, client: "AgentClient") -> None:
        super().__init__(
            tools=[
                make_say_tool(client),
                make_delegate_tool(client),
                make_peers_tool(client),
                make_sessions_tool(client),
                make_tasks_tool(client),
                make_context_tool(client),
            ],
        )
        self._client = client

    def register(self, agent: "Agent") -> None:
        """Wire tools + assembly policy onto the agent. Idempotent-ish.

        Calling ``register`` more than once on the same agent will add
        the tools / policies again. ``HubClient.register`` only attaches
        once per ``(Agent, identity)``, so this is rare in practice.
        """
        super().register(agent)
        agent.add_policy(NetworkContextPolicy(self._client))

    def register_workflow(self, graph: TransitionGraph) -> list[object]:
        """Materialise one LLM tool per ``ToolCalled → AgentTarget``
        transition in ``graph`` and append them to the bound agent's
        ``tools`` list. Returns the new tool objects so callers can
        later remove them if the workflow ends and the surface should
        be trimmed.

        Each tool's body returns a typed
        :class:`autogen.beta.network.handoff.Handoff` carrying the
        target agent's name. The workflow adapter reads the Handoff
        from the agent's local ``ToolResultEvent`` stream and routes
        the next speaker accordingly — no matching ``ToolCalled``
        graph rule fires (the Handoff carries the routing intent).

        ``ToolCalled`` transitions whose ``then`` is *not* an
        :class:`AgentTarget` (e.g. ``TerminateTarget``,
        ``StayTarget``, ``RevertToInitiatorTarget``,
        ``RoundRobinTarget``) are skipped — those targets cannot be
        encoded as a typed ``Handoff``. Users wanting tool-driven
        routing into those targets must hand-write their own tool
        bodies and rely on the graph's ``ToolCalled`` rule.

        Tools are scoped **per-agent, not per-session**. The same
        tool object is visible on every session the agent is part of.

        ⚠️  **Cross-workflow footgun.** If the same agent registers
        workflows A and B, tool ``foo`` from A is *also* visible while
        the agent is taking its turn in B. If the LLM invokes ``foo``
        during B's turn, the Handoff target may not be a participant
        of B and routing will fall back to B's ``default_target`` —
        which is commonly ``TerminateTarget``, prematurely closing B.
        Mitigations:

        * Use distinct, namespaced tool names across workflows
          (e.g. ``triage_to_eng`` vs ``billing_to_eng``).
        * Register only one workflow per agent at a time, calling
          ``register_workflow`` lazily as the agent enters each one
          and removing the tool objects when leaving.
        * Pick a non-terminating ``default_target`` (e.g.
          ``StayTarget()``) on graphs the agent might be running
          alongside other workflows.
        """
        mapping: dict[str, str] = {}
        for transition in graph.transitions:
            when = transition.when
            then = transition.then
            if not isinstance(when, ToolCalled):
                continue
            if not isinstance(then, AgentTarget):
                continue
            mapping.setdefault(when.tool_name, then.agent_id)

        new_tools = make_handoff_tools(mapping)
        existing = {t.name for t in self._client.agent.tools}
        attached: list[object] = []
        for t in new_tools:
            if t.name in existing:
                continue
            self._client.agent.tools.append(t)
            attached.append(t)
        return attached


# Make ``NetworkPlugin`` satisfy ``AssemblyPolicy`` indirectly via its
# context-policy member. The Protocol is structural; ``NetworkContextPolicy``
# implements ``apply`` correctly so the implicit assertion holds.
_: AssemblyPolicy
