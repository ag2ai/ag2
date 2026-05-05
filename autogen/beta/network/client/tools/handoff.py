# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow handoff tools.

Each :class:`ToolCalled` transition in a workflow's :class:`TransitionGraph`
becomes one LLM tool. When the LLM invokes the tool, it posts an
``ag2.handoff`` envelope into the active session; the
:class:`WorkflowAdapter` reads it in ``fold`` and advances
``expected_next_speaker`` per the matching transition.

The LLM never sees ``expected_next_speaker`` directly — it sees a
button labelled with the tool name and a one-line description. The
protocol does the rest. **Handoffs are a UX over the choreography.**

Usage:
    from autogen.beta.network.client.tools.handoff import (
        make_handoff_tools_for_graph,
    )

    graph = TransitionGraph(...)
    handoff_tools = make_handoff_tools_for_graph(client, graph)
    for t in handoff_tools:
        client.agent.tools.append(t)

Or via the convenience method on :class:`NetworkPlugin`:
    plugin.register_workflow(graph)
"""

from typing import TYPE_CHECKING, Any

from autogen.beta.tools import tool

from ...envelope import EV_HANDOFF
from ...transitions import ToolCalled, TransitionGraph
from ..inject import SessionInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient


__all__ = ("make_handoff_tool", "make_handoff_tools_for_graph")


def make_handoff_tool(
    agent_client: "AgentClient",
    tool_name: str,
    description: str | None = None,
) -> Any:
    """Build one handoff tool that posts ``EV_HANDOFF`` into the
    current session.

    The LLM calls ``<tool_name>(reason="...")``; the tool emits an
    ``ag2.handoff`` envelope tagged with ``tool_name``. The active
    workflow adapter's ``ToolCalled`` condition matches and routes
    the next turn accordingly.
    """
    desc = description or f"Hand off the workflow via the {tool_name!r} transition."

    @tool(name=tool_name, description=desc)
    async def _handoff(
        reason: str = "",
        session: SessionInject = None,
    ) -> str:
        if session is None:
            return f"Error: {tool_name!r} requires an active session"
        try:
            await session.send(
                content=f"[handoff] {reason}" if reason else "[handoff]",
                event_type=EV_HANDOFF,
                event_data={"tool": tool_name, "reason": reason},
            )
        except Exception as exc:
            return f"Error: {tool_name!r} failed to post handoff: {exc}"
        return f"handoff posted via {tool_name}"

    return _handoff


def make_handoff_tools_for_graph(
    agent_client: "AgentClient",
    graph: TransitionGraph,
) -> list[Any]:
    """Materialise one handoff tool per unique ``ToolCalled`` condition
    in ``graph.transitions``.

    Tool names match ``ToolCalled.tool_name``. Duplicates (same tool
    name appearing in multiple transitions, e.g. with different
    priorities) are de-duplicated.
    """
    seen: set[str] = set()
    tools: list[Any] = []
    for transition in graph.transitions:
        when = transition.when
        if not isinstance(when, ToolCalled):
            continue
        if when.tool_name in seen:
            continue
        seen.add(when.tool_name)
        tools.append(make_handoff_tool(agent_client, when.tool_name))
    return tools
