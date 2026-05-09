# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow handoff tools.

Each ``(tool_name, target_name)`` pair becomes one LLM tool. When the
LLM invokes the tool, the body returns a typed
:class:`autogen.beta.network.handoff.Handoff` instance, which the
:class:`WorkflowAdapter` reads off the agent's local
``ToolResultEvent`` stream and uses to populate ``routing.target`` on
the round's ``EV_PACKET`` envelope. The next-speaker decision flows
straight from the tool body — no matching ``ToolCalled`` graph rule
is required.

The LLM never sees ``Handoff`` directly — it sees a button labelled
with the tool name and a one-line description. The protocol does
the rest. **Handoffs are a UX over the choreography.**

Usage:
    from autogen.beta.network.client.tools.handoff import (
        make_handoff_tools,
    )

    tools = make_handoff_tools({"delegate_bob": "bob", "delegate_carol": "carol"})
    for t in tools:
        client.agent.tools.append(t)

Or via the convenience method on :class:`NetworkPlugin`:
    plugin.register_workflow(graph)  # walks ToolCalled→AgentTarget pairs
"""

from collections.abc import Callable
from typing import Any

from autogen.beta.tools import tool

from ...handoff import Handoff

__all__ = ("make_handoff_tools",)


def make_handoff_tools(
    handoffs: dict[str, str],
    *,
    description: Callable[[str, str], str] | None = None,
) -> list[Any]:
    """Generate one Handoff-returning tool per ``(tool_name, target_name)``.

    Each generated tool has the body::

        async def <tool_name>(reason: str = "") -> Handoff:
            return Handoff(target="<target_name>", reason=reason)

    The workflow adapter sees the :class:`Handoff` in the agent's
    local ``ToolResultEvent`` stream and routes the next speaker to
    ``target_name`` — no ``ToolCalled`` graph rule required.

    ``description`` is an optional ``(tool_name, target_name) -> str``
    callback to customise the LLM-facing description; the default is
    ``f"Hand off to {target_name}."``.
    """
    tools: list[Any] = []
    for tool_name, target_name in handoffs.items():
        desc = description(tool_name, target_name) if description is not None else f"Hand off to {target_name}."
        tools.append(_build_handoff_tool(tool_name, target_name, desc))
    return tools


def _build_handoff_tool(tool_name: str, target_name: str, desc: str) -> Any:
    """Construct one Handoff-returning ``@tool``-decorated coroutine.

    Defined at module scope so the ``target_name`` is bound through
    the function argument rather than a loop-late closure.
    """

    @tool(name=tool_name, description=desc)
    async def _handoff(reason: str = "") -> Handoff:
        return Handoff(target=target_name, reason=reason)

    return _handoff
