# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag_ui.core import (
    AgentCapabilities,
    HumanInTheLoopCapabilities,
    IdentityCapabilities,
    MultiAgentCapabilities,
    MultimodalCapabilities,
    MultimodalInputCapabilities,
    ReasoningCapabilities,
    StateCapabilities,
    ToolsCapabilities,
    TransportCapabilities,
)

from ag2 import Agent

# The parts `map_agui_content_to_input` turns into model input. A document
# part is the protocol's PDF; it has no part for a file of any other kind.
_INPUT = MultimodalInputCapabilities(image=True, audio=True, video=True, pdf=True)


def served_capabilities(agent: Agent, *, client_tools: bool, state_snapshots: bool) -> AgentCapabilities:
    """What a server running `agent` tells a client it can do, before any run starts.

    Read off the agent alone: what a single run is handed — tools, a hook — is
    not known until it starts. `client_tools` and `state_snapshots` are the
    transport's: whether it runs the tools a client offers, and whether it
    sends `STATE_SNAPSHOT`.
    """
    # Undeclared rather than declared false where ag2 cannot tell: the protocol
    # reads an omitted field as saying nothing. A delegation tool from
    # `as_tool` keeps no mark of the agent behind it, so only `run_subtask`
    # is known to delegate.
    return AgentCapabilities(
        identity=IdentityCapabilities(name=agent.name, type="ag2"),
        transport=TransportCapabilities(streaming=True),
        tools=ToolsCapabilities(supported=True, client_provided=client_tools or None),
        state=StateCapabilities(snapshots=True) if state_snapshots else None,
        multi_agent=MultiAgentCapabilities(supported=True, delegation=True) if agent.tasks is not None else None,
        reasoning=ReasoningCapabilities(encrypted=False),
        multimodal=MultimodalCapabilities(input=_INPUT),
        # A question the agent's own hook answers never reaches the client.
        human_in_the_loop=HumanInTheLoopCapabilities(supported=True, interrupts=agent._hitl_hook is None),
    )


__all__ = ("served_capabilities",)
