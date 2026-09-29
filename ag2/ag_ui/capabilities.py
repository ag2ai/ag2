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
    SubagentInfo,
    ToolsCapabilities,
    TransportCapabilities,
)

from ag2 import Agent
from ag2.config.gemini.config import GeminiConfig
from ag2.tools.final import Toolkit
from ag2.tools.subagents.subagent_tool import SubagentTool
from ag2.tools.tool import Tool

from .input_acceptance import input_modalities


def _subagents(tools: tuple[Tool, ...]) -> list[SubagentInfo]:
    result = []
    for tool in tools:
        if isinstance(tool, SubagentTool):
            result.append(SubagentInfo(name=tool.agent.name, description=tool.description))
        elif isinstance(tool, Toolkit):
            result.extend(_subagents(tool.tools))
    return result


def served_capabilities(agent: Agent, *, client_tools: bool, state_snapshots: bool) -> AgentCapabilities:
    """What a server running `agent` tells a client it can do, before any run starts.

    Read off the agent alone: what a single run is handed — tools, a hook — is
    not known until it starts. `client_tools` and `state_snapshots` are the
    transport's: whether it runs the tools a client offers, and whether it
    sends `STATE_SNAPSHOT`.
    """
    # Undeclared rather than declared false where ag2 cannot tell: the protocol
    # reads an omitted field as saying nothing.
    subagents = _subagents(tuple(agent.tools))
    return AgentCapabilities(
        identity=IdentityCapabilities(name=agent.name, type="ag2"),
        transport=TransportCapabilities(streaming=True),
        tools=ToolsCapabilities(supported=True, client_provided=client_tools or None),
        state=StateCapabilities(snapshots=True) if state_snapshots else None,
        multi_agent=MultiAgentCapabilities(supported=True, delegation=True, subagents=subagents or None)
        if agent.tasks is not None or subagents
        else None,
        reasoning=ReasoningCapabilities(encrypted=isinstance(agent.config, GeminiConfig)),
        multimodal=MultimodalCapabilities(input=MultimodalInputCapabilities(**input_modalities(agent.config))),
        # A question the agent's own hook answers never reaches the client.
        human_in_the_loop=HumanInTheLoopCapabilities(supported=True, interrupts=agent._hitl_hook is None),
    )


__all__ = ("served_capabilities",)
