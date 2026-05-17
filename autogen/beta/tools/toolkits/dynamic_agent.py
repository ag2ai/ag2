# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""DynamicAgentToolkit — let an orchestrator agent spawn specialist sub-agents on demand.

An orchestrating agent equipped with this toolkit can call ``ask_specialist``
to create a one-shot specialist agent, delegate a task to it, and receive its
reply — all inside a single tool call.

Example::

    from autogen.beta import Agent
    from autogen.beta.tools import DynamicAgentToolkit
    from autogen.beta.config import OpenAIConfig

    config = OpenAIConfig("gpt-4o-mini")
    coordinator = Agent(
        "coordinator",
        config=config,
        prompt=(
            "You are a coordinator. Break complex requests into specialist tasks "
            "and delegate each part using ask_specialist."
        ),
        tools=[DynamicAgentToolkit(config=config)],
    )

    reply = await coordinator.ask("Research and summarise the history of the internet.")

The coordinator will autonomously call ``ask_specialist`` to spawn an ephemeral
researcher agent, collect its reply, and return a synthesised answer.

Sharing tools with specialists::

    from autogen.beta.tools import DynamicAgentToolkit, FilesystemToolkit

    fs = FilesystemToolkit()
    toolkit = DynamicAgentToolkit(config=config, tools=[fs])

    # Any specialist spawned by the orchestrator inherits the filesystem tools.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

if TYPE_CHECKING:
    from autogen.beta.config.config import ModelConfig
    from autogen.beta.tools.final import Tool

__all__ = ("DynamicAgentToolkit",)


class DynamicAgentToolkit(Toolkit):
    """Toolkit that lets an agent spawn specialist sub-agents on demand.

    Adds a single ``ask_specialist`` tool.  When the orchestrating agent calls
    it, an ephemeral :class:`~autogen.beta.Agent` is created with the given
    role, asked the given task, and its reply is returned as a string.

    Args:
        config:
            The :class:`~autogen.beta.config.ModelConfig` used for every
            spawned specialist.  Typically the same config as the orchestrator.
        tools:
            Tools to equip every spawned specialist with.  Defaults to none.

    Example::

        config = OpenAIConfig("gpt-4o-mini")
        orchestrator = Agent(
            "orchestrator",
            config=config,
            tools=[DynamicAgentToolkit(config=config)],
        )
    """

    __slots__ = ("_config", "_shared_tools")

    def __init__(
        self,
        config: ModelConfig,
        tools: Iterable[Tool] = (),
    ) -> None:
        self._config = config
        self._shared_tools: list[Any] = list(tools)

        super().__init__(
            self._ask_specialist_tool(),
            name="dynamic_agent_toolkit",
        )

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def _ask_specialist_tool(self) -> FunctionTool:
        config = self._config
        shared_tools = self._shared_tools

        @tool(
            name="ask_specialist",
            description=(
                "Spawn a temporary specialist agent with a custom role and delegate "
                "a task to it. Returns the specialist's reply as plain text. "
                "Use this to break complex requests into focused subtasks handled "
                "by purpose-built agents."
            ),
        )
        async def ask_specialist(
            role: Annotated[
                str,
                Field(
                    description=(
                        "System-prompt that defines the specialist's persona and expertise, "
                        "e.g. 'You are an expert data analyst specialised in time-series data.'"
                    )
                ),
            ],
            task: Annotated[
                str,
                Field(description="The task or question to delegate to the specialist."),
            ],
            name: Annotated[
                str,
                Field(description="Optional name for the specialist agent (used in logs)."),
            ] = "specialist",
        ) -> str:
            # Late import avoids the tools → agent → tools circular dependency.
            from autogen.beta.agent import Agent

            agent = Agent(name=name, prompt=role, config=config, tools=shared_tools)
            reply = await agent.ask(task)
            content = await reply.content()
            return content or ""

        return ask_specialist  # type: ignore[return-value]
