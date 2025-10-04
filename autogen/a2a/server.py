# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING

from a2a.server.apps import CallContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler, RequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from pydantic import Field

from autogen import ConversableAgent

from .agent_executor import AutogenAgentExecutor

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from autogen import ConversableAgent


class CardSettings(AgentCard):
    name: str | None = None
    description: str | None = None
    url: str = "http://localhost:8000"
    version: str = "0.1.0"

    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    capabilities: AgentCapabilities = Field(default_factory=lambda: AgentCapabilities(streaming=True))
    skills: list[AgentSkill] = Field(default_factory=list)


class A2aAgentServer:
    def __init__(
        self,
        agent: "ConversableAgent",
        *,
        agent_card: CardSettings | None = None,
        card_modifier: Callable[[AgentCard], AgentCard] | None = None,
        extended_agent_card: AgentCard | None = None,
        extended_card_modifier: Callable[[AgentCard, ServerCallContext], AgentCard] | None = None,
    ) -> None:
        self.agent = agent

        if not agent_card:
            agent_card = CardSettings()

        self.card = AgentCard.model_validate({
            # use agent options by default
            "name": agent.name,
            "description": agent.description,
            "supports_authenticated_extended_card": extended_agent_card is not None,
            # exclude name and description if not provided
            **agent_card.model_dump(exclude_none=True),
        })

        self.extended_agent_card = extended_agent_card
        self.card_modifier = card_modifier
        self.extended_card_modifier = extended_card_modifier

    @property
    def agent_executor(self) -> AutogenAgentExecutor:
        return AutogenAgentExecutor(self.agent)

    def build_starlette_app(
        self,
        *,
        request_handler: RequestHandler | None = None,
        context_builder: CallContextBuilder | None = None,
    ) -> "Starlette":
        from a2a.server.apps import A2AStarletteApplication

        return A2AStarletteApplication(
            agent_card=self.card,
            extended_agent_card=self.extended_agent_card,
            http_handler=request_handler
            or DefaultRequestHandler(
                agent_executor=self.agent_executor,
                task_store=InMemoryTaskStore(),
            ),
            context_builder=context_builder,
            card_modifier=self.card_modifier,
            extended_card_modifier=self.extended_card_modifier,
        ).build()

    build = build_starlette_app  # default alias for build_starlette_app
