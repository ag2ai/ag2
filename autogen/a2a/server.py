# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from pydantic import Field

from autogen import ConversableAgent
from autogen.doc_utils import export_module

from .agent_executor import AutogenAgentExecutor

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContextBuilder
    from a2a.server.apps import CallContextBuilder
    from a2a.server.context import ServerCallContext
    from a2a.server.events import QueueManager
    from a2a.server.request_handlers import RequestHandler
    from a2a.server.tasks import PushNotificationConfigStore, PushNotificationSender, TaskStore
    from starlette.applications import Starlette

    from autogen import ConversableAgent


@export_module("autogen.a2a")
class CardSettings(AgentCard):
    """Original A2A AgentCard object inheritor making some fields optional."""

    name: str | None = None  # type: ignore[assignment]
    """
    A human-readable name for the agent. Uses original agent name if not set.
    """

    description: str | None = None  # type: ignore[assignment]
    """
    A human-readable description of the agent, assisting users and other agents
    in understanding its purpose. Uses original agent description if not set.
    """

    url: str | None = None  # type: ignore[assignment]
    """
    The preferred endpoint URL for interacting with the agent.
    This URL MUST support the transport specified by 'preferredTransport'.
    Uses original A2aAgentServer url if not set.
    """

    version: str = "0.1.0"
    """
    The agent's own version number. The format is defined by the provider.
    """

    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    """
    Default set of supported input MIME types for all skills, which can be
    overridden on a per-skill basis.
    """

    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    """
    Default set of supported output MIME types for all skills, which can be
    overridden on a per-skill basis.
    """

    capabilities: AgentCapabilities = Field(default_factory=lambda: AgentCapabilities(streaming=True))
    """
    A declaration of optional capabilities supported by the agent.
    """

    skills: list[AgentSkill] = Field(default_factory=list)
    """
    The set of skills, or distinct capabilities, that the agent can perform.
    """


@export_module("autogen.a2a")
class A2aAgentServer:
    """A server wrapper for running an AG2 agent via the A2A protocol.

    This class provides functionality to wrap an AG2 ConversableAgent into an A2A server
    that can be used to interact with the agent through A2A requests.
    """

    def __init__(
        self,
        agent: "ConversableAgent",
        *,
        url: str | None = "http://localhost:8000",
        agent_card: CardSettings | None = None,
        card_modifier: Callable[["AgentCard"], "AgentCard"] | None = None,
        extended_agent_card: CardSettings | None = None,
        extended_card_modifier: Callable[["AgentCard", "ServerCallContext"], "AgentCard"] | None = None,
    ) -> None:
        """Initialize the A2aAgentServer.

        Args:
            agent: The Autogen ConversableAgent to serve.
            url: The base URL for the A2A server.
            agent_card: Configuration for the base agent card.
            card_modifier: Function to modify the base agent card.
            extended_agent_card: Configuration for the extended agent card.
            extended_card_modifier: Function to modify the extended agent card.
        """
        self.agent = agent

        if not agent_card:
            agent_card = CardSettings()

        if agent_card.url and url != "http://localhost:8000":
            warnings.warn(
                (
                    "You can't use `agent_card.url` and `url` options in the same time. "
                    f"`agent_card.url` has a higher priority, so `{agent_card.url}` will be used."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        self.card = AgentCard.model_validate({
            # use agent options by default
            "name": agent.name,
            "description": agent.description,
            "url": url,
            "supports_authenticated_extended_card": extended_agent_card is not None,
            # exclude name and description if not provided
            **agent_card.model_dump(exclude_none=True),
        })

        self.extended_agent_card: AgentCard | None = None
        if extended_agent_card:
            if extended_agent_card.url and url != "http://localhost:8000":
                warnings.warn(
                    (
                        "You can't use `extended_agent_card.url` and `url` options in the same time. "
                        f"`agent_card.url` has a higher priority, so `{extended_agent_card.url}` will be used."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

            self.extended_agent_card = AgentCard.model_validate({
                "name": agent.name,
                "description": agent.description,
                "url": url,
                **extended_agent_card.model_dump(exclude_none=True),
            })

        self.card_modifier = card_modifier
        self.extended_card_modifier = extended_card_modifier

    @property
    def executor(self) -> AutogenAgentExecutor:
        """Get the A2A agent executor."""
        return AutogenAgentExecutor(self.agent)

    def build_request_handler(
        self,
        *,
        task_store: "TaskStore | None" = None,
        queue_manager: "QueueManager | None" = None,
        push_config_store: "PushNotificationConfigStore | None" = None,
        push_sender: "PushNotificationSender | None" = None,
        request_context_builder: "RequestContextBuilder | None" = None,
    ) -> "RequestHandler":
        """Build a request handler for A2A application.

        Args:
            task_store: The task store to use.
            queue_manager: The queue manager to use.
            push_config_store: The push notification config store to use.
            push_sender: The push notification sender to use.
            request_context_builder: The request context builder to use.

        Returns:
            A configured RequestHandler instance.
        """
        return DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=task_store or InMemoryTaskStore(),
            queue_manager=queue_manager,
            push_config_store=push_config_store,
            push_sender=push_sender,
            request_context_builder=request_context_builder,
        )

    def build_starlette_app(
        self,
        *,
        request_handler: "RequestHandler | None" = None,
        context_builder: "CallContextBuilder | None" = None,
    ) -> "Starlette":
        """Build a Starlette A2A application for ASGI server.

        Args:
            request_handler: The request handler to use.
            context_builder: The context builder to use.

        Returns:
            A configured Starlette application instance.
        """
        from a2a.server.apps import A2AStarletteApplication

        return A2AStarletteApplication(
            agent_card=self.card,
            extended_agent_card=self.extended_agent_card,
            http_handler=request_handler
            or DefaultRequestHandler(
                agent_executor=self.executor,
                task_store=InMemoryTaskStore(),
            ),
            context_builder=context_builder,
            card_modifier=self.card_modifier,
            extended_card_modifier=self.extended_card_modifier,
        ).build()

    build = build_starlette_app  # default alias for build_starlette_app
