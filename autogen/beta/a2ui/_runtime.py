# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Internal A2UI runtime shared by the transports: holds the schema manager,
response parser, action declarations, system-prompt section, and validation
middleware factory built from a transport's flat A2UI kwargs. Not exported.
"""

import os

from autogen.beta.agent import Agent
from autogen.beta.middleware.base import MiddlewareFactory

from ._types import A2UIVersion, JsonSchema
from .action_tool import A2UIActionTool
from .actions import A2UIEventAction
from .capabilities import A2UIClientCapabilities, capabilities_to_prompt
from .middleware import A2UIValidationMiddleware
from .parser import A2UIResponseParser
from .schema_manager import A2UISchemaManager

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant that can generate rich user interfaces "
    "using the A2UI protocol. When the user's request would benefit from a "
    "visual UI (cards, forms, lists, etc.), generate A2UI output. "
    "For simple text responses, respond normally without A2UI."
)


class _A2UIRuntime:
    """Carries A2UI configuration and per-turn injection for a plain ``Agent``.

    Built once per transport wrapper from its A2UI kwargs. Exposes the read-only
    bits the transports need (``protocol_version``, ``catalog_id``,
    ``version_string``, ``get_action``), the prompt section to prepend, the
    validation middleware factories to inject, and a capabilities-prompt helper.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        validate_responses: bool = True,
        validation_retries: int = 1,
        system_message: str | None = None,
    ) -> None:
        self.schema_manager = A2UISchemaManager(
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
        )
        self._validate_responses = validate_responses
        self.parser = A2UIResponseParser(
            version_string=self.schema_manager.version_string,
            server_to_client_schema=(self.schema_manager.server_to_client_schema if validate_responses else None),
            schema_registry=(self.schema_manager.build_schema_registry() if validate_responses else None),
            component_schemas=(self.schema_manager.get_component_schemas() if validate_responses else None),
            catalog_id=(self.schema_manager.catalog_id if validate_responses else None),
        )
        # Clickable buttons are tool-backed (@a2ui_action) and ride on the
        # agent's tool list; discover them there.
        self.actions: tuple[A2UIEventAction, ...] = _collect_actions(agent)

        prompt_section = self.schema_manager.generate_prompt_section(
            include_schema=include_schema_in_prompt,
            include_rules=include_rules_in_prompt,
            actions=list(self.actions),
        )
        base_prompt = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
        self.system_prompt_section = f"{base_prompt}\n\n{prompt_section}"

        self._middleware: MiddlewareFactory | None = (
            A2UIValidationMiddleware(self.parser, validation_retries) if validate_responses else None
        )

    @property
    def protocol_version(self) -> A2UIVersion:
        """The A2UI protocol version this runtime targets."""
        return self.schema_manager.protocol_version

    @property
    def catalog_id(self) -> str:
        """The A2UI catalog ID in use."""
        return self.schema_manager.catalog_id

    @property
    def version_string(self) -> A2UIVersion:
        """The wire ``version`` string stamped on messages (e.g. ``"v0.9.1"``)."""
        return self.schema_manager.version_string

    def get_action(self, name: str) -> A2UIEventAction | None:
        """Look up a tool-backed action by name (used to route incoming clicks)."""
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def middleware_factories(self) -> list[MiddlewareFactory]:
        """Validation middleware to inject for the turn (empty when disabled)."""
        return [self._middleware] if self._middleware is not None else []

    def capabilities_prompt(self, caps: A2UIClientCapabilities | None) -> str:
        """Render negotiated client capabilities as a per-turn prompt fragment."""
        return capabilities_to_prompt(caps, catalog_id=self.catalog_id)


def _collect_actions(agent: Agent) -> tuple[A2UIEventAction, ...]:
    """Collect A2UI action declarations for the prompt and click routing.

    Tool-backed buttons are discovered from ``agent.tools`` — each
    :class:`A2UIActionTool` carries its own ``event`` action. Tool names are
    already unique on the agent, so no extra de-duplication is needed.
    """
    return tuple(tool.action for tool in agent.tools if isinstance(tool, A2UIActionTool))
