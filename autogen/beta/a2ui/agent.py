# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

from autogen.beta.agent import Agent, KnowledgeConfig, Plugin, PromptType, TaskConfig
from autogen.beta.assembly import AssemblyPolicy
from autogen.beta.config import ModelConfig
from autogen.beta.hitl import HumanHook
from autogen.beta.middleware.base import MiddlewareFactory
from autogen.beta.observers import Observer
from autogen.beta.response import ResponseProto
from autogen.beta.tools.tool import Tool

from ._types import JsonSchema
from .actions import A2UIAction
from .constants import A2UI_DEFAULT_DELIMITER
from .middleware import A2UIValidationMiddleware
from .parser import A2UIResponseParser
from .schema_manager import A2UISchemaManager

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant that can generate rich user interfaces "
    "using the A2UI protocol. When the user's request would benefit from a "
    "visual UI (cards, forms, lists, etc.), generate A2UI output. "
    "For simple text responses, respond normally without A2UI."
)


class A2UIAgent(Agent):
    """An autogen.beta.Agent that produces A2UI rich UI output.

    Supports protocol version v0.9 with the basic catalog and optional custom
    catalogs. The LLM emits text + delimiter + A2UI JSON; the agent validates
    the JSON against the catalog schema via a retry-on-error middleware
    before yielding the final ``ModelResponse``.

    Example::

        from autogen.beta.config import AnthropicConfig
        from autogen.beta.a2ui import A2UIAgent, A2UIAction

        agent = A2UIAgent(
            name="ui_agent",
            config=AnthropicConfig(...),
            actions=[A2UIAction("submit", tool_name="submit_form")],
            validate_responses=True,
            validation_retries=2,
        )
        result = await agent.ask("show me a booking form")
    """

    def __init__(
        self,
        name: str = "a2ui_agent",
        prompt: "PromptType | Iterable[PromptType]" = (),
        *,
        config: ModelConfig | None = None,
        # A2UI-specific
        protocol_version: str = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        response_delimiter: str = A2UI_DEFAULT_DELIMITER,
        validate_responses: bool = True,
        validation_retries: int = 1,
        actions: Sequence[A2UIAction] = (),
        system_message: str | None = None,
        # Standard Agent kwargs — forwarded as-is
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        response_schema: ResponseProto[Any] | type | None = None,
        plugins: Iterable[Plugin] = (),
        knowledge: KnowledgeConfig | None = None,
        tasks: "TaskConfig | Literal[False]" = False,
        assembly: Iterable[AssemblyPolicy] = (),
    ) -> None:
        """Initialize the A2UIAgent.

        Args:
            name: The agent name.
            prompt: Optional extra prompt(s) appended to the default A2UI
                system prompt. May be a string, a callable prompt hook, or
                an iterable of either.
            config: LLM model configuration.
            protocol_version: A2UI protocol version. Currently only "v0.9".
            custom_catalog: A custom catalog that extends the basic catalog.
                Can be a path or dict. Must include a ``$id`` field used as
                the catalogId in A2UI messages.
            custom_catalog_rules: Plain-text rules for the custom catalog
                components, appended to the basic catalog rules.
            include_schema_in_prompt: Whether to include the full JSON schema
                in the system prompt (better validation, more tokens).
            include_rules_in_prompt: Whether to include catalog rules.
            response_delimiter: Delimiter separating text from A2UI JSON.
            validate_responses: Validate A2UI output against the schema and
                retry on failure. Requires ``ag2[a2ui]`` (jsonschema).
            validation_retries: Number of *additional* retries when validation
                fails. The middleware runs ``validation_retries + 1`` total
                LLM attempts. Setting to 0 disables retry — invalid output is
                silently degraded to text-only. Only used when
                ``validate_responses=True``.
            actions: A2UIAction definitions for button handling. Available
                actions are auto-injected into the system prompt.
            system_message: Custom prefix system message. If None, uses
                ``DEFAULT_SYSTEM_MESSAGE``.
            hitl_hook: Human-in-the-loop hook, forwarded to ``autogen.beta.Agent``.
            tools: Tools/callables the agent may call, forwarded to ``autogen.beta.Agent``.
            middleware: Extra middleware factories, forwarded to ``autogen.beta.Agent``
                (the A2UI validation middleware is appended automatically).
            observers: Event observers, forwarded to ``autogen.beta.Agent``.
            dependencies: Dependency-injection values, forwarded to ``autogen.beta.Agent``.
            variables: Context variables, forwarded to ``autogen.beta.Agent``.
            response_schema: Structured-output schema, forwarded to ``autogen.beta.Agent``.
            plugins: Plugins, forwarded to ``autogen.beta.Agent``.
            knowledge: Knowledge configuration, forwarded to ``autogen.beta.Agent``.
            tasks: Sub-task delegation configuration, forwarded to ``autogen.beta.Agent``.
            assembly: Assembly policies, forwarded to ``autogen.beta.Agent``.
        """
        self.schema_manager = A2UISchemaManager(
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
        )
        self.parser = A2UIResponseParser(
            version_string=self.schema_manager.version_string,
            delimiter=response_delimiter,
            server_to_client_schema=(self.schema_manager.server_to_client_schema if validate_responses else None),
            schema_registry=(self.schema_manager.build_schema_registry() if validate_responses else None),
            component_schemas=(self.schema_manager.get_component_schemas() if validate_responses else None),
            catalog_id=(self.schema_manager.catalog_id if validate_responses else None),
        )
        self.actions: tuple[A2UIAction, ...] = tuple(actions)
        self.response_delimiter = response_delimiter
        self.validation_retries = validation_retries

        self.a2ui_prompt_section = self.schema_manager.generate_prompt_section(
            include_schema=include_schema_in_prompt,
            include_rules=include_rules_in_prompt,
            response_delimiter=response_delimiter,
            actions=list(self.actions),
        )
        base_prompt = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
        full_system_prompt = f"{base_prompt}\n\n{self.a2ui_prompt_section}"

        if isinstance(prompt, str) or callable(prompt):
            extra_prompts: list[PromptType] = [prompt]
        else:
            extra_prompts = list(prompt)
        final_prompt: list[PromptType] = [full_system_prompt, *extra_prompts]

        final_middleware: list[MiddlewareFactory] = list(middleware)
        if validate_responses:
            final_middleware.append(A2UIValidationMiddleware(self.parser, validation_retries))

        super().__init__(
            name=name,
            prompt=final_prompt,
            config=config,
            hitl_hook=hitl_hook,
            tools=tools,
            middleware=final_middleware,
            observers=observers,
            dependencies=dependencies,
            variables=variables,
            response_schema=response_schema,
            plugins=plugins,
            knowledge=knowledge,
            tasks=tasks,
            assembly=assembly,
        )

    @property
    def protocol_version(self) -> str:
        """The A2UI protocol version this agent targets."""
        return self.schema_manager.protocol_version

    @property
    def catalog_id(self) -> str:
        """The A2UI catalog ID this agent uses."""
        return self.schema_manager.catalog_id

    def get_action(self, name: str) -> A2UIAction | None:
        """Look up a registered action by name."""
        for action in self.actions:
            if action.name == name:
                return action
        return None
