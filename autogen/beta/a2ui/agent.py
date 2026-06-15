# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable, Iterable
from typing import Any, Literal

from autogen.beta.agent import Agent, KnowledgeConfig, Plugin, PromptType, TaskConfig
from autogen.beta.annotations import Context
from autogen.beta.assembly import AssemblyPolicy
from autogen.beta.config import ModelConfig
from autogen.beta.hitl import HumanHook
from autogen.beta.middleware.base import MiddlewareFactory
from autogen.beta.observers import Observer
from autogen.beta.response import ResponseProto
from autogen.beta.tools.tool import Tool

from ._types import A2UIVersion, JsonSchema
from .action_tool import A2UIActionTool
from .actions import A2UIAction
from .capabilities import (
    A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY,
    A2UIClientCapabilities,
    capabilities_to_prompt,
)
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

    Supports protocol versions v0.9 (default), v0.9.1, and v1.0 with the basic
    catalog and optional custom catalogs. The LLM emits conversational prose
    plus an ``<a2ui-json>…</a2ui-json>`` block; the agent validates the messages
    against the catalog schema via a retry-on-error middleware, emits each
    validated message as an ``A2UIMessageEvent`` on the stream, and yields a
    prose-only ``ModelResponse``.

    A2UI actions (clickable buttons) are passed inline via ``tools=``:

    - A function decorated with :func:`a2ui_action` becomes an
      :class:`A2UIActionTool` — an executable tool plus a server ``event`` action
      routed back to it on click.
    - A bare :class:`A2UIAction` declares an action with no executable tool body
      (handled by the LLM, or a client-side ``functionCall``).
    - Anything else is a regular tool.

    Example::

        from autogen.beta.config import AnthropicConfig
        from autogen.beta.a2ui import A2UIAgent, a2ui_action


        @a2ui_action(description="Submit the booking form")
        def submit_form(date: str) -> str: ...


        agent = A2UIAgent(
            name="ui_agent",
            config=AnthropicConfig(...),
            tools=[submit_form],
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
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        validate_responses: bool = True,
        validation_retries: int = 1,
        system_message: str | None = None,
        # Standard Agent kwargs — forwarded as-is
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool | A2UIAction] = (),
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
            protocol_version: A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A custom catalog that extends the basic catalog.
                Can be a path or dict. Must include a ``$id`` field used as
                the catalogId in A2UI messages.
            custom_catalog_rules: Plain-text rules for the custom catalog
                components, appended to the basic catalog rules.
            include_schema_in_prompt: Whether to include the full JSON schema
                in the system prompt (better validation, more tokens).
            include_rules_in_prompt: Whether to include catalog rules.
            validate_responses: Validate A2UI output against the schema and
                retry on failure. Requires ``ag2[a2ui]`` (jsonschema).
            validation_retries: Number of *additional* retries when validation
                fails. The middleware runs ``validation_retries + 1`` total
                LLM attempts. Setting to 0 disables retry — invalid output is
                silently degraded to text-only. Only used when
                ``validate_responses=True``.
            system_message: Custom prefix system message. If None, uses
                ``DEFAULT_SYSTEM_MESSAGE``.
            hitl_hook: Human-in-the-loop hook, forwarded to ``autogen.beta.Agent``.
            tools: Tools/callables the agent may call, forwarded to ``autogen.beta.Agent``.
                May also include :func:`a2ui_action`-decorated tools and bare
                :class:`A2UIAction` declarations; both are collected as A2UI actions
                and injected into the system prompt. Bare ``A2UIAction`` items are
                not forwarded to ``autogen.beta.Agent`` as executable tools.
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
            server_to_client_schema=(self.schema_manager.server_to_client_schema if validate_responses else None),
            schema_registry=(self.schema_manager.build_schema_registry() if validate_responses else None),
            component_schemas=(self.schema_manager.get_component_schemas() if validate_responses else None),
            catalog_id=(self.schema_manager.catalog_id if validate_responses else None),
        )
        collected_actions, executable_tools = _split_tools(tools)
        self.actions: tuple[A2UIAction, ...] = collected_actions
        self.validation_retries = validation_retries

        self.a2ui_prompt_section = self.schema_manager.generate_prompt_section(
            include_schema=include_schema_in_prompt,
            include_rules=include_rules_in_prompt,
            actions=list(self.actions),
        )
        base_prompt = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
        full_system_prompt = f"{base_prompt}\n\n{self.a2ui_prompt_section}"

        if isinstance(prompt, str) or callable(prompt):
            extra_prompts: list[PromptType] = [prompt]
        else:
            extra_prompts = list(prompt)
        # The capabilities hook runs per-turn on the direct ``ask()`` path,
        # reading caps a caller stashed in dependencies. Transports inject caps
        # themselves (they build the context prompt directly and skip dynamic
        # hooks), so this only adds negotiation to in-process ``ask()`` use.
        final_prompt: list[PromptType] = [full_system_prompt, self._capabilities_prompt_hook, *extra_prompts]

        final_middleware: list[MiddlewareFactory] = list(middleware)
        if validate_responses:
            final_middleware.append(A2UIValidationMiddleware(self.parser, validation_retries))

        super().__init__(
            name=name,
            prompt=final_prompt,
            config=config,
            hitl_hook=hitl_hook,
            tools=executable_tools,
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
    def protocol_version(self) -> A2UIVersion:
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

    async def _capabilities_prompt_hook(self, context: Context) -> str:
        """Per-turn prompt fragment from caller-supplied client capabilities.

        On the direct ``ask()`` path a caller can stash an
        :class:`A2UIClientCapabilities` under
        ``A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY`` in ``dependencies`` to drive
        catalog negotiation. Returns ``""`` when none is present.
        """
        caps = context.dependencies.get(A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY)
        if not isinstance(caps, A2UIClientCapabilities):
            return ""
        return capabilities_to_prompt(caps, catalog_id=self.catalog_id)


def _split_tools(
    tools: Iterable[Callable[..., Any] | Tool | A2UIAction],
) -> tuple[tuple[A2UIAction, ...], list[Callable[..., Any] | Tool]]:
    """Partition ``tools=`` into collected A2UI actions and executable tools.

    - :class:`A2UIActionTool` contributes both its ``.action`` and itself (as an
      executable tool).
    - A bare :class:`A2UIAction` contributes only its declaration; it is **not**
      forwarded to ``autogen.beta.Agent`` as an executable tool.
    - Anything else is forwarded unchanged as an executable tool.

    Raises:
        ValueError: if two collected actions share the same ``name``.
    """
    actions: list[A2UIAction] = []
    executable: list[Callable[..., Any] | Tool] = []
    for item in tools:
        if isinstance(item, A2UIActionTool):
            actions.append(item.action)
            executable.append(item)
        elif isinstance(item, A2UIAction):
            actions.append(item)
        else:
            executable.append(item)

    seen: set[str] = set()
    for action in actions:
        if action.name in seen:
            raise ValueError(
                f"Duplicate A2UI action name {action.name!r}; action names passed via tools= must be unique."
            )
        seen.add(action.name)

    return tuple(actions), executable
