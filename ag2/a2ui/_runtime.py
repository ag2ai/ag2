# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The A2UI configuration a transport serves with, built once from the flat A2UI kwargs, and the
per-turn wiring every transport shares (middleware list, message collection). Not exported."""

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from ag2.middleware.base import Middleware

from ._types import A2UIVersion, JsonSchema, ServerToClientMessage
from ._versions import A2UIProtocol
from .actions import A2UIAction, A2UIEventAction, collect_action_declarations, collect_server_actions
from .capabilities import A2UIClientCapabilities, capabilities_to_prompt
from .events import A2UIMessageEvent
from .incoming import A2UIIncomingParseResult
from .middleware import A2UIExtractionMiddleware, A2UIInboundMiddleware, A2UIValidationMiddleware
from .parser import A2UIResponseParser
from .schema_manager import A2UISchemaManager

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant that can generate rich user interfaces "
    "using the A2UI protocol. When the user's request would benefit from a "
    "visual UI (cards, forms, lists, etc.), generate A2UI output. "
    "For simple text responses, respond normally without A2UI."
)


@dataclass(frozen=True, slots=True)
class _A2UIRuntime:
    """What every turn of an A2UI deployment shares."""

    protocol: A2UIProtocol
    catalog_id: str
    actions: tuple[A2UIEventAction, ...]
    """Declared to the LLM so it can render the buttons."""
    server_actions: Mapping[str, A2UIAction]
    """Run on the server when clicked, without invoking the agent."""
    system_prompt_section: str
    middleware: Middleware
    """Extracts the A2UI block from each response, and validates it unless validation is off."""

    def get_action(self, name: str) -> A2UIEventAction | None:
        """The declared action ``name``, which an incoming click is routed to."""
        return next((action for action in self.actions if action.name == name), None)

    def capabilities_prompt(self, caps: A2UIClientCapabilities | None) -> str:
        """Render negotiated client capabilities as a per-turn prompt fragment."""
        return capabilities_to_prompt(caps, catalog_id=self.catalog_id)

    def turn_middleware(self, interactions: Sequence[A2UIIncomingParseResult]) -> list[Middleware]:
        """The middleware one turn runs with: the runtime's own, plus one surfacing the client's interactions."""
        if not interactions:
            return [self.middleware]
        return [self.middleware, Middleware(A2UIInboundMiddleware, interactions=interactions)]


class _A2UIMessageCollector:
    """Appends each ``A2UIMessageEvent``'s message to ``into``; a module-level callable, not a per-turn closure."""

    def __init__(self, into: list[ServerToClientMessage]) -> None:
        self.into = into

    async def __call__(self, event: A2UIMessageEvent) -> None:
        self.into.append(event.message)


def collect_a2ui_messages(into: list[ServerToClientMessage]) -> Callable[[A2UIMessageEvent], Awaitable[None]]:
    """Collect each validated message from an ``A2UIMessageEvent`` subscription into ``into``."""
    return _A2UIMessageCollector(into)


def build_runtime(
    actions: Sequence[A2UIAction],
    *,
    protocol_version: A2UIVersion,
    custom_catalog: "str | os.PathLike[str] | JsonSchema | None",
    custom_catalog_rules: str | None,
    include_schema_in_prompt: bool,
    include_rules_in_prompt: bool,
    validate_responses: bool,
    validation_retries: int,
    system_message: str | None,
) -> _A2UIRuntime:
    """Build the runtime for ``A2UIServer`` and ``A2UIAgentExecutor`` from their shared kwargs."""
    schema_manager = A2UISchemaManager(
        protocol_version=protocol_version,
        custom_catalog=custom_catalog,
        custom_catalog_rules=custom_catalog_rules,
    )
    declarations = tuple(collect_action_declarations(actions))
    prompt_section = schema_manager.generate_prompt_section(
        include_schema=include_schema_in_prompt,
        include_rules=include_rules_in_prompt,
        actions=list(declarations),
    )
    base_prompt = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
    middleware: Middleware
    if validate_responses:
        if validation_retries < 0:
            raise ValueError(f"validation_retries must be >= 0, got {validation_retries}")
        parser = A2UIResponseParser(
            protocol=schema_manager.protocol,
            server_to_client_schema=schema_manager.server_to_client_schema,
            schema_registry=schema_manager.build_schema_registry(),
            component_references=schema_manager.component_references(),
        )
        middleware = Middleware(A2UIValidationMiddleware, parser=parser, max_retries=validation_retries)
    else:
        # Extraction always runs, so prose and UI travel as separate channels, never raw JSON in the text.
        middleware = Middleware(A2UIExtractionMiddleware, parser=A2UIResponseParser(protocol=schema_manager.protocol))
    return _A2UIRuntime(
        protocol=schema_manager.protocol,
        catalog_id=schema_manager.catalog_id,
        actions=declarations,
        server_actions=collect_server_actions(actions),
        system_prompt_section=f"{base_prompt}\n\n{prompt_section}",
        middleware=middleware,
    )
