# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The A2UI configuration a transport serves with, built once from the flat A2UI kwargs, and the
per-turn wiring every transport shares (middleware list, message collection). Not exported."""

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant that can generate rich user interfaces "
    "using the A2UI protocol. When the user's request would benefit from a "
    "visual UI (cards, forms, lists, etc.), generate A2UI output. "
    "For simple text responses, respond normally without A2UI."
)


@dataclass(frozen=True, slots=True)
class A2UIOffer:
    """What the agent offers one renderer on one turn: the catalogs both sides have, and what follows from them.

    A renderer that advertises ``supportedCatalogIds`` gets only the agent's catalogs it names, as the official
    SDK's ``resolve_catalogs`` does. With none in common, or (v1.0) no capabilities at all, the turn offers no
    A2UI: :attr:`sends_ui` is false, the prompt has no A2UI section, and the transport sends no A2UI surfaces.
    """

    sends_ui: bool
    answers_function_calls: bool
    """Whether a ``callAgentFunction`` gets its ``agentFunctionResponse`` even without UI. The agent "MUST return"
    one (v1_0 a2ui_protocol.md, "callAgentFunction"), so a renderer that shares no catalog with the agent still gets
    its answer; one that withdrew A2UI by omitting its capabilities gets no A2UI message at all."""
    catalog_id: str | None
    """The catalog the prompt's ``createSurface`` names; ``None`` without A2UI."""
    catalog_ids: tuple[str, ...]
    """Every catalog a surface or function call may name this turn."""
    system_prompt_section: str
    middleware: Middleware
    """Extracts the A2UI block from each response, and validates it unless validation is off."""
    message_validator: A2UIResponseParser | None
    """Checks the messages server-side handlers return; ``None`` when validation is off or there is no A2UI."""

    def deliverable(self, messages: list[ServerToClientMessage], *, source: str) -> list[ServerToClientMessage]:
        """The ``messages`` from ``source`` this turn may send; the rest are dropped with a warning."""
        if self.sends_ui or not messages:
            return messages
        kept = [
            message
            for message in messages
            if self.answers_function_calls and isinstance(message, dict) and "agentFunctionResponse" in message
        ]
        logger.warning(
            "The renderer offers no A2UI this turn; dropping %d A2UI message(s) from %s.",
            len(messages) - len(kept),
            source,
        )
        return kept


@dataclass(frozen=True, slots=True)
class _A2UIRuntime:
    """What every turn of an A2UI deployment shares."""

    protocol: A2UIProtocol
    actions: tuple[A2UIEventAction, ...]
    """Declared to the LLM so it can render the buttons."""
    server_actions: Mapping[str, A2UIAction]
    """Run on the server when clicked, without invoking the agent."""
    full_offer: A2UIOffer
    """The offer to a renderer that names no catalogs: every catalog the agent serves."""
    no_ui_offer: A2UIOffer
    """The offer to a renderer with no catalog in common with the agent."""
    withdrawn_offer: A2UIOffer
    """The offer to a renderer that sent no capabilities where they are required (v1.0): no A2UI message at all."""
    _schema_manager: A2UISchemaManager
    _settings: "_OfferSettings"
    _offers: dict[frozenset[str], A2UIOffer] = field(default_factory=dict)
    """Offers built for a renderer's catalog subset, by the catalogs they serve; at most one per subset."""

    @property
    def catalog_ids(self) -> tuple[str, ...]:
        """Every catalog the agent serves."""
        return self.full_offer.catalog_ids

    def get_action(self, name: str) -> A2UIEventAction | None:
        """The declared action ``name``, which an incoming click is routed to."""
        return next((action for action in self.actions if action.name == name), None)

    def offer_for(self, caps: A2UIClientCapabilities | None) -> A2UIOffer:
        """What this turn offers a renderer with capabilities ``caps`` (``None`` when it sent none).

        In v1.0 a request without capabilities offers no A2UI ("omitting them withdraws A2UI for that turn",
        a2ui_protocol.md, "Renderer capabilities"); before v1.0 every catalog is offered, as the SDK does.
        """
        if caps is None:
            return self.withdrawn_offer if self.protocol.turn_scoped_capabilities else self.full_offer
        restricted = self._schema_manager.restricted_to(caps.supported_catalog_ids)
        if restricted is None:
            logger.warning(
                "A2UI renderer supports none of the agent's catalogs %r (it supports %r); this turn sends no A2UI.",
                list(self.catalog_ids),
                caps.supported_catalog_ids,
            )
            return self.no_ui_offer
        if restricted is self._schema_manager:
            return self.full_offer
        served = frozenset(restricted.catalog_ids)
        if served not in self._offers:
            self._offers[served] = _build_offer(restricted, self.actions, self._settings)
        return self._offers[served]

    def capabilities_prompt(self, caps: A2UIClientCapabilities | None, offer: A2UIOffer) -> str:
        """Render negotiated client capabilities as a per-turn prompt fragment."""
        if not offer.sends_ui or offer.catalog_id is None:
            return ""
        return capabilities_to_prompt(caps, catalog_id=offer.catalog_id)

    def turn_middleware(self, interactions: Sequence[A2UIIncomingParseResult], offer: A2UIOffer) -> list[Middleware]:
        """The middleware one turn runs with: the offer's own, plus one surfacing the client's interactions."""
        middleware = offer.middleware
        if not interactions:
            return [middleware]
        return [middleware, Middleware(A2UIInboundMiddleware, interactions=interactions)]


class A2UIMessageCollector:
    """Appends each ``A2UIMessageEvent``'s message to ``into``; subscribe it to a turn's stream."""

    def __init__(self, into: list[ServerToClientMessage]) -> None:
        self.into = into

    async def __call__(self, event: A2UIMessageEvent) -> None:
        self.into.append(event.message)


@dataclass(frozen=True, slots=True)
class _OfferSettings:
    """The deployment's prompt and validation settings, which every offer is built with."""

    include_schema_in_prompt: bool
    include_rules_in_prompt: bool
    validate_responses: bool
    validation_retries: int
    base_prompt: str


def _build_offer(
    schema_manager: A2UISchemaManager, declarations: Sequence[A2UIEventAction], settings: _OfferSettings
) -> A2UIOffer:
    prompt_section = schema_manager.generate_prompt_section(
        include_schema=settings.include_schema_in_prompt,
        include_rules=settings.include_rules_in_prompt,
        actions=list(declarations),
    )
    middleware: Middleware
    validator: A2UIResponseParser | None = None
    if settings.validate_responses:
        parser = A2UIResponseParser(
            protocol=schema_manager.protocol,
            server_to_client_schema=schema_manager.server_to_client_schema,
            schema_registry=schema_manager.build_schema_registry(),
            component_types=schema_manager.component_types(),
            catalog_ids=schema_manager.catalog_ids,
        )
        middleware = Middleware(A2UIValidationMiddleware, parser=parser, max_retries=settings.validation_retries)
        validator = parser
    else:
        # Extraction always runs, so prose and UI travel as separate channels, never raw JSON in the text.
        middleware = Middleware(A2UIExtractionMiddleware, parser=A2UIResponseParser(protocol=schema_manager.protocol))
    return A2UIOffer(
        sends_ui=True,
        answers_function_calls=True,
        catalog_id=schema_manager.catalog_id,
        catalog_ids=schema_manager.catalog_ids,
        system_prompt_section=f"{settings.base_prompt}\n\n{prompt_section}",
        middleware=middleware,
        message_validator=validator,
    )


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
    if validate_responses and validation_retries < 0:
        raise ValueError(f"validation_retries must be >= 0, got {validation_retries}")
    schema_manager = A2UISchemaManager(
        protocol_version=protocol_version,
        custom_catalog=custom_catalog,
        custom_catalog_rules=custom_catalog_rules,
    )
    declarations = tuple(collect_action_declarations(actions))
    settings = _OfferSettings(
        include_schema_in_prompt=include_schema_in_prompt,
        include_rules_in_prompt=include_rules_in_prompt,
        validate_responses=validate_responses,
        validation_retries=validation_retries,
        base_prompt=system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE,
    )
    protocol = schema_manager.protocol
    return _A2UIRuntime(
        protocol=protocol,
        actions=declarations,
        server_actions=collect_server_actions(actions),
        full_offer=_build_offer(schema_manager, declarations, settings),
        no_ui_offer=_no_ui_offer(protocol, system_message, answers_function_calls=True),
        withdrawn_offer=_no_ui_offer(protocol, system_message, answers_function_calls=False),
        _schema_manager=schema_manager,
        _settings=settings,
    )


def _no_ui_offer(protocol: A2UIProtocol, system_message: str | None, *, answers_function_calls: bool) -> A2UIOffer:
    return A2UIOffer(
        sends_ui=False,
        answers_function_calls=answers_function_calls,
        catalog_id=None,
        catalog_ids=(),
        # The default system message is about A2UI; a user's own one still applies.
        system_prompt_section=system_message or "",
        # Extraction still runs, so a stray A2UI block never reaches the prose; the transport drops its messages.
        middleware=Middleware(A2UIExtractionMiddleware, parser=A2UIResponseParser(protocol=protocol)),
        message_validator=None,
    )
