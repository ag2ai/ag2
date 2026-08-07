# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from a2a.server.agent_execution import AgentExecutor
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.routing import BaseRoute

from ..errors import A2AStaleCardSignatureError

CardModifier: TypeAlias = Callable[[AgentCard], Awaitable[AgentCard]]
ExtendedCardModifier: TypeAlias = Callable[[AgentCard, ServerCallContext], Awaitable[AgentCard]]

CardSigner: TypeAlias = Callable[[AgentCard], AgentCard]


def copy_card(card: AgentCard, *, drop_signatures: bool = False) -> AgentCard:
    """Deep-copy ``card``, optionally dropping the signatures carried on it."""
    fresh = AgentCard()
    fresh.CopyFrom(card)
    if drop_signatures:
        del fresh.signatures[:]
    return fresh


def sign_card(card: AgentCard, signer: CardSigner | None) -> AgentCard:
    """Sign a copy of ``card`` with prior signatures dropped; identity when no signer."""
    # The SDK signer appends in place instead of replacing, so signing a
    # stripped copy is what keeps the caller's card unmutated and stops a
    # signature over some earlier payload shipping next to the fresh one.
    # Composed signers (key rotation) still land both: the drop happens once,
    # before the callable runs.
    if signer is None:
        return card
    return signer(copy_card(card, drop_signatures=True))


def wrap_card_modifier(modifier: CardModifier | None, signer: CardSigner | None) -> CardModifier | None:
    """Re-sign the modifier's per-request output so mutation doesn't void the JWS."""
    # The modifier gets a scratch copy because the SDK hands it the one
    # long-lived card shared by every request — mutate-and-return must not
    # make the served card drift.
    if modifier is None or signer is None:
        return modifier

    async def signed_modifier(card: AgentCard) -> AgentCard:
        return sign_card(await modifier(copy_card(card)), signer)

    return signed_modifier


def wrap_extended_card_modifier(
    modifier: ExtendedCardModifier | None, signer: CardSigner | None
) -> ExtendedCardModifier | None:
    """Extended-card twin of :func:`wrap_card_modifier` (modifier also takes ``ServerCallContext``)."""
    if modifier is None or signer is None:
        return modifier

    async def signed_modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
        return sign_card(await modifier(copy_card(card), context), signer)

    return signed_modifier


# Legacy v0.x server-side card alias. Kept so pre-v1 clients still discover the card.
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"
DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


def clone_card_with_capabilities(card: AgentCard, *, extended: bool, push: bool) -> AgentCard:
    """Deep-copy ``card`` with capability flags flipped — never mutate caller's card."""
    new_card = AgentCard()
    new_card.CopyFrom(card)
    if extended:
        new_card.capabilities.extended_agent_card = True
    if push:
        new_card.capabilities.push_notifications = True
    return new_card


def prepare_public_card(
    card: AgentCard,
    *,
    extended: bool,
    push: bool,
    signer: CardSigner | None,
) -> AgentCard:
    """Flip the capability flags the server derives, then sign what we serve."""
    # Signing after the flip puts the flags inside the signed payload. A
    # caller-signed card with no signer to redo it can't take the flip at
    # all — that would serve a signature over a different payload.
    prepared = clone_card_with_capabilities(card, extended=extended, push=push)
    if signer is None and card.signatures and prepared.capabilities != card.capabilities:
        raise A2AStaleCardSignatureError(
            flipped=tuple(
                name
                for name, flip in (("extended_agent_card", extended), ("push_notifications", push))
                if flip and not getattr(card.capabilities, name)
            )
        )
    return sign_card(prepared, signer)


def build_default_handler(
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    extended_agent_card: AgentCard | None,
    extended_card_modifier: ExtendedCardModifier | None,
    task_store: TaskStore | None,
    push_config_store: PushNotificationConfigStore | None,
    push_sender: PushNotificationSender | None,
) -> DefaultRequestHandlerV2:
    """Build the SDK request handler shared by all transports."""
    return DefaultRequestHandlerV2(
        agent_executor=agent_executor,
        task_store=task_store or InMemoryTaskStore(),
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )


def build_card_routes_with_legacy(
    agent_card: AgentCard,
    *,
    card_modifier: CardModifier | None,
    card_url: str,
    legacy_card_url: str | None,
) -> list[BaseRoute]:
    """Card routes at v1.x ``card_url`` plus optional v0.x alias at ``legacy_card_url``."""
    routes: list[BaseRoute] = list(
        create_agent_card_routes(agent_card, card_modifier=card_modifier, card_url=card_url),
    )
    if legacy_card_url:
        routes.extend(
            create_agent_card_routes(
                agent_card,
                card_modifier=card_modifier,
                card_url=legacy_card_url,
            ),
        )
    return routes
