# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import TypeAlias

import httpx
from a2a.server.agent_execution import AgentExecutor
from a2a.server.cluster import TaskEventStream, VersionedTaskStore
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard, CancelTaskRequest, Task, TaskState
from a2a.utils.errors import TaskNotCancelableError

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
    """Hand the modifier a scratch copy and re-sign its per-request output."""
    # The SDK hands the modifier the one long-lived card shared by every
    # request and by the request handler's input-mode validation —
    # mutate-and-return must not make either drift, signed or not.
    if modifier is None:
        return None

    async def signed_modifier(card: AgentCard) -> AgentCard:
        return sign_card(await modifier(copy_card(card)), signer)

    return signed_modifier


def wrap_extended_card_modifier(
    modifier: ExtendedCardModifier | None, signer: CardSigner | None
) -> ExtendedCardModifier | None:
    """Extended-card twin of :func:`wrap_card_modifier` (modifier also takes ``ServerCallContext``)."""
    if modifier is None:
        return None

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


class _RequestHandler(DefaultRequestHandlerV2):
    """``DefaultRequestHandlerV2`` that refuses a cancel the task outran.

    A task that ends some other way while a cancel is in flight comes back from the SDK in the
    state that won. It was not cancelable, so answer ``TaskNotCancelableError`` as the spec asks
    for a terminal task, and as the SDK itself does when a cancel loses on the cross-replica path.
    """

    async def on_cancel_task(self, params: CancelTaskRequest, context: ServerCallContext) -> Task | None:
        task: Task | None = await super().on_cancel_task(params, context)
        if task is not None and task.status.state != TaskState.TASK_STATE_CANCELED:
            raise TaskNotCancelableError(
                message=f"Task finished as {TaskState.Name(task.status.state)} before it could be canceled"
            )
        return task


def build_default_handler(
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    extended_agent_card: AgentCard | None,
    extended_card_modifier: ExtendedCardModifier | None,
    task_store: TaskStore | VersionedTaskStore | None,
    event_stream: TaskEventStream | None,
    push_config_store: PushNotificationConfigStore | None,
    push_sender: PushNotificationSender | None,
    push_url_validator: Callable[[str], Awaitable[bool]] | None = None,
    validate_input_modes: bool = False,
) -> DefaultRequestHandlerV2:
    """Build the SDK request handler shared by all transports."""
    if push_config_store is not None and push_sender is None:
        # A store makes the card advertise push, and an advertised webhook must be called (spec §4.3);
        # the SDK's own default sends nothing. Delivery re-checks each URL with the same policy.
        push_sender = BasePushNotificationSender(
            httpx.AsyncClient(), push_config_store, push_url_validator=push_url_validator
        )
    return _RequestHandler(
        agent_executor=agent_executor,
        task_store=task_store or InMemoryTaskStore(),
        event_stream=event_stream,
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        push_config_store=push_config_store,
        push_sender=push_sender,
        push_url_validator=push_url_validator,
        validate_input_modes=validate_input_modes,
    )
