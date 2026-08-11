# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""First-class A2UI events for the agent stream: each A2UI server→client message
is surfaced as one :class:`A2UIMessageEvent` for transport adapters to consume.
The events are transient and not persisted to durable history.
"""

from ag2.events import BaseEvent, Field

from ._types import ServerToClientMessage
from .incoming import A2UIIncomingParseResult


class A2UIValidationFailedEvent(BaseEvent):
    """Signals that A2UI validation failed after all retries were exhausted.

    Emitted by the A2UI validation middleware on the turn's stream when the
    model could not produce schema-valid A2UI within ``validation_retries + 1``
    attempts. The middleware then *gracefully degrades* to a prose-only
    response: it strips the invalid A2UI block and emits **no**
    :class:`A2UIMessageEvent`.

    This is deliberately an internal observability seam, **not** a wire frame.
    The A2UI spec has no server→client error message (errors are client→server
    only; server→client is createSurface/updateComponents/updateDataModel/
    deleteSurface [+ v1.0 callFunction/actionResponse]), so emitting a wire-level
    error would diverge from the protocol. The transports therefore keep their
    graceful-degradation behaviour unchanged; observers/monitoring subscribed to
    the stream consume this event to tell "failed to build UI" apart from
    "agent intentionally answered with text".

    Transient: derived from the model response, not persisted to durable history.
    """

    __transient__ = True

    # ``Field`` is a runtime descriptor; mypy can't see that it resolves to the
    # annotated type, so the (correct) annotations need an assignment ignore —
    # the same framework-wide pattern as the other ``BaseEvent`` subclasses.
    errors: list[str] = Field(kw_only=False)  # type: ignore[assignment]
    attempts: int = Field(kw_only=False)  # type: ignore[assignment]


class A2UIMessageEvent(BaseEvent):
    """A single, fully-formed A2UI server→client message.

    Emitted by the A2UI validation middleware after a model response is parsed
    and validated — one event per A2UI message (level A.1, per-message). The
    payload is the canonical A2UI message dict (e.g. ``createSurface`` /
    ``updateComponents``), ready to serialize to the A2UI wire format.

    Transient: the message is reconstructable from the validated response and
    is carried out-of-band of durable history (which keeps the prose only).
    """

    __transient__ = True

    message: ServerToClientMessage = Field(kw_only=False)  # type: ignore[assignment]


class A2UIClientEvent(BaseEvent):
    """A single client→server A2UI interaction received from the renderer.

    Emitted on the turn's stream when a transport receives a client→server
    envelope — an ``action`` (a click on a server ``event`` button), a v1.0
    ``functionResponse`` (the client's reply to a server-initiated
    ``callFunction``), or an ``error``. Symmetric to :class:`A2UIMessageEvent`
    (server→client): it lets server-side observers react to / log client
    interactions, in addition to the envelope being rewritten into the LLM's
    prompt for the turn.

    Per the A2UI spec a purely client-side ``functionCall`` action (e.g.
    ``openUrl``) runs on the renderer with **no** network round-trip, so it
    produces no envelope and therefore no event here.

    Emitted from a per-turn middleware (so it fires inside the turn, where the
    agent's observers are subscribed). Transient: derived from the request, not
    persisted to durable history.
    """

    __transient__ = True

    interaction: A2UIIncomingParseResult = Field(kw_only=False)  # type: ignore[assignment]


__all__ = ("A2UIClientEvent", "A2UIMessageEvent", "A2UIValidationFailedEvent")
