# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""First-class A2UI events for the agent stream.

A2UI messages are surfaced as :class:`A2UIMessageEvent` — one event per whole
A2UI server→client message — so every transport adapter (A2A, REST/SSE, AG-UI)
consumes a single, typed event seam instead of re-parsing the LLM's text.

The events are transient: they are derived from the model response and are not
persisted to durable history (the conversational prose is kept as a
``ModelMessage`` instead). See ``middleware.py`` for emission and
``a2a/executor.py`` for consumption.
"""

from autogen.beta.events import BaseEvent, Field

from ._types import ServerToClientMessage


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

    message: ServerToClientMessage = Field(kw_only=False)


__all__ = ("A2UIMessageEvent",)
