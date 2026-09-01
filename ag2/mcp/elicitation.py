# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Route a served agent's ``context.input()`` to the human behind the MCP client.

A tool inside a served agent asks for input the way it always has. This module is
what carries the question out to the caller, so the same tool runs unchanged
in-process, over ACP, and over MCP.

The question travels as an MCP *elicitation*, and which shape that takes is
decided by the negotiated protocol revision rather than by configuration:

* **handshake era** (up to 2025-11-25) — a standalone ``elicitation/create``
  server-to-client request, awaited inline. Nothing pauses, nothing is stored.
* **modern era** (2026-07-28) — the revision defines no server-to-client request
  at all, so the question comes back as the *result* of the call, inside an
  ``InputRequiredResult``. See :mod:`ag2.mcp.pause`.

Both eras share everything above the transport: the same rendered form, the same
capability check, and the same policy. :class:`ClientElicitor` is registered as a
stream *interrupter* ahead of the agent's own ``hitl_hook``, which is what makes
the fallback chain read the way the failure model already reads — the calling
client's human first, the server-side hook second, and the existing "nobody
could be asked" failure when there is neither.
"""

import logging
from typing import TYPE_CHECKING, Any

from mcp.types import (
    ClientCapabilities,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestedSchema,
    ElicitResult,
)

from ag2.annotations import Context
from ag2.events import BaseEvent, HumanInputRequest, HumanMessage
from ag2.hitl import ElicitationPolicy

from .errors import MCPElicitationDeclinedError
from .pause import SuspendedTurn

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext
    from mcp.server.session import ServerSession

logger = logging.getLogger(__name__)

# The single field of the form a ``context.input()`` question renders as. That
# call's contract is one string in, one string out, so the form has exactly one
# string property and the answer is read straight back out of it.
ANSWER_FIELD = "answer"


def input_form_schema() -> ElicitRequestedSchema:
    """The requested schema for a ``context.input()`` question.

    A restricted subset of JSON Schema — top-level properties only, no nesting —
    which is all elicitation's form mode permits, and all this question needs.
    """
    return {
        "type": "object",
        "properties": {
            ANSWER_FIELD: {
                "type": "string",
                "title": "Answer",
                "description": "Your answer to the agent's question.",
            },
        },
        "required": [ANSWER_FIELD],
    }


def input_request(message: str) -> ElicitRequest:
    """Render one human-input request as its wire elicitation.

    The same rendering on both transports, so a client sees the identical
    question whichever era it negotiated — and so the modern era's answer/question
    pinning (which digests this rendering) cannot disagree with the handshake era.
    """
    return ElicitRequest(params=ElicitRequestFormParams(message=message, requested_schema=input_form_schema()))


def answer_from(result: ElicitResult) -> str:
    """The human's answer, or raise because there is none.

    Raises:
        MCPElicitationDeclinedError: The client declined or dismissed the
            question, or accepted it without the answer the form asked for.
    """
    if result.action != "accept":
        raise MCPElicitationDeclinedError(result.action)
    value = (result.content or {}).get(ANSWER_FIELD)
    if not isinstance(value, str):
        # An accept whose content does not fit the schema the server sent. The
        # channel worked and the human is done with the question, so re-asking
        # would loop; this is the same "no answer" outcome as a decline.
        logger.warning("MCP elicitation accepted with no %r string; treated as no answer", ANSWER_FIELD)
        raise MCPElicitationDeclinedError("accept")
    return value


def can_answer(session: "ServerSession", policy: ElicitationPolicy) -> bool:
    """Whether this client may be put a question at all.

    Two gates, and both have to pass before anything is sent:

    * the **policy** — ``"decline"`` means this deployment never asks its
      clients, so nothing is sent and nothing is checked;
    * the client's own **declaration** — a client that did not advertise form
      elicitation is never asked, so it is never handed a question it must
      refuse.

    A bare ``elicitation: {}`` — the only shape there was before modes existed —
    counts as form support; a url-only declaration does not, since a URL is not
    somewhere a free-text answer can come from.
    """
    if policy == "decline":
        return False
    capabilities: ClientCapabilities | None = session.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    if elicitation is None:
        return False
    return elicitation.form is not None or elicitation.url is None


class ClientElicitor:
    """Answers a served agent's ``context.input()`` from the calling client's human.

    Registered as a stream interrupter on the turn's stream, ahead of whatever
    the agent registers for itself. Returning ``None`` consumes the question (the
    answer has been published); returning the event passes it on, which is how a
    client that cannot answer falls through to the agent's own ``hitl_hook`` — or,
    with none configured, to the existing "nobody could be asked" failure and its
    instructional message. A silent decline in its place would hand the caller a
    degraded result whose origin they could not see.
    """

    __slots__ = ("_request_context", "_policy", "_suspended")

    def __init__(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        *,
        policy: ElicitationPolicy = "ask",
        suspended: SuspendedTurn | None = None,
    ) -> None:
        self._request_context = request_context
        self._policy = policy
        # Present exactly on the modern era, where the question has to come back
        # as the *result* of the call. The era is read from the negotiated
        # revision by whoever built this, not decided here.
        self._suspended = suspended

    async def __call__(self, event: HumanInputRequest, context: Context) -> "BaseEvent | None":
        session = self._request_context.session
        if not can_answer(session, self._policy):
            return event
        if self._suspended is not None:
            result = await self._suspended.ask(input_request(event.content))
        else:
            # ``can_send_request`` gates only this branch: it asks whether *this
            # channel* can carry a server-initiated request, which is a question
            # the modern era does not have — nothing is sent there. A
            # handshake-era client on a stateless transport has no back-channel
            # even having advertised elicitation.
            if not session.can_send_request:
                return event
            result = await session.elicit_form(
                event.content,
                input_form_schema(),
                related_request_id=self._request_context.request_id,
            )
        await context.send(HumanMessage.ensure_message(answer_from(result), parent_id=event.id))
        return None


__all__ = (
    "ANSWER_FIELD",
    "ClientElicitor",
    "answer_from",
    "can_answer",
    "input_form_schema",
    "input_request",
)
