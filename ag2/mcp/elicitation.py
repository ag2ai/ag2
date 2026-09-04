# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Route a served agent's ``context.input()`` to the human behind the MCP client.

The question travels as an MCP *elicitation*, in whichever shape the negotiated
revision has: a standalone ``elicitation/create`` request awaited inline on the
handshake era, or — from 2026-07-28, which defines no server-to-client request —
back as the call's own result (see :mod:`ag2.mcp.pause`). Everything above the
transport is shared: one rendering, one capability check, one policy.
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

# ``context.input()`` is one string in, one string out, so the form it renders as
# has exactly one string property.
ANSWER_FIELD = "answer"


def input_form_schema() -> ElicitRequestedSchema:
    """The requested schema for a ``context.input()`` question.

    Top-level properties only, which is all elicitation's form mode permits.
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
    """Render one human-input request as its wire elicitation."""
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
        # The channel worked and the human is done, so re-asking would loop: this
        # is the same "no answer" outcome as a decline.
        logger.warning("MCP elicitation accepted with no %r string; treated as no answer", ANSWER_FIELD)
        raise MCPElicitationDeclinedError("accept")
    return value


def can_answer(session: "ServerSession", policy: ElicitationPolicy) -> bool:
    """Whether this client may be put a question at all.

    Two gates: the deployment's policy, and the client's own declaration. A bare
    ``elicitation: {}`` — the only shape there was before modes existed — counts
    as form support; a url-only declaration does not, since a URL is not
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

    Registered as a stream interrupter ahead of whatever the agent registers for
    itself, which is what builds the fallback chain: returning the event passes
    the question on to the agent's own ``hitl_hook``, or — with none — to the
    existing "nobody could be asked" failure. Answering with a silent decline
    instead would hand the caller a degraded result of unexplained origin.
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
        # Present exactly on the modern era; the era is decided by whoever built
        # this, from the negotiated revision, not here.
        self._suspended = suspended

    async def __call__(self, event: HumanInputRequest, context: Context) -> "BaseEvent | None":
        session = self._request_context.session
        if not can_answer(session, self._policy):
            return event
        if self._suspended is not None:
            answered = await self._suspended.ask_for(input_request(event.content), ElicitResult)
            if answered is None:
                # The channel is not usable, so fall through to the agent's own
                # hook rather than invent a reply.
                return event
            result = answered
        else:
            # This branch only: a handshake-era client on a stateless transport
            # has no back-channel even having advertised elicitation.
            if not session.can_send_request:
                return event
            result = await session.elicit_form(
                event.content,
                input_form_schema(),
                related_request_id=self._request_context.request_id,
            )
        await context.send(HumanMessage.ensure_message(answer_from(result), parent_id=event.id))
        return None


__all__ = ("ANSWER_FIELD", "ClientElicitor")
