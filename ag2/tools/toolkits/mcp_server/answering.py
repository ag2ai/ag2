# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Answer a third-party MCP server's requests for input from the agent's own resources.

A proxied tool call can come back asking for something instead of returning a
result. This module is what answers: an *elicitation* goes to the agent's own
human-input channel (``context.input()`` — the same channel a local tool would
use), a *sampling* request runs on the agent's own model, and a *roots* request
is answered from directories the operator named.

All three are opt-in, and each is advertised to the server only when it is
enabled — the ``mcp`` client derives its declared capabilities from which
callbacks were supplied, so not enabling one means a conforming server never asks
for it. An untrusted server must not be able to provoke questions to users
unbidden or spend the operator's model budget just because the transport allows
it.

What a *refusal* looks like on the wire differs by request type, because the
protocol gives them different vocabularies. Elicitation has a first-class
``decline`` action, so a question this agent will not answer is declined and the
server can degrade deliberately. Sampling and roots have no such arm, so a
request for one that was never enabled comes back as an error. Either way the
server hears an answer rather than a dropped connection.
"""

import base64
import contextlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fast_depends.pydantic import PydanticSerializer
from mcp.client.session import ClientRequestContext
from mcp.types import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    AudioContent,
    CreateMessageRequestParams,
    CreateMessageResult,
    ElicitRequestParams,
    ElicitResult,
    ErrorData,
    ImageContent,
    ListRootsResult,
    Root,
    SamplingMessage,
    TextContent,
)

from ag2.annotations import Context
from ag2.context import ConversationContext
from ag2.events import BaseEvent, BinaryInput, BinaryType, Input, ModelMessage, ModelRequest, TextInput
from ag2.hitl import ElicitationPolicy
from ag2.stream import MemoryStream
from ag2.utils import MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY

if TYPE_CHECKING:
    # Under ``TYPE_CHECKING`` only: ``ag2.config`` reaches back into ``ag2.tools``
    # for the tool schemas, and this module is on that eager import path.
    from ag2.config.config import ModelConfig

logger = logging.getLogger(__name__)

# The one field a question rendered by AG2's own served side asks for. Read
# preferentially so an AG2-to-AG2 round trip answers the question it was asked;
# any other single-property form is answered on whatever property it names.
_AG2_ANSWER_FIELD = "answer"

_DECLINED = ElicitResult(action="decline")


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """What this agent answers when a third-party MCP server asks it for input.

    Every field is off by default: the toolkit reaches servers the operator may
    not control, and each of these hands one of the operator's own resources —
    their users' attention, their model budget, their filesystem layout — to the
    other end.

    Attributes:
        elicitation: ``"ask"`` routes a server's question to the agent's own
            human-input channel (``context.input()``, and thus the agent's
            ``hitl_hook``) and advertises the elicitation capability.
            ``"decline"`` (default) advertises nothing, so a conforming server
            never asks; one that asks anyway is declined. Same word, same two
            values, same reasoning as :data:`ag2.hitl.ElicitationPolicy`.
        sampling: When true, a server's ``sampling/createMessage`` runs on the
            agent's own model — the server gets an LLM without ever holding a
            key, and **the operator pays for it**. Off by default for that
            reason.
        roots: Directories reported to a server that scopes its work to roots.
            With none configured the capability is not advertised, which is the
            honest answer: the agent has no roots to report.
        max_rounds: How many times a server may come back asking for more before
            the call is abandoned. A server that re-asks indefinitely would
            otherwise loop the agent.
    """

    elicitation: ElicitationPolicy = "decline"
    sampling: bool = False
    roots: Sequence[str | os.PathLike[str]] = ()
    max_rounds: int = 10

    @property
    def answers_anything(self) -> bool:
        """Whether any request type is enabled at all."""
        return self.elicitation == "ask" or self.sampling or bool(self.roots)


class InputRequestAnswerer:
    """Answers one operation's worth of a server's input requests.

    Bound to the live agent :class:`~ag2.context.ConversationContext`, because
    that is where the agent's human and the agent's model both are.
    """

    __slots__ = ("_policy", "_context")

    def __init__(self, policy: AnswerPolicy, context: Context) -> None:
        self._policy = policy
        self._context = context

    def session_kwargs(self) -> dict[str, Any]:
        """The ``ClientSession`` callbacks for the enabled request types, and only those.

        The client derives what it advertises from which callbacks were supplied,
        so omitting one *is* how the capability goes unadvertised — there is no
        second switch to keep in step with this one.
        """
        kwargs: dict[str, Any] = {}
        if self._policy.elicitation == "ask":
            kwargs["elicitation_callback"] = self.on_elicitation
        if self._policy.sampling:
            kwargs["sampling_callback"] = self.on_sampling
        if self._policy.roots:
            kwargs["list_roots_callback"] = self.on_roots
        return kwargs

    async def on_elicitation(
        self, context: ClientRequestContext, params: ElicitRequestParams
    ) -> ElicitResult | ErrorData:
        """Put a server's question to the agent's own human.

        A ``HumanInputError`` is deliberately **not** caught: an absent channel is
        not a refusal, and reporting it as one would hand the server a decline
        the operator never made. It propagates and fails the tool call with the
        existing failure and its instructional message.
        """
        if self._policy.elicitation != "ask":
            return _DECLINED
        if params.mode != "form":
            # A URL-mode elicitation asks the human to complete an out-of-band
            # flow in a browser. ``context.input()`` is a text channel and has no
            # way to confirm a navigation happened, so declining is the honest
            # answer rather than accepting on a promise.
            logger.debug("declining a %r-mode MCP elicitation: no rendering for it", params.mode)
            return _DECLINED
        field = _sole_field(params.requested_schema)
        if field is None:
            # Anything beyond one property would mean deciding how to split a
            # single free-text answer across fields, which is fabricating data.
            logger.debug("declining an MCP elicitation whose form is not a single property")
            return _DECLINED
        answer = await self._context.input(params.message)
        return ElicitResult(action="accept", content={field: answer})

    async def on_sampling(
        self, context: ClientRequestContext, params: CreateMessageRequestParams
    ) -> CreateMessageResult | ErrorData:
        """Run a server's requested completion on the agent's own model.

        One completion against the model client, not a turn of the calling
        agent: the caller's tools, history and response schema stay out of it, so
        a borrowed model cannot reach back into the agent that lent it.

        The server's generation parameters — ``max_tokens``, ``temperature``,
        ``stop_sequences`` — are *not* forwarded. AG2 fixes those on the config,
        and the operator's configuration is what governs a call they are paying
        for. This is stated where the option is exposed.
        """
        if not self._policy.sampling:
            return ErrorData(code=INVALID_REQUEST, message="This client does not lend its model for sampling.")
        # The config the agent published for this turn, which is what "the
        # agent's own model" means: the same key the metrics middleware reads.
        config = cast("ModelConfig | None", self._context.dependencies.get(MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY))
        if config is None:
            return ErrorData(code=INVALID_REQUEST, message="This client has no model configured to sample with.")
        if not params.messages:
            return ErrorData(code=INVALID_PARAMS, message="A sampling request must carry at least one message.")
        sampling_context = ConversationContext(
            stream=MemoryStream(),
            prompt=[params.system_prompt] if params.system_prompt else [],
        )
        response = await config.create()(
            _to_events(params.messages),
            context=sampling_context,
            tools=(),
            response_schema=None,
            serializer=PydanticSerializer(
                pydantic_config={"arbitrary_types_allowed": True},
                use_fastdepends_errors=False,
            ),
        )
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=response.content or ""),
            model=response.model or _model_name(config),
            stop_reason=response.finish_reason,
        )

    async def on_roots(self, context: ClientRequestContext) -> ListRootsResult | ErrorData:
        """Report the directories the operator named."""
        if not self._policy.roots:
            return ErrorData(code=INVALID_REQUEST, message="This client reports no roots.")
        return ListRootsResult(
            roots=[Root(uri=Path(p).resolve().as_uri(), name=Path(p).name or None) for p in self._policy.roots]
        )


def _sole_field(requested_schema: Any) -> str | None:
    """The name of the form's single property, or ``None`` when it has anything else.

    Prefers AG2's own ``answer`` field so a served AG2 agent's question is
    answered on the property it asked about, and otherwise takes whichever single
    property the form names.
    """
    properties = requested_schema.get("properties") if isinstance(requested_schema, dict) else None
    if not isinstance(properties, dict) or len(properties) != 1:
        return None
    (name,) = properties
    return _AG2_ANSWER_FIELD if name == _AG2_ANSWER_FIELD else str(name)


def _to_events(messages: Sequence[SamplingMessage]) -> list[BaseEvent]:
    """The MCP sampling conversation as the AG2 events a model client converts.

    A user turn becomes a ``ModelRequest`` (which carries every content kind, so
    an image or audio block survives), an assistant turn a ``ModelMessage``.
    """
    events: list[BaseEvent] = []
    for message in messages:
        parts = _to_inputs(message)
        if message.role == "user":
            events.append(ModelRequest(parts))
        else:
            events.append(ModelMessage("".join(p.content for p in parts if isinstance(p, TextInput))))
    return events


def _to_inputs(message: SamplingMessage) -> list[Input]:
    blocks = message.content if isinstance(message.content, list) else [message.content]
    inputs: list[Input] = []
    for block in blocks:
        if isinstance(block, TextContent):
            inputs.append(TextInput(content=block.text))
        elif isinstance(block, ImageContent):
            inputs.append(
                BinaryInput(data=base64.b64decode(block.data), media_type=block.mime_type, kind=BinaryType.IMAGE)
            )
        elif isinstance(block, AudioContent):
            inputs.append(
                BinaryInput(data=base64.b64decode(block.data), media_type=block.mime_type, kind=BinaryType.AUDIO)
            )
        else:
            # A tool-use block from a sampling-with-tools flow, which this client
            # does not declare. Preserved as text rather than dropped.
            inputs.append(TextInput(content=block.model_dump_json(exclude_none=True)))
    return inputs


def _model_name(config: "ModelConfig") -> str:
    """The model name to report back, which the result requires.

    ``ModelConfig.model`` is allowed to raise for a config that names none (test
    doubles do), and the wire field is not optional, so this reports what it can.
    """
    with contextlib.suppress(NotImplementedError):
        return config.model
    return "unknown"


__all__ = ("AnswerPolicy", "InputRequestAnswerer")
