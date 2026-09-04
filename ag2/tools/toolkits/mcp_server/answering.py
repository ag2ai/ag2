# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Answer a third-party MCP server's requests for input from the agent's own resources.

An *elicitation* goes to the agent's own ``context.input()``, a *sampling*
request runs on its own model, and a *roots* request is answered from directories
the operator named. All three are opt-in, and each is advertised only when
enabled — the ``mcp`` client derives its declared capabilities from which
callbacks were supplied, so not enabling one *is* how it goes unadvertised.

Refusal looks different per request type, and only one kind reaches the server.
Elicitation has a ``decline`` action, so a question this agent will not answer is
declined on the wire. Sampling and roots have no such arm: the ``ErrorData``
returned for one aborts the ``ClientSession``'s request loop, so the *tool call*
fails with that message instead. A conforming server never provokes either; the
arms exist for one that asks anyway.
"""

import base64
import binascii
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

# What AG2's own served side names its answer field; preferred so an AG2-to-AG2
# round trip answers on it, with any other single-property form answered on
# whatever property it names.
_AG2_ANSWER_FIELD = "answer"


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """What this agent answers when a third-party MCP server asks it for input.

    Every field is off by default: the toolkit reaches servers the operator may
    not control, and each hands over one of the operator's own resources.

    Attributes:
        elicitation: ``"ask"`` routes a server's question to the agent's own
            ``context.input()`` and advertises the capability; ``"decline"``
            (default) advertises nothing. Same word, values and reasoning as
            :data:`ag2.hitl.ElicitationPolicy`.
        sampling: When true, a server's ``sampling/createMessage`` runs on the
            agent's own model — the server gets an LLM without holding a key, and
            **the operator pays for it**. Off by default for that reason.
        roots: Directories reported to a server that scopes its work to them.
            With none configured the capability is not advertised.
        max_rounds: How many times a server may come back asking before the call
            is abandoned, so one that re-asks indefinitely cannot loop the agent.
    """

    elicitation: ElicitationPolicy = "decline"
    sampling: bool = False
    roots: Sequence[str | os.PathLike[str]] = ()
    max_rounds: int = 10


class InputRequestAnswerer:
    """Answers one operation's worth of a server's input requests.

    Bound to the live :class:`~ag2.context.ConversationContext`, where the
    agent's human and the agent's model both are.
    """

    __slots__ = ("_policy", "_context")

    def __init__(self, policy: AnswerPolicy, context: Context) -> None:
        self._policy = policy
        self._context = context

    def session_kwargs(self) -> dict[str, Any]:
        """The ``ClientSession`` callbacks for the enabled request types, and only those.

        Omitting one *is* how its capability goes unadvertised; there is no
        second switch to keep in step with this.
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
        the operator never made.
        """
        if self._policy.elicitation != "ask":
            return _declined()
        if params.mode != "form":
            # ``context.input()`` is a text channel and cannot confirm that an
            # out-of-band browser flow happened, so declining beats accepting on
            # a promise.
            logger.debug("declining a %r-mode MCP elicitation: no rendering for it", params.mode)
            return _declined()
        field = _sole_field(params.requested_schema)
        if field is None:
            # Splitting one free-text answer across fields is fabricating data.
            logger.debug("declining an MCP elicitation whose form is not a single property")
            return _declined()
        answer = await self._context.input(params.message)
        return ElicitResult(action="accept", content={field: answer})

    async def on_sampling(
        self, context: ClientRequestContext, params: CreateMessageRequestParams
    ) -> CreateMessageResult | ErrorData:
        """Run a server's requested completion on the agent's own model.

        One completion, not a turn of the calling agent: its tools, history and
        response schema stay out, so a borrowed model cannot reach back into the
        agent that lent it. The server's generation parameters are not forwarded
        either — the operator's configuration governs a call they pay for.
        """
        if not self._policy.sampling:
            return ErrorData(code=INVALID_REQUEST, message="This client does not lend its model for sampling.")
        # What "the agent's own model" means: the config it published for this
        # turn, under the same key the metrics middleware reads.
        config = cast("ModelConfig | None", self._context.dependencies.get(MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY))
        if config is None:
            return ErrorData(code=INVALID_REQUEST, message="This client has no model configured to sample with.")
        if not params.messages:
            return ErrorData(code=INVALID_PARAMS, message="A sampling request must carry at least one message.")
        try:
            messages = _to_events(params.messages)
        except ValueError as e:
            return ErrorData(code=INVALID_PARAMS, message=str(e))
        sampling_context = ConversationContext(
            stream=MemoryStream(),
            prompt=[params.system_prompt] if params.system_prompt else [],
        )
        response = await config.create()(
            messages,
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


def _declined() -> ElicitResult:
    """A fresh decline each time: the SDK is handed this and may hold or mutate
    it, and ``ElicitResult`` is not frozen."""
    return ElicitResult(action="decline")


def _sole_field(requested_schema: Any) -> str | None:
    """The name of the form's single property, or ``None`` when it has anything else."""
    properties = requested_schema.get("properties") if isinstance(requested_schema, dict) else None
    if not isinstance(properties, dict) or len(properties) != 1:
        return None
    (name,) = properties
    return _AG2_ANSWER_FIELD if name == _AG2_ANSWER_FIELD else str(name)


def _to_events(messages: Sequence[SamplingMessage]) -> list[BaseEvent]:
    """The MCP sampling conversation as the AG2 events a model client converts.

    A user turn becomes a ``ModelRequest``, which carries every content kind, so
    an image or audio block survives; an assistant turn becomes a ``ModelMessage``.
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
            inputs.append(BinaryInput(data=_decode(block.data), media_type=block.mime_type, kind=BinaryType.IMAGE))
        elif isinstance(block, AudioContent):
            inputs.append(BinaryInput(data=_decode(block.data), media_type=block.mime_type, kind=BinaryType.AUDIO))
        else:
            # A tool-use block from a flow this client does not declare;
            # preserved as text rather than dropped.
            inputs.append(TextInput(content=block.model_dump_json(exclude_none=True)))
    return inputs


def _decode(data: str) -> bytes:
    """The block's bytes, decoded.

    The payload is the server's, so malformed base64 is a thing one can send;
    unguarded it surfaces as a ``binascii.Error`` naming nothing.

    Raises:
        ValueError: The block's data is not base64.
    """
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error as e:
        raise ValueError(f"an MCP server sent a media block that is not base64: {e}") from e


def _model_name(config: "ModelConfig") -> str:
    """The model name to report back.

    ``ModelConfig.model`` may raise for a config naming none, and the wire field
    is not optional, so this reports what it can.
    """
    with contextlib.suppress(NotImplementedError):
        return config.model
    return "unknown"


__all__ = ("AnswerPolicy", "InputRequestAnswerer")
