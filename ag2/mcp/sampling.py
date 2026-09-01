# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Run a served agent's own reasoning on the *calling client's* model.

MCP lets a server ask its client for an LLM completion. For a served AG2 agent
that turns into a deployment property worth stating plainly: the agent runs, but
the thinking is bought, chosen and observed by whoever called it.

* **Cost** moves to the caller. Every turn the agent takes spends their budget.
* **Capability** becomes theirs. Which model answers, and how good it is, is a
  fact about the client, not about this deployment — so the same agent gives
  different answers to different callers, and a served agent's quality is no
  longer something its operator controls.
* **Reproducibility** goes with it. A trace cannot be re-run against a known
  model, because the model was the peer's.

None of that is a reason not to do it — a deployment with no credentials of its
own can serve an agent that needs one, which is the whole point — but it is a
decision, so it is off unless a :class:`ClientModel` is passed, and never turned
on because the transport happens to allow it.

The mechanics are small: :class:`~ag2.config.LLMClient` is a one-method protocol,
and the two protocol eras differ here exactly as they do for a question — a
standalone ``sampling/createMessage`` up to 2025-11-25, and from 2026-07-28 the
request comes back as the result of the call, resumed by the client's retry. The
same :class:`~ag2.mcp.pause.SuspendedTurn` carries both.
"""

import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_depends.library.serializer import SerializerProto
from mcp.types import (
    AudioContent,
    ClientCapabilities,
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    ImageContent,
    SamplingMessage,
    TextContent,
)
from typing_extensions import Self

from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig, ModelProvider
from ag2.context import ConversationContext
from ag2.events import BaseEvent, BinaryInput, BinaryType, ModelMessage, ModelRequest, ModelResponse, TextInput
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .errors import MCPSamplingRefusedError
from .pause import SuspendedTurn

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mcp.server.context import ServerRequestContext
    from mcp.server.session import ServerSession

    from ag2.files.protocol import FilesClient

logger = logging.getLogger(__name__)

# What ``CreateMessageResult.model`` is reported as when a client returns none.
UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True, slots=True)
class ClientModel:
    """Serve an agent whose model is the calling client's.

    Passing one to :class:`~ag2.mcp.MCPServer` is what enables this; the default
    is ``None``, and a server that has not enabled it never sends a sampling
    request and never needs the capability of its clients.

    Read :mod:`ag2.mcp.sampling` before enabling it: the caller pays for every
    turn, the model that answers is theirs rather than yours, and a trace cannot
    be re-run against a known model afterwards.

    Attributes:
        max_tokens: The generation bound sent with each request; the protocol
            requires one. It is the only generation parameter sent — temperature
            and stop sequences belong to a model configuration, and here there is
            no configuration to take them from.
        fallback: What to do about a client that advertised no sampling
            capability. ``False`` (the default) fails the turn, so a deployment
            that has no model of its own says so rather than answering by some
            other means. ``True`` falls back to the agent's own configured model,
            which needs one — a deployment holding credentials it prefers not to
            spend.
    """

    max_tokens: int = 4096
    fallback: bool = False


def client_can_sample(session: "ServerSession") -> bool:
    """Whether this client said it can run a completion for the server."""
    capabilities: ClientCapabilities | None = session.client_capabilities
    return capabilities is not None and capabilities.sampling is not None


class ClientModelConfig(ModelConfig):
    """A :class:`~ag2.config.ModelConfig` whose completions run on the MCP peer.

    Built per turn by the serving path, because what it needs — the live request
    context, and on the modern era the suspended run to ask through — belongs to
    one call.
    """

    __slots__ = ("_request_context", "_suspended", "_max_tokens")

    def __init__(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        *,
        suspended: SuspendedTurn | None,
        max_tokens: int,
    ) -> None:
        self._request_context = request_context
        self._suspended = suspended
        self._max_tokens = max_tokens

    @property
    def provider(self) -> ModelProvider:
        raise NotImplementedError("The calling MCP client's model has no AG2 provider.")

    @property
    def model(self) -> str:
        # Not known until a completion comes back and names one, and it may name
        # a different one next time.
        return "mcp-client"

    def copy(self) -> Self:
        return self

    def create(self) -> "ClientModelClient":
        return ClientModelClient(
            self._request_context,
            suspended=self._suspended,
            max_tokens=self._max_tokens,
        )

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError("The calling MCP client's model does not serve files.")


class ClientModelClient(LLMClient):
    """One completion, run by the calling MCP client on the agent's behalf."""

    __slots__ = ("_request_context", "_suspended", "_max_tokens")

    def __init__(
        self,
        request_context: "ServerRequestContext[Any, Any]",
        *,
        suspended: SuspendedTurn | None,
        max_tokens: int,
    ) -> None:
        self._request_context = request_context
        self._suspended = suspended
        self._max_tokens = max_tokens

    async def __call__(
        self,
        messages: "Sequence[BaseEvent]",
        context: ConversationContext,
        *,
        tools: "Iterable[ToolSchema]" = (),
        response_schema: ResponseProto | None = None,
        serializer: SerializerProto | None = None,
    ) -> ModelResponse:
        """Ask the peer to complete this conversation, and read the answer back.

        Raises:
            MCPSamplingRefusedError: The agent needs something of a model this
                channel cannot carry — tools, or a structured response — or the
                peer answered with something other than a completion.
        """
        if any(True for _ in tools):
            # Sampling with tools needs a capability of its own, and an agent
            # whose tools silently vanished would answer as though it had none.
            raise MCPSamplingRefusedError(
                "a served agent with tools cannot borrow the calling client's model: "
                "the sampling request carries no tools this deployment can offer"
            )
        if response_schema is not None:
            raise MCPSamplingRefusedError(
                "a served agent with a response schema cannot borrow the calling client's model: "
                "sampling returns free text, which nothing here could validate against the schema"
            )
        request = CreateMessageRequest(
            params=CreateMessageRequestParams(
                messages=to_sampling_messages(messages),
                systemPrompt="\n".join(context.prompt) or None,
                maxTokens=self._max_tokens,
            )
        )
        if self._suspended is not None:
            answered = await self._suspended.ask(request)
            if not isinstance(answered, CreateMessageResult):
                raise MCPSamplingRefusedError(
                    f"the calling MCP client answered a completion request with {type(answered).__name__}"
                )
            result = answered
        else:
            assert request.params is not None
            result = _as_completion(
                await self._request_context.session.create_message(
                    request.params.messages,
                    max_tokens=request.params.max_tokens,
                    system_prompt=request.params.system_prompt,
                    related_request_id=self._request_context.request_id,
                )
            )
        message = ModelMessage(_text_of(result))
        # Published like any other model reply, so a served turn borrowing a
        # model streams and records the same way one on its own model does.
        await context.send(message)
        return ModelResponse(
            message=message,
            model=result.model or UNKNOWN_MODEL,
            finish_reason=result.stop_reason,
        )


def _as_completion(result: Any) -> CreateMessageResult:
    if isinstance(result, CreateMessageResult):
        return result
    # ``CreateMessageResultWithTools`` — only reachable if tools were sent, and
    # they never are here.
    raise MCPSamplingRefusedError(f"the calling MCP client answered with {type(result).__name__}")


def to_sampling_messages(messages: "Sequence[BaseEvent]") -> list[SamplingMessage]:
    """Render the agent's conversation as the sampling messages the peer receives.

    Deliberately narrow: text, images and audio, under the two roles the protocol
    has. Tool traffic has no rendering here and cannot occur — a turn on this
    model refuses tools before it starts — and anything else is dropped rather
    than guessed at, which is logged.
    """
    rendered: list[SamplingMessage] = []
    for message in messages:
        if isinstance(message, ModelMessage):
            rendered.append(SamplingMessage(role="assistant", content=TextContent(type="text", text=message.content)))
        elif isinstance(message, ModelRequest):
            rendered.extend(SamplingMessage(role="user", content=block) for block in _blocks(message))
        else:
            logger.debug("dropping %s from a sampling request: no rendering for it", type(message).__name__)
    return rendered


def _blocks(request: ModelRequest) -> "list[TextContent | ImageContent | AudioContent]":
    blocks: list[TextContent | ImageContent | AudioContent] = []
    for part in request.parts:
        if isinstance(part, TextInput):
            blocks.append(TextContent(type="text", text=part.content))
        elif isinstance(part, BinaryInput) and part.kind is BinaryType.IMAGE:
            blocks.append(ImageContent(type="image", data=_b64(part.data), mimeType=part.media_type))
        elif isinstance(part, BinaryInput) and part.kind is BinaryType.AUDIO:
            blocks.append(AudioContent(type="audio", data=_b64(part.data), mimeType=part.media_type))
        else:
            logger.debug("dropping a %s part from a sampling request", type(part).__name__)
    return blocks


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _text_of(result: CreateMessageResult) -> str:
    content = result.content
    blocks = content if isinstance(content, list) else [content]
    return "".join(block.text for block in blocks if isinstance(block, TextContent))


__all__ = (
    "UNKNOWN_MODEL",
    "ClientModel",
    "ClientModelClient",
    "ClientModelConfig",
    "client_can_sample",
    "to_sampling_messages",
)
