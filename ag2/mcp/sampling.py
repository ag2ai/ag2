# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Run a served agent's own reasoning on the *calling client's* model.

Cost, capability and reproducibility all move to the caller, so this is off
unless ``MCPServer(..., client_model=True)`` says otherwise — and a turn needing
tools or a response schema refuses rather than losing them, since this channel
carries neither.

The request travels exactly as a question does: a standalone
``sampling/createMessage`` up to 2025-11-25, and from 2026-07-28 back as the
call's result, over the same :class:`~ag2.mcp.pause.SuspendedTurn`.
"""

import base64
import logging
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


# The generation bound sent with each borrowed request, and the only generation
# parameter sent: the rest belong to a model configuration, and a borrowed model
# has none to take them from.
CLIENT_MODEL_MAX_TOKENS = 4096


def client_can_sample(session: "ServerSession") -> bool:
    """Whether this client said it can run a completion for the server."""
    capabilities: ClientCapabilities | None = session.client_capabilities
    return capabilities is not None and capabilities.sampling is not None


class ClientModelConfig(ModelConfig):
    """A :class:`~ag2.config.ModelConfig` whose completions run on the MCP peer.

    Built per turn: the request context it holds belongs to one call.
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
        # Not known until a completion names one, and it may name another next
        # time.
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
            MCPSamplingRefusedError: The agent needs tools or a structured
                response, which this channel cannot carry, or the peer answered
                with something other than a usable completion.
        """
        if any(True for _ in tools):
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
            answered = await self._suspended.ask_for(request, CreateMessageResult)
            if answered is None:
                raise MCPSamplingRefusedError(
                    "the calling MCP client answered a completion request with something that is not a completion"
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
    """Narrow the SDK's result union; the tools arm is unreachable here."""
    if isinstance(result, CreateMessageResult):
        return result
    raise MCPSamplingRefusedError(f"the calling MCP client answered with {type(result).__name__}")


def to_sampling_messages(messages: "Sequence[BaseEvent]") -> list[SamplingMessage]:
    """Render the agent's conversation as the sampling messages the peer receives.

    Text, images and audio under the protocol's two roles; anything else is
    logged and dropped rather than guessed at.
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
    """The completion's text, or refuse because there is none.

    Raises:
        MCPSamplingRefusedError: The peer's completion carries no text block.
    """
    content = result.content
    blocks = content if isinstance(content, list) else [content]
    spoken = [block for block in blocks if isinstance(block, TextContent)]
    if not spoken:
        # No text *block* — an empty text block is a model allowed to say nothing.
        raise MCPSamplingRefusedError(
            "the calling MCP client's completion carried no text, only "
            + ", ".join(sorted({block.type for block in blocks}))
        )
    return "".join(block.text for block in spoken)


__all__ = (
    "UNKNOWN_MODEL",
    "ClientModelClient",
    "ClientModelConfig",
    "client_can_sample",
    "to_sampling_messages",
)
