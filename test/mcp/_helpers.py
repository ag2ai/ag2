# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool
from pydantic import BaseModel
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig


class Weather(BaseModel):
    """The stock structured reply these tests hand an agent as its response schema."""

    city: str
    temp_c: float


def text_of(result: CallToolResult) -> str:
    """The text carried by a result's first content block."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def tool_named(result: ListToolsResult, name: str) -> MCPTool:
    """The advertised tool called ``name``.

    By name and not by index: the conversational tool shares the listing, so a
    positional read passes just as happily against the wrong tool.
    """
    return next(t for t in result.tools if t.name == name)


class ChunkConfig(ModelConfig):
    """Test config whose client streams ``ModelMessageChunk`` events before the final reply.

    Used to exercise the executor's progress / log forwarding. The final body
    defaults to the concatenation of the chunks.
    """

    def __init__(self, *chunks: str, final: str | None = None) -> None:
        self._chunks = chunks
        self._final = final if final is not None else "".join(chunks)

    def copy(self) -> Self:
        return self

    def create(self) -> "ChunkClient":
        return ChunkClient(self._chunks, self._final)

    def create_files_client(self) -> None:
        raise NotImplementedError


class ChunkClient(LLMClient):
    def __init__(self, chunks: Sequence[str], final: str) -> None:
        self._chunks = chunks
        self._final = final

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        for chunk in self._chunks:
            await context.send(ModelMessageChunk(chunk))
        message = ModelMessage(self._final)
        await context.send(message)
        return ModelResponse(message=message)


def make_agent(
    *, name: str = "test-agent", prompt: str = "", config: ModelConfig | None = None, **kwargs: Any
) -> Agent:
    """An agent for server-side tests.

    ``config`` defaults to a scripted one-line reply, for the many tests where
    what the model says never enters the assertion.
    """
    return Agent(name, prompt, config=config if config is not None else TestConfig("hi"), **kwargs)


class RecordingConfig(ModelConfig):
    """Records the whole message list the framework sends the LLM on each turn.

    ``TrackingConfig`` keeps only each turn's last message, but a conversation is
    exactly what accumulates *before* it — so continuity tests read the full list
    through :attr:`prompts`.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.calls: list[list[BaseEvent]] = []

    @property
    def prompts(self) -> list[list[str]]:
        """The text inputs replayed on each turn: one list per turn, in order."""
        return [
            [
                part.content
                for m in call
                if isinstance(m, ModelRequest)
                for part in m.parts
                if isinstance(part, TextInput)
            ]
            for call in self.calls
        ]

    def copy(self) -> Self:
        return self

    def create(self) -> "RecordingClient":
        return RecordingClient(self.config.create(), self.calls)

    def create_files_client(self) -> None:
        raise NotImplementedError


class RecordingClient(LLMClient):
    def __init__(self, client: LLMClient, sink: list[list[BaseEvent]]) -> None:
        self.client = client
        self.sink = sink

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.sink.append(list(messages))
        return await self.client(messages, context=context, **kwargs)
