# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from typing_extensions import Self

from autogen.beta import Agent, Context
from autogen.beta.config.client import LLMClient
from autogen.beta.config.config import ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse


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


def make_agent(*, name: str = "test-agent", prompt: str = "", config: ModelConfig, **kwargs: Any) -> Agent:
    return Agent(name, prompt, config=config, **kwargs)
