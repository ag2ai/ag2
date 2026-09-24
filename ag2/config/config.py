# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from .client import LLMClient

if TYPE_CHECKING:
    from ag2.files.protocol import FilesClient


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    DASHSCOPE = "dashscope"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    XAI = "xai"
    ZAI = "zai"


class ModelConfig(Protocol):
    @property
    def provider(self) -> ModelProvider:
        raise NotImplementedError

    @property
    def model(self) -> str | None:
        raise NotImplementedError

    def copy(self) -> Self: ...

    def create(self) -> LLMClient: ...

    def create_files_client(self) -> "FilesClient":
        # A default a config without a Files client inherits, not an abstract member.
        # The checker reads a protocol method whose body only raises as abstract, so
        # the message is bound first.
        message = f"{type(self).__name__} does not support Files API."
        raise NotImplementedError(message)
