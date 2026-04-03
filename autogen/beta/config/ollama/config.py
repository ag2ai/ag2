# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .ollama_client import OLLAMA_DEFAULT_HOST, CreateOptions, OllamaClient


class OllamaConfigOverrides(TypedDict, total=False):
    model: str
    host: str
    temperature: float | None
    top_p: float | None
    streaming: bool
    max_tokens: int | None
    stop: str | list[str] | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None


@dataclass(slots=True)
class OllamaConfig(ModelConfig):
    model: str
    host: str = OLLAMA_DEFAULT_HOST
    temperature: float | None = None
    top_p: float | None = None
    streaming: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def copy(self, /, **overrides: Unpack[OllamaConfigOverrides]) -> "OllamaConfig":
        return replace(self, **overrides)

    def create(self) -> OllamaClient:
        options = CreateOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        return OllamaClient(
            model=self.model,
            host=self.host,
            streaming=self.streaming,
            create_options=options,
        )
