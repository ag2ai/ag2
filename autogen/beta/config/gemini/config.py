# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .gemini_client import CreateConfig, GeminiClient


class GeminiConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_output_tokens: int | None
    stop_sequences: list[str] | None
    streaming: bool
    presence_penalty: float | None
    frequency_penalty: float | None
    seed: int | None


@dataclass(slots=True)
class GeminiConfig(ModelConfig):
    model: str
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    streaming: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None

    def copy(self, /, **overrides: Unpack[GeminiConfigOverrides]) -> "GeminiConfig":
        return replace(self, **overrides)

    def create(self) -> GeminiClient:
        config = CreateConfig()

        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences is not None:
            config["stop_sequences"] = self.stop_sequences
        if self.presence_penalty is not None:
            config["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            config["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            config["seed"] = self.seed

        return GeminiClient(
            model=self.model,
            api_key=self.api_key,
            streaming=self.streaming,
            create_config=config,
        )
