# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypedDict

import httpx2
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider
from ag2.exceptions import FilesAPIUnsupportedError

from .gemini_decisions_client import GeminiDecisionsClient

if TYPE_CHECKING:
    from ag2.files.protocol import FilesClient


class GeminiDecisionsConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    api_version: str
    http_client: httpx2.AsyncClient | None
    boolean_threshold: float
    descriptions: Mapping[str, str] | None
    question_instructions: str | None
    extra_body: Mapping[str, Any] | None


@dataclass(slots=True)
class GeminiDecisionsConfig(ModelConfig):
    """Gemini's Decisions API: typed questions taken from the agent's ``response_schema``, sent in one request.

    A scalar schema is one question; a flat class is one question per field.
    Gemini Developer API only.
    """

    model: str = "gemini-decision-1"
    api_key: str | None = None
    # The endpoint is served on `v1beta`; the SDK's default `v1` answers 404.
    api_version: str = "v1beta"
    http_client: httpx2.AsyncClient | None = None
    # A ``bool`` schema is ``True`` once the probability of ``true`` reaches this.
    boolean_threshold: float = 0.5
    # Option descriptions keyed by choice value or score level as a string; they
    # outrank the docstrings under ``Enum`` members. Single-question schemas only.
    descriptions: Mapping[str, str] | None = None
    # Text the question carries after the agent prompt; the schema description is still appended.
    question_instructions: str | None = None
    extra_body: Mapping[str, Any] | None = None

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.GEMINI

    def copy(self, /, **overrides: Unpack[GeminiDecisionsConfigOverrides]) -> "GeminiDecisionsConfig":
        return replace(self, **overrides)

    def create(self) -> GeminiDecisionsClient:
        return GeminiDecisionsClient(
            model=self.model,
            api_key=self.api_key,
            api_version=self.api_version,
            http_client=self.http_client,
            boolean_threshold=self.boolean_threshold,
            descriptions=self.descriptions,
            question_instructions=self.question_instructions,
            extra_body=self.extra_body,
        )

    def create_files_client(self) -> "FilesClient":
        # The endpoint takes inline data and URIs; there is no Files API client for it.
        raise FilesAPIUnsupportedError(type(self).__name__)
