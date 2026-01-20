# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Gemini V2 Client for AG2.

This client is a wrapper around the Gemini V2 API.
"""

from typing import Any, Literal

from pydantic import Field
from autogen.llm_config.client import ModelClient
from autogen.llm_config.entry import LLMConfigEntry, LLMConfigEntryDict
from .client_v2 import ModelClientV2

class GeminiV2EntryDict(LLMConfigEntryDict, total=False):
    """Entry dict for Gemini V2 client configuration."""

    api_type: Literal["gemini_v2"]
    project_id: str | None
    location: str | None
    google_application_credentials: str | None
    credentials: Any | str | None
    stream: bool
    safety_settings: list[dict[str, Any]] | dict[str, Any] | None
    price: list[float] | None
    tool_config: dict[str, Any] | None
    proxy: str | None
    include_thoughts: bool | None
    thinking_budget: int | None
    thinking_level: Literal["High", "Medium", "Low", "Minimal"] | None


class GeminiV2LLMConfigEntry(LLMConfigEntry):
    """LLM config entry for Gemini V2 client."""

    api_type: Literal["gemini_v2"] = "gemini_v2"
    project_id: str | None = None
    location: str | None = None
    google_application_credentials: str | None = None
    credentials: Any | str | None = None
    stream: bool = False
    safety_settings: list[dict[str, Any]] | dict[str, Any] | None = None
    price: list[float] | None = Field(default=None, min_length=2, max_length=2)
    tool_config: dict[str, Any] | None = None
    proxy: str | None = None
    include_thoughts: bool | None = None
    thinking_budget: int | None = None
    thinking_level: Literal["High", "Medium", "Low", "Minimal"] | None = None

    def create_client(self) -> ModelClient:  # pragma: no cover
        """Create GeminiV2Client instance."""
        raise NotImplementedError("GeminiV2LLMConfigEntry.create_client() is not implemented.")



class GeminiV2Client(ModelClientV2):
    """Gemini V2 Client for AG2."""
    pass