# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A model config without a Files client can be built under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2.config import BedrockConfig, DashScopeConfig, OllamaConfig, VertexAIConfig
from ag2.config.config import ModelConfig, ModelProvider
from ag2.testing import TestClient

# `ModelConfig.create_files_client` is a default every config inherits, not a member
# each one must implement; these configs rely on it.
BedrockConfig(model="anthropic.claude-sonnet-5")
DashScopeConfig(model="qwen-max")
OllamaConfig(model="llama3")
VertexAIConfig(model="gemini-2.5-flash")


class OwnConfig(ModelConfig):
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def model(self) -> str:
        return "own"

    def copy(self) -> "OwnConfig":
        return self

    def create(self) -> TestClient:
        return TestClient()


OwnConfig()
