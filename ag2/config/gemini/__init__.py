# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency

from .config import GeminiConfig, VertexAIConfig
from .decisions_errors import DecisionBlockedError, UnsupportedResponseSchemaError
from .gemini_client import GeminiClient

try:
    from .decisions_config import GeminiDecisionsConfig
    from .gemini_decisions_client import GeminiDecisionsClient
except ImportError as e:
    # The Decisions API arrived in google-genai 3.2, above the `ag2[gemini]` floor.
    GeminiDecisionsConfig = missing_additional_dependency("GeminiDecisionsConfig", "google-genai>=3.2,<4", e)  # type: ignore[misc]
    GeminiDecisionsClient = missing_additional_dependency("GeminiDecisionsClient", "google-genai>=3.2,<4", e)  # type: ignore[misc]

__all__ = (
    "DecisionBlockedError",
    "GeminiClient",
    "GeminiConfig",
    "GeminiDecisionsClient",
    "GeminiDecisionsConfig",
    "UnsupportedResponseSchemaError",
    "VertexAIConfig",
)
