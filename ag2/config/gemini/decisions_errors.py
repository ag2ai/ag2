# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Errors of `GeminiDecisionsConfig`, importable whatever google-genai version is installed."""

from ag2.exceptions import AG2Error
from ag2.response import NotADecisionError


class UnsupportedResponseSchemaError(NotADecisionError):
    """Raised when a ``response_schema`` cannot be expressed as a Gemini Decisions question."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"{reason} The Gemini Decisions API is decision-only: use `bool`, an `Enum` of strings, "
            "an `IntEnum` numbered from 0, a `0..1` number schema from `ResponseSchema.from_schema`, "
            "or a flat class whose required fields are each one of those as `response_schema`."
        )


class DecisionBlockedError(AG2Error):
    """Raised when Gemini's input filters block the request, so no question is answered."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        self.message = message
        detail = f": {message}" if message else ""
        super().__init__(f"The Gemini Decisions API blocked the input ({reason}){detail}.")
