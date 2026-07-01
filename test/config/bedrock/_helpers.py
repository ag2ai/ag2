# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock


def make_converse_response(
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canned Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content if content is not None else [{"text": "ok"}],
            },
        },
        "stopReason": stop_reason,
        "usage": usage if usage is not None else {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


class FakeBedrockRuntime:
    """bedrock-runtime client stand-in capturing converse kwargs."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        stream_events: Iterable[dict[str, Any]] = (),
    ) -> None:
        self.response = response if response is not None else make_converse_response()
        self.stream_events = list(stream_events)
        self.converse_kwargs: dict[str, Any] | None = None
        self.converse_stream_kwargs: dict[str, Any] | None = None

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_kwargs = kwargs
        return self.response

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_stream_kwargs = kwargs
        return {"stream": iter(self.stream_events)}


class StubSession:
    """boto3.Session stand-in — the injection seam for unit tests."""

    def __init__(self, client: FakeBedrockRuntime) -> None:
        self._client = client
        self.client_args: tuple[str, dict[str, Any]] | None = None

    def client(self, service_name: str, **kwargs: Any) -> FakeBedrockRuntime:
        self.client_args = (service_name, kwargs)
        return self._client


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
