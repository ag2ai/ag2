# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: Gemini emits ``Part(text="")`` at function-call boundaries.

Empty parts must not produce ``ModelMessage`` or ``ModelMessageChunk`` events —
they crash AG-UI's ``TextMessageContentEvent`` validator (which enforces
``MinLen(1)`` on ``delta``) and carry no payload for any other subscriber.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import types

from autogen.beta import Context, MemoryStream
from autogen.beta.config.gemini.gemini_client import GeminiClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelReasoning


def _candidate(parts: list) -> SimpleNamespace:
    return SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=None,
        grounding_metadata=None,
    )


def _response(candidates: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(candidates=candidates, usage_metadata=None)


@pytest.fixture
def client() -> GeminiClient:
    with patch("autogen.beta.config.gemini.gemini_client.genai.Client"):
        return GeminiClient(model="gemini-2.5-flash", api_key="test-key")


@pytest.fixture
def memory_context() -> tuple[Context, MemoryStream, list[BaseEvent]]:
    # ``ModelMessageChunk`` / ``ModelMessage`` / ``ModelReasoning`` are
    # transient and not persisted to history, so capture them via a
    # subscriber instead.
    stream = MemoryStream()
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)
    return ConversationContext(stream=stream), stream, captured


@pytest.mark.asyncio
class TestProcessStreamSkipsEmptyTextParts:
    async def test_empty_text_chunks_dropped(
        self,
        client: GeminiClient,
        memory_context: tuple[Context, MemoryStream, list[BaseEvent]],
    ) -> None:
        ctx, _, captured = memory_context

        async def chunks():
            yield _response([_candidate([types.Part(text="Hello")])])
            yield _response([_candidate([types.Part(text="")])])
            yield _response([_candidate([types.Part(text=" world")])])

        await client._process_stream(chunks(), ctx)

        message_chunks = [e for e in captured if isinstance(e, ModelMessageChunk)]
        messages = [e for e in captured if isinstance(e, ModelMessage)]

        assert message_chunks == [ModelMessageChunk("Hello"), ModelMessageChunk(" world")]
        assert messages == [ModelMessage("Hello world")]

    async def test_empty_thought_text_dropped(
        self,
        client: GeminiClient,
        memory_context: tuple[Context, MemoryStream, list[BaseEvent]],
    ) -> None:
        ctx, _, captured = memory_context

        async def chunks():
            yield _response([_candidate([types.Part(text="", thought=True)])])
            yield _response([_candidate([types.Part(text="real thought", thought=True)])])

        await client._process_stream(chunks(), ctx)

        reasoning = [e for e in captured if isinstance(e, ModelReasoning)]
        assert reasoning == [ModelReasoning("real thought")]


@pytest.mark.asyncio
class TestProcessResponseSkipsEmptyTextParts:
    async def test_empty_text_part_does_not_emit_model_message(
        self,
        client: GeminiClient,
        memory_context: tuple[Context, MemoryStream, list[BaseEvent]],
    ) -> None:
        ctx, _, captured = memory_context
        response = _response([_candidate([types.Part(text="")])])

        await client._process_response(response, ctx)

        assert [e for e in captured if isinstance(e, ModelMessage)] == []
