# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Gemini parallel-tool-call id collisions.

Vertex/Gemini does not always populate ``fc.id`` on returned function calls.
Ensure that parallel tool call usage still results in unique ids.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from autogen.beta import Context
from autogen.beta.config.gemini.gemini_client import GeminiClient


def _part(*, function_call=None, text=None, thought=None, thought_signature=None) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=function_call,
        text=text,
        thought=thought,
        thought_signature=thought_signature,
    )


def _function_call(name: str, args: dict, fc_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=fc_id, name=name, args=args)


def _response(parts: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=None)],
        usage_metadata=None,
    )


@pytest.fixture
def client() -> GeminiClient:
    with patch("autogen.beta.config.gemini.gemini_client.genai.Client"):
        return GeminiClient(model="gemini-2.5-flash", api_key="test-key")


@pytest.fixture
def context() -> Context:
    return Context(stream=AsyncMock())


@pytest.mark.asyncio
class TestProcessResponseToolCallIds:
    async def test_parallel_calls_to_same_tool_get_unique_ids(self, client: GeminiClient, context: Context) -> None:
        """Two parallel calls to the same tool with no provider id must
        receive distinct ids so the executor can route their results.
        """
        response = _response([
            _part(function_call=_function_call("query_dossier", {"source": "gps"})),
            _part(function_call=_function_call("query_dossier", {"source": "smart_home"})),
        ])

        result = await client._process_response(response, context)

        ids = [c.id for c in result.tool_calls.calls]
        assert len(ids) == len(set(ids)), f"expected unique ids, got {ids}"

    async def test_provider_supplied_id_is_preserved(self, client: GeminiClient, context: Context) -> None:
        response = _response([
            _part(function_call=_function_call("query_dossier", {"source": "gps"}, fc_id="provider-id-1")),
        ])

        result = await client._process_response(response, context)

        assert [c.id for c in result.tool_calls.calls] == ["provider-id-1"]


@pytest.mark.asyncio
class TestProcessStreamToolCallIds:
    async def test_parallel_calls_to_same_tool_get_unique_ids(self, client: GeminiClient, context: Context) -> None:
        chunk = _response([
            _part(function_call=_function_call("query_dossier", {"source": "gps"})),
            _part(function_call=_function_call("query_dossier", {"source": "smart_home"})),
        ])

        async def stream():
            yield chunk

        result = await client._process_stream(stream(), context)

        ids = [c.id for c in result.tool_calls.calls]
        assert len(ids) == len(set(ids)), f"expected unique ids, got {ids}"

    async def test_provider_supplied_id_is_preserved(self, client: GeminiClient, context: Context) -> None:
        chunk = _response([
            _part(function_call=_function_call("query_dossier", {"source": "gps"}, fc_id="provider-id-1")),
        ])

        async def stream():
            yield chunk

        result = await client._process_stream(stream(), context)

        assert [c.id for c in result.tool_calls.calls] == ["provider-id-1"]
