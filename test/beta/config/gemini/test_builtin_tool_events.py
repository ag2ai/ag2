# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import types

from autogen.beta import Context, MemoryStream, ToolResult
from autogen.beta.config.gemini.events import (
    GeminiServerToolCallEvent,
    GeminiServerToolResultEvent,
)
from autogen.beta.config.gemini.gemini_client import GeminiClient
from autogen.beta.config.gemini.mappers import grounding_tool_name
from autogen.beta.context import ConversationContext
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


def _candidate(parts: list, grounding_metadata=None) -> SimpleNamespace:
    return SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=None,
        grounding_metadata=grounding_metadata,
    )


def _response(candidates: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(candidates=candidates, usage_metadata=None)


@pytest.fixture
def client() -> GeminiClient:
    with patch("autogen.beta.config.gemini.gemini_client.genai.Client"):
        return GeminiClient(model="gemini-2.5-flash", api_key="test-key")


@pytest.fixture
def memory_context() -> tuple[Context, MemoryStream]:
    stream = MemoryStream()
    return ConversationContext(stream=stream), stream


class TestFactoryFromExecutableCode:
    def test_returns_event_for_executable_code_part(self) -> None:
        part = types.Part(executable_code=types.ExecutableCode(code="print(1)", language="PYTHON"))

        event = GeminiServerToolCallEvent.from_executable_code(part)

        assert event == GeminiServerToolCallEvent(
            id=event.id,
            name=CODE_EXECUTION_TOOL_NAME,
            arguments='{"code": "print(1)", "language": "PYTHON"}',
            part=part,
        )

    def test_returns_none_for_non_code_part(self) -> None:
        assert GeminiServerToolCallEvent.from_executable_code(types.Part(text="hello")) is None


class TestFactoryFromCodeExecutionResult:
    def test_returns_event_for_result_part(self) -> None:
        part = types.Part(code_execution_result=types.CodeExecutionResult(outcome="OUTCOME_OK", output="42"))

        event = GeminiServerToolResultEvent.from_code_execution_result(part, parent_id="call-1")

        assert event == GeminiServerToolResultEvent(
            parent_id="call-1",
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(),
            part=part,
        )

    def test_returns_none_for_non_result_part(self) -> None:
        assert GeminiServerToolResultEvent.from_code_execution_result(types.Part(text="hello"), parent_id="x") is None


class TestFactoryFromGrounding:
    def test_call_carries_queries_in_arguments(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["bitcoin price"])

        event = GeminiServerToolCallEvent.from_grounding(gm, name=WEB_SEARCH_TOOL_NAME)

        assert event == GeminiServerToolCallEvent(
            id=event.id,
            name=WEB_SEARCH_TOOL_NAME,
            arguments='{"queries": ["bitcoin price"]}',
            grounding_metadata=gm,
        )

    def test_result_links_to_call_via_parent_id(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["x"])

        event = GeminiServerToolResultEvent.from_grounding(gm, parent_id="call-1", name=WEB_SEARCH_TOOL_NAME)

        assert event == GeminiServerToolResultEvent(
            parent_id="call-1",
            name=WEB_SEARCH_TOOL_NAME,
            result=ToolResult(),
            grounding_metadata=gm,
        )


class TestGroundingToolName:
    def test_web_search_when_queries_present(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["bitcoin"])

        assert grounding_tool_name(gm) == WEB_SEARCH_TOOL_NAME

    def test_web_fetch_when_no_queries(self) -> None:
        gm = types.GroundingMetadata()

        assert grounding_tool_name(gm) == WEB_FETCH_TOOL_NAME


@pytest.mark.asyncio
class TestProcessResponseEmitsBuiltinEvents:
    async def test_code_execution_pair(
        self, client: GeminiClient, memory_context: tuple[Context, MemoryStream]
    ) -> None:
        ctx, stream = memory_context
        code_part = types.Part(executable_code=types.ExecutableCode(code="print(1)", language="PYTHON"))
        result_part = types.Part(code_execution_result=types.CodeExecutionResult(outcome="OUTCOME_OK", output="1\n"))
        response = _response([_candidate([code_part, result_part])])

        await client._process_response(response, ctx)

        events = list(await stream.history.get_events())
        [call_event, result_event] = events
        assert events == [
            GeminiServerToolCallEvent(
                id=call_event.id,
                name=CODE_EXECUTION_TOOL_NAME,
                arguments='{"code": "print(1)", "language": "PYTHON"}',
                part=code_part,
            ),
            GeminiServerToolResultEvent(
                parent_id=call_event.id,
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(),
                part=result_part,
            ),
        ]

    async def test_grounding_synthesises_call_result_pair(
        self, client: GeminiClient, memory_context: tuple[Context, MemoryStream]
    ) -> None:
        ctx, stream = memory_context
        gm = types.GroundingMetadata(web_search_queries=["bitcoin price"])
        response = _response([_candidate([types.Part(text="It is $X")], grounding_metadata=gm)])

        await client._process_response(response, ctx)

        events = list(await stream.history.get_events())
        [call_event, _] = events
        assert events == [
            GeminiServerToolCallEvent(
                id=call_event.id,
                name=WEB_SEARCH_TOOL_NAME,
                arguments='{"queries": ["bitcoin price"]}',
                grounding_metadata=gm,
            ),
            GeminiServerToolResultEvent(
                parent_id=call_event.id,
                name=WEB_SEARCH_TOOL_NAME,
                result=ToolResult(),
                grounding_metadata=gm,
            ),
        ]

    async def test_url_context_uses_web_fetch_name(
        self, client: GeminiClient, memory_context: tuple[Context, MemoryStream]
    ) -> None:
        ctx, stream = memory_context
        gm = types.GroundingMetadata()
        response = _response([_candidate([types.Part(text="H1: Example")], grounding_metadata=gm)])

        await client._process_response(response, ctx)

        events = list(await stream.history.get_events())
        [call_event, _] = events
        assert events == [
            GeminiServerToolCallEvent(
                id=call_event.id,
                name=WEB_FETCH_TOOL_NAME,
                arguments='{"queries": []}',
                grounding_metadata=gm,
            ),
            GeminiServerToolResultEvent(
                parent_id=call_event.id,
                name=WEB_FETCH_TOOL_NAME,
                result=ToolResult(),
                grounding_metadata=gm,
            ),
        ]


@pytest.mark.asyncio
class TestProcessStreamEmitsBuiltinEvents:
    async def test_code_execution_pair_across_chunks(
        self, client: GeminiClient, memory_context: tuple[Context, MemoryStream]
    ) -> None:
        ctx, stream = memory_context
        code_part = types.Part(executable_code=types.ExecutableCode(code="print(2)", language="PYTHON"))
        result_part = types.Part(code_execution_result=types.CodeExecutionResult(outcome="OUTCOME_OK", output="2\n"))

        async def chunks():
            yield _response([_candidate([code_part])])
            yield _response([_candidate([result_part])])

        await client._process_stream(chunks(), ctx)

        events = list(await stream.history.get_events())
        [call_event, _] = events
        assert events == [
            GeminiServerToolCallEvent(
                id=call_event.id,
                name=CODE_EXECUTION_TOOL_NAME,
                arguments='{"code": "print(2)", "language": "PYTHON"}',
                part=code_part,
            ),
            GeminiServerToolResultEvent(
                parent_id=call_event.id,
                name=CODE_EXECUTION_TOOL_NAME,
                result=ToolResult(),
                part=result_part,
            ),
        ]

    async def test_grounding_emitted_once_after_stream_completes(
        self, client: GeminiClient, memory_context: tuple[Context, MemoryStream]
    ) -> None:
        ctx, stream = memory_context
        gm = types.GroundingMetadata(web_search_queries=["x"])

        async def chunks():
            yield _response([_candidate([types.Part(text="part 1 ")], grounding_metadata=gm)])
            yield _response([_candidate([types.Part(text="part 2")], grounding_metadata=gm)])

        await client._process_stream(chunks(), ctx)

        events = list(await stream.history.get_events())
        [call_event, _] = events
        assert events == [
            GeminiServerToolCallEvent(
                id=call_event.id,
                name=WEB_SEARCH_TOOL_NAME,
                arguments='{"queries": ["x"]}',
                grounding_metadata=gm,
            ),
            GeminiServerToolResultEvent(
                parent_id=call_event.id,
                name=WEB_SEARCH_TOOL_NAME,
                result=ToolResult(),
                grounding_metadata=gm,
            ),
        ]
