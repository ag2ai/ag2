# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from ag2.events import (
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ToolCallEvent,
    Usage,
)
from test.config._helpers import make_tool
from test.config.bedrock._helpers import FakeBedrock, ask, make_converse_response


@pytest.mark.asyncio
async def test_minimal_request(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config())

    [request] = bedrock.requests
    assert request.path == "/model/m1/converse"
    assert request.body == {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}


@pytest.mark.asyncio
async def test_tools_serialized_as_tool_spec(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config(), tools=[make_tool().schema])

    assert bedrock.body == IsPartialDict({
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "search_docs",
                        "description": "Search documentation by query.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer", "minimum": 1},
                                },
                                "required": ["query"],
                            },
                        },
                    },
                }
            ],
        },
    })


@pytest.mark.asyncio
async def test_system_prompt_lands_in_system_param(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config(), prompt=["You are helpful.", "Be brief."])

    assert bedrock.body == IsPartialDict({"system": [{"text": "You are helpful.\nBe brief."}]})


@pytest.mark.asyncio
async def test_non_streaming_response(bedrock: FakeBedrock) -> None:
    bedrock.response = make_converse_response(
        content=[
            {"text": "The answer is 42."},
            {"toolUse": {"toolUseId": "tc_1", "name": "search_docs", "input": {"query": "x"}}},
        ],
        stop_reason="tool_use",
        usage={"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    )

    result, events = await ask(bedrock.config())

    assert result.content == "The answer is 42."
    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]
    assert result.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert result.model == "m1"
    assert result.provider == "bedrock"
    assert result.finish_reason == "tool_use"
    assert events == [ModelMessage("The answer is 42.")]


@pytest.mark.asyncio
async def test_non_streaming_reasoning_content(bedrock: FakeBedrock) -> None:
    bedrock.response = make_converse_response(
        content=[
            {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
            {"text": "done"},
        ],
    )

    _, events = await ask(bedrock.config())

    assert events == [ModelReasoning("thinking..."), ModelMessage("done")]


@pytest.mark.asyncio
class TestStreaming:
    async def test_goes_to_converse_stream(self, bedrock: FakeBedrock) -> None:
        await ask(bedrock.config(streaming=True))

        [request] = bedrock.requests
        assert request.path == "/model/m1/converse-stream"

    async def test_text_chunks(self, bedrock: FakeBedrock) -> None:
        bedrock.stream_events = [
            ("messageStart", {"role": "assistant"}),
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello "}}),
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "world"}}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", {"usage": {"inputTokens": 4, "outputTokens": 2, "totalTokens": 6}, "metrics": {}}),
        ]

        result, events = await ask(bedrock.config(streaming=True))

        assert result.content == "Hello world"
        assert result.usage == Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
        assert result.finish_reason == "end_turn"
        assert events == [ModelMessageChunk("Hello "), ModelMessageChunk("world"), ModelMessage("Hello world")]

    async def test_tool_use_accumulation(self, bedrock: FakeBedrock) -> None:
        bedrock.stream_events = [
            (
                "contentBlockStart",
                {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tc_1", "name": "alpha"}}},
            ),
            (
                "contentBlockStart",
                {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tc_2", "name": "beta"}}},
            ),
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"a"'}}}),
            ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"b": 2}'}}}),
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"toolUse": {"input": ": 1}"}}}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 1}),
            ("messageStop", {"stopReason": "tool_use"}),
        ]

        result, _ = await ask(bedrock.config(streaming=True))

        assert result.tool_calls.calls == [
            ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}'),
            ToolCallEvent(id="tc_2", name="beta", arguments='{"b": 2}'),
        ]
        assert result.finish_reason == "tool_use"

    async def test_empty_tool_input_falls_back_to_empty_object(self, bedrock: FakeBedrock) -> None:
        bedrock.stream_events = [
            (
                "contentBlockStart",
                {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tc_1", "name": "noop"}}},
            ),
            ("contentBlockStop", {"contentBlockIndex": 0}),
        ]

        result, _ = await ask(bedrock.config(streaming=True))

        assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="noop", arguments="{}")]

    async def test_reasoning_delta(self, bedrock: FakeBedrock) -> None:
        bedrock.stream_events = [
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}),
            ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "answer"}}),
        ]

        _, events = await ask(bedrock.config(streaming=True))

        assert events == [ModelReasoning("hmm"), ModelMessageChunk("answer"), ModelMessage("answer")]

    async def test_missing_metadata_yields_empty_usage(self, bedrock: FakeBedrock) -> None:
        bedrock.stream_events = [
            ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
        ]

        result, _ = await ask(bedrock.config(streaming=True))

        assert result.usage == Usage()
