# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2.config.bedrock import BedrockClient
from ag2.events import (
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    TextInput,
    ToolCallEvent,
    Usage,
)
from test.config._helpers import make_tool
from test.config.bedrock._helpers import (
    FakeBedrockRuntime,
    StubSession,
    make_call_context,
    make_converse_response,
)


def _make_client(fake: FakeBedrockRuntime, streaming: bool = False) -> BedrockClient:
    return BedrockClient(
        session=StubSession(fake),
        create_options={"model": "m1", "stream": streaming},
    )


async def _ask(client: BedrockClient, context=None, tools=(), response_schema=None):
    return await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=context if context is not None else make_call_context(),
        tools=tools,
        response_schema=response_schema,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_empty_tools_omits_tool_config() -> None:
    fake = FakeBedrockRuntime()
    client = _make_client(fake)

    await _ask(client)

    assert "toolConfig" not in fake.converse_kwargs
    assert "inferenceConfig" not in fake.converse_kwargs
    assert "system" not in fake.converse_kwargs


@pytest.mark.asyncio
async def test_tools_serialized_as_tool_spec() -> None:
    fake = FakeBedrockRuntime()
    client = _make_client(fake)

    await _ask(client, tools=[make_tool().schema])

    assert fake.converse_kwargs["toolConfig"] == {
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
    }


@pytest.mark.asyncio
async def test_system_prompt_lands_in_system_param() -> None:
    fake = FakeBedrockRuntime()
    client = _make_client(fake)

    await _ask(client, context=make_call_context(prompt=["You are helpful.", "Be brief."]))

    assert fake.converse_kwargs["system"] == [{"text": "You are helpful.\nBe brief."}]


@pytest.mark.asyncio
async def test_non_streaming_response() -> None:
    fake = FakeBedrockRuntime(
        response=make_converse_response(
            content=[
                {"text": "The answer is 42."},
                {"toolUse": {"toolUseId": "tc_1", "name": "search_docs", "input": {"query": "x"}}},
            ],
            stop_reason="tool_use",
            usage={"inputTokens": 10, "outputTokens": 5},
        ),
    )
    client = _make_client(fake)
    context = make_call_context()

    result = await _ask(client, context=context)

    assert result.content == "The answer is 42."
    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]
    assert result.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert result.model == "m1"
    assert result.provider == "bedrock"
    assert result.finish_reason == "tool_use"

    sent = [c.args[0] for c in context.send.call_args_list]
    assert sent == [ModelMessage("The answer is 42.")]


@pytest.mark.asyncio
async def test_non_streaming_reasoning_content() -> None:
    fake = FakeBedrockRuntime(
        response=make_converse_response(
            content=[
                {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                {"text": "done"},
            ],
        ),
    )
    client = _make_client(fake)
    context = make_call_context()

    await _ask(client, context=context)

    sent = [c.args[0] for c in context.send.call_args_list]
    assert sent == [ModelReasoning("thinking..."), ModelMessage("done")]


@pytest.mark.asyncio
async def test_streaming_text_chunks() -> None:
    fake = FakeBedrockRuntime(
        stream_events=[
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello "}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 4, "outputTokens": 2}}},
        ],
    )
    client = _make_client(fake, streaming=True)
    context = make_call_context()

    result = await _ask(client, context=context)

    assert result.content == "Hello world"
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    assert result.finish_reason == "end_turn"

    sent = [c.args[0] for c in context.send.call_args_list]
    assert sent == [ModelMessageChunk("Hello "), ModelMessageChunk("world"), ModelMessage("Hello world")]


@pytest.mark.asyncio
async def test_streaming_tool_use_accumulation() -> None:
    fake = FakeBedrockRuntime(
        stream_events=[
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tc_1", "name": "alpha"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "tc_2", "name": "beta"}},
                }
            },
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"a"'}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"b": 2}'}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": ": 1}"}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )
    client = _make_client(fake, streaming=True)

    result = await _ask(client)

    assert result.tool_calls.calls == [
        ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}'),
        ToolCallEvent(id="tc_2", name="beta", arguments='{"b": 2}'),
    ]
    assert result.finish_reason == "tool_use"


@pytest.mark.asyncio
async def test_streaming_empty_tool_input_falls_back_to_empty_object() -> None:
    fake = FakeBedrockRuntime(
        stream_events=[
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tc_1", "name": "noop"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ],
    )
    client = _make_client(fake, streaming=True)

    result = await _ask(client)

    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="noop", arguments="{}")]


@pytest.mark.asyncio
async def test_streaming_reasoning_delta() -> None:
    fake = FakeBedrockRuntime(
        stream_events=[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "answer"}}},
        ],
    )
    client = _make_client(fake, streaming=True)
    context = make_call_context()

    await _ask(client, context=context)

    sent = [c.args[0] for c in context.send.call_args_list]
    assert sent == [ModelReasoning("hmm"), ModelMessageChunk("answer"), ModelMessage("answer")]


@pytest.mark.asyncio
async def test_streaming_missing_metadata_yields_empty_usage() -> None:
    fake = FakeBedrockRuntime(
        stream_events=[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        ],
    )
    client = _make_client(fake, streaming=True)

    result = await _ask(client)

    assert result.usage == Usage()
