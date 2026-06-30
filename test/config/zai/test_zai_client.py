# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls
from pydantic import BaseModel

from ag2.config.zai import ZAIClient
from ag2.events import (
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    TextInput,
    ToolCallEvent,
    Usage,
)
from ag2.response import PromptedSchema
from test.config._helpers import make_tool
from test.config.zai._helpers import (
    FakeCompletions,
    FakeZAIClient,
    make_call_context,
    make_response,
    make_stream_chunk,
    make_stream_tool_call,
    make_tool_call,
    make_usage,
)


class Verdict(BaseModel):
    answer: str


def _make_client(completions: FakeCompletions, *, streaming: bool = False) -> ZAIClient:
    client = ZAIClient(create_options={"model": "glm-test", "stream": streaming})
    client._client = FakeZAIClient(completions)
    return client


async def _ask(client: ZAIClient, context=None, tools=(), response_schema=None):
    return await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=context if context is not None else make_call_context(),
        tools=tools,
        response_schema=response_schema,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
async def test_empty_tools_are_omitted() -> None:
    completions = FakeCompletions()

    await _ask(_make_client(completions))

    assert "tools" not in completions.kwargs


@pytest.mark.asyncio
async def test_function_tools_serialize() -> None:
    completions = FakeCompletions()

    await _ask(_make_client(completions), tools=[make_tool().schema])

    assert completions.kwargs == IsPartialDict({
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search documentation by query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    })


@pytest.mark.asyncio
async def test_system_prompt_and_response_schema_prompt_land_in_messages() -> None:
    completions = FakeCompletions()
    schema = PromptedSchema(Verdict)

    await _ask(_make_client(completions), context=make_call_context(["You are helpful."]), response_schema=schema)

    messages = completions.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": f"You are helpful.\n{schema.system_prompt}"}


@pytest.mark.asyncio
async def test_non_streaming_text_reasoning_tool_calls_usage_finish_reason() -> None:
    completions = FakeCompletions(
        response=make_response(
            content="The answer is 42.",
            reasoning_content="thinking...",
            tool_calls=[make_tool_call(arguments={"query": "x"})],
            finish_reason="tool_calls",
            usage=make_usage(prompt_tokens=10, completion_tokens=5),
            model="glm-5.2",
        )
    )
    context = make_call_context()

    result = await _ask(_make_client(completions), context=context)

    assert result.content == "The answer is 42."
    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]
    assert result.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert result.model == "glm-5.2"
    assert result.provider == "zai"
    assert result.finish_reason == "tool_calls"
    assert [c.args[0] for c in context.send.call_args_list] == [
        ModelReasoning("thinking..."),
        ModelMessage("The answer is 42."),
    ]


@pytest.mark.asyncio
async def test_streaming_text_reasoning_usage_and_finish_reason() -> None:
    completions = FakeCompletions(
        stream_chunks=[
            make_stream_chunk(reasoning_content="hmm"),
            make_stream_chunk(content="Hello "),
            make_stream_chunk(
                content="world", finish_reason="stop", usage=make_usage(prompt_tokens=4, completion_tokens=2)
            ),
        ]
    )
    context = make_call_context()

    result = await _ask(_make_client(completions, streaming=True), context=context)

    assert result.content == "Hello world"
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    assert result.finish_reason == "stop"
    assert [c.args[0] for c in context.send.call_args_list] == [
        ModelReasoning("hmm"),
        ModelMessageChunk("Hello "),
        ModelMessageChunk("world"),
        ModelMessage("Hello world"),
    ]


@pytest.mark.asyncio
async def test_streaming_tool_call_accumulation_and_empty_input() -> None:
    completions = FakeCompletions(
        stream_chunks=[
            make_stream_chunk(tool_calls=[make_stream_tool_call(0, call_id="tc_1", name="alpha", arguments='{"a"')]),
            make_stream_chunk(tool_calls=[make_stream_tool_call(1, call_id="tc_2", name="beta")]),
            make_stream_chunk(tool_calls=[make_stream_tool_call(0, arguments=": 1}")], finish_reason="tool_calls"),
        ]
    )

    result = await _ask(_make_client(completions, streaming=True))

    assert result.tool_calls.calls == [
        ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}'),
        ToolCallEvent(id="tc_2", name="beta", arguments="{}"),
    ]
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_tool_call_empty_first_arguments_fragment() -> None:
    # OpenAI-compatible streams routinely send the first tool-call delta with
    # arguments="". That empty fragment must not inject "{}" into the accumulator.
    completions = FakeCompletions(
        stream_chunks=[
            make_stream_chunk(tool_calls=[make_stream_tool_call(0, call_id="tc_1", name="alpha", arguments="")]),
            make_stream_chunk(tool_calls=[make_stream_tool_call(0, arguments='{"a"')]),
            make_stream_chunk(tool_calls=[make_stream_tool_call(0, arguments=": 1}")], finish_reason="tool_calls"),
        ]
    )

    result = await _ask(_make_client(completions, streaming=True))

    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}')]


@pytest.mark.asyncio
async def test_streaming_missing_usage_yields_empty_usage() -> None:
    completions = FakeCompletions(stream_chunks=[make_stream_chunk(content="hi")])

    result = await _ask(_make_client(completions, streaming=True))

    assert result.usage == Usage()
