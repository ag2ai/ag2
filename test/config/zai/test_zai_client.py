# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock

import pytest
from dirty_equals import IsPartialDict
from fast_depends.pydantic import PydanticSerializer
from pydantic import BaseModel
from zai.types.chat.chat_completion import CompletionUsage
from zai.types.chat.chat_completion_chunk import CompletionUsage as ChunkUsage

from ag2.config.zai import ZAIConfig
from ag2.events import (
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    Usage,
)
from ag2.response import PromptedSchema, ResponseProto
from ag2.tools.schemas import ToolSchema
from test.config._helpers import WireRecorder, json_response, make_tool, sse_response
from test.config.zai._helpers import (
    chunk_json,
    completion_json,
    make_call_context,
    tool_call_delta_json,
    tool_call_json,
    wire_config,
    with_object_arguments,
)


class Verdict(BaseModel):
    answer: str


async def _ask(
    config: ZAIConfig,
    context: AsyncMock | None = None,
    tools: list[ToolSchema] | None = None,
    response_schema: ResponseProto[Any] | None = None,
) -> ModelResponse:
    return await config.create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=context if context is not None else make_call_context(),
        tools=tools or [],
        response_schema=response_schema,
        serializer=PydanticSerializer(),
    )


@pytest.mark.asyncio
async def test_empty_tools_are_omitted() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder) as config:
        await _ask(config)

    [body] = recorder.bodies
    assert "tools" not in body


@pytest.mark.asyncio
async def test_function_tools_serialize() -> None:
    recorder = WireRecorder(json_response(completion_json()))

    with wire_config(recorder) as config:
        await _ask(config, tools=[make_tool().schema])

    assert recorder.bodies == [
        IsPartialDict({
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
    ]


@pytest.mark.asyncio
async def test_system_prompt_and_response_schema_prompt_land_in_messages() -> None:
    recorder = WireRecorder(json_response(completion_json()))
    schema = PromptedSchema(Verdict)

    with wire_config(recorder) as config:
        await _ask(config, context=make_call_context(["You are helpful."]), response_schema=schema)

    assert recorder.bodies == [
        IsPartialDict({
            "messages": [
                {"role": "system", "content": f"You are helpful.\n{schema.system_prompt}"},
                {"role": "user", "content": "hello"},
            ]
        })
    ]


@pytest.mark.asyncio
async def test_non_streaming_text_reasoning_tool_calls_usage_finish_reason() -> None:
    recorder = WireRecorder(
        json_response(
            with_object_arguments(
                completion_json(
                    content="The answer is 42.",
                    reasoning_content="thinking...",
                    tool_calls=[tool_call_json()],
                    finish_reason="tool_calls",
                    usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="glm-5.2",
                ),
                {"query": "x"},
            )
        )
    )
    context = make_call_context()

    with wire_config(recorder) as config:
        result = await _ask(config, context=context)

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
    recorder = WireRecorder(
        sse_response(
            chunk_json(reasoning_content="hmm"),
            chunk_json(content="Hello "),
            chunk_json(
                content="world",
                finish_reason="stop",
                usage=ChunkUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            ),
        )
    )
    context = make_call_context()

    with wire_config(recorder, streaming=True) as config:
        result = await _ask(config, context=context)

    assert recorder.bodies == [IsPartialDict({"stream": True})]
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
    recorder = WireRecorder(
        sse_response(
            chunk_json(tool_calls=[tool_call_delta_json(0, call_id="tc_1", name="alpha", arguments='{"a"')]),
            chunk_json(tool_calls=[tool_call_delta_json(1, call_id="tc_2", name="beta")]),
            chunk_json(tool_calls=[tool_call_delta_json(0, arguments=": 1}")], finish_reason="tool_calls"),
        )
    )

    with wire_config(recorder, streaming=True) as config:
        result = await _ask(config)

    assert result.tool_calls.calls == [
        ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}'),
        ToolCallEvent(id="tc_2", name="beta", arguments="{}"),
    ]
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_tool_call_empty_first_arguments_fragment() -> None:
    # OpenAI-compatible streams routinely send the first tool-call delta with
    # arguments="". That empty fragment must not inject "{}" into the accumulator.
    recorder = WireRecorder(
        sse_response(
            chunk_json(tool_calls=[tool_call_delta_json(0, call_id="tc_1", name="alpha", arguments="")]),
            chunk_json(tool_calls=[tool_call_delta_json(0, arguments='{"a"')]),
            chunk_json(tool_calls=[tool_call_delta_json(0, arguments=": 1}")], finish_reason="tool_calls"),
        )
    )

    with wire_config(recorder, streaming=True) as config:
        result = await _ask(config)

    assert result.tool_calls.calls == [ToolCallEvent(id="tc_1", name="alpha", arguments='{"a": 1}')]


@pytest.mark.asyncio
async def test_streaming_missing_usage_yields_empty_usage() -> None:
    recorder = WireRecorder(sse_response(chunk_json(content="hi")))

    with wire_config(recorder, streaming=True) as config:
        result = await _ask(config)

    assert result.usage == Usage()
