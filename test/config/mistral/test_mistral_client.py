# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from dirty_equals import IsPartialDict, IsStr
from fast_depends.pydantic import PydanticSerializer
from pydantic import BaseModel
from typing_extensions import Unpack

from ag2.config.mistral import MistralClient
from ag2.config.mistral.mistral_client import CreateOptions
from ag2.events import (
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolResult,
    UrlInput,
    Usage,
)
from ag2.response import PromptedSchema, ResponseProto
from ag2.tools.schemas import ToolSchema
from test.config._helpers import WireRecorder, json_response, make_tool, sse_response
from test.config.mistral._helpers import (
    CHAT_URL,
    FailingFetches,
    image_response,
    make_agentic_response,
    make_call_context,
    make_response,
    make_stream_chunk,
    make_tool_call,
    make_turn,
    make_usage,
)

pytestmark = pytest.mark.asyncio

GENERATED_URL = "https://example.com/generated.jpg"


class Verdict(BaseModel):
    answer: str


def _make_client(
    wire: WireRecorder | FailingFetches, *, streaming: bool = False, **options: Unpack[CreateOptions]
) -> MistralClient:
    # One transport answers both the chat API and generated-image fetches.
    return MistralClient(
        "mistral-test",
        streaming=streaming,
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        create_options=options or None,
    )


async def _ask(
    client: MistralClient,
    context: AsyncMock | None = None,
    tools: tuple[ToolSchema, ...] = (),
    response_schema: ResponseProto[Any] | None = None,
) -> ModelResponse:
    return await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=context if context is not None else make_call_context(),
        tools=tools,
        response_schema=response_schema,
        serializer=PydanticSerializer(),
    )


class TestRequest:
    async def test_model_and_messages_are_sent(self) -> None:
        wire = WireRecorder(json_response(make_response()))

        await _ask(_make_client(wire))

        assert [str(r.url) for r in wire.requests] == [CHAT_URL]
        assert wire.bodies == [
            IsPartialDict({"model": "mistral-test", "messages": [{"role": "user", "content": "hello"}]})
        ]

    async def test_empty_tools_are_omitted(self) -> None:
        wire = WireRecorder(json_response(make_response()))

        await _ask(_make_client(wire))

        [body] = wire.bodies
        assert "tools" not in body
        assert "response_format" not in body

    async def test_function_tool_is_forwarded(self) -> None:
        wire = WireRecorder(json_response(make_response()))

        await _ask(_make_client(wire), tools=(make_tool().schema,))

        assert wire.bodies == [
            IsPartialDict({
                "tools": [IsPartialDict({"type": "function", "function": IsPartialDict({"name": "search_docs"})})]
            })
        ]

    async def test_create_options_are_forwarded(self) -> None:
        wire = WireRecorder(json_response(make_response()))

        await _ask(_make_client(wire, temperature=0.25, max_tokens=64))

        assert wire.bodies == [IsPartialDict({"temperature": 0.25, "max_tokens": 64})]

    async def test_none_options_are_dropped(self) -> None:
        wire = WireRecorder(json_response(make_response()))

        await _ask(_make_client(wire, temperature=None, max_tokens=0))

        [body] = wire.bodies
        assert "temperature" not in body
        assert body == IsPartialDict({"max_tokens": 0})

    async def test_prompted_schema_appends_system_prompt(self) -> None:
        wire = WireRecorder(json_response(make_response()))
        context = make_call_context(["Be brief."])

        await _ask(_make_client(wire), context=context, response_schema=PromptedSchema(Verdict))

        assert wire.bodies == [
            IsPartialDict({
                "messages": [
                    IsPartialDict({"role": "system", "content": IsStr(regex=r"Be brief\.\n.*", regex_flags=re.DOTALL)}),
                    IsPartialDict({"role": "user"}),
                ]
            })
        ]


class TestNonStreaming:
    async def test_text_response(self) -> None:
        wire = WireRecorder(json_response(make_response(content="hi there")))
        context = make_call_context()

        response = await _ask(_make_client(wire), context=context)

        assert response.message == ModelMessage("hi there")
        assert response.provider == "mistral"
        assert response.model == "mistral-test"
        assert response.finish_reason == "stop"
        context.send.assert_awaited_with(ModelMessage("hi there"))

    async def test_empty_content_yields_no_message(self) -> None:
        wire = WireRecorder(json_response(make_response(content="")))

        response = await _ask(_make_client(wire))

        assert response.message is None

    async def test_think_chunks_become_reasoning(self) -> None:
        wire = WireRecorder(
            json_response(
                make_response(
                    content=[
                        {"type": "thinking", "thinking": [{"type": "text", "text": "let me see"}]},
                        {"type": "text", "text": "the answer"},
                    ]
                )
            )
        )
        context = make_call_context()

        response = await _ask(_make_client(wire), context=context)

        assert response.message == ModelMessage("the answer")
        context.send.assert_any_await(ModelReasoning("let me see"))

    async def test_tool_calls(self) -> None:
        wire = WireRecorder(
            json_response(make_response(content=None, tool_calls=[make_tool_call()], finish_reason="tool_calls"))
        )

        response = await _ask(_make_client(wire))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]
        assert response.finish_reason == "tool_calls"

    async def test_dict_arguments_are_json_encoded(self) -> None:
        wire = WireRecorder(json_response(make_response(tool_calls=[make_tool_call(arguments={"query": "x"})])))

        response = await _ask(_make_client(wire))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]

    async def test_usage_is_normalised(self) -> None:
        """``prompt_tokens_details`` is not in the SDK's model, so it rides along as an extra."""
        usage = {**make_usage(10, 5, 15), "prompt_tokens_details": {"cached_tokens": 4}}
        wire = WireRecorder(json_response({**make_response(), "usage": usage}))

        response = await _ask(_make_client(wire))

        assert response.usage == Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cache_read_input_tokens=4,
        )


class TestServerExecutedTools:
    """``image_generation`` runs on Mistral's side; the whole exchange comes back
    in ``choice.messages`` with ``choice.message`` set to None."""

    async def test_answer_is_read_from_messages(self) -> None:
        wire = WireRecorder(json_response(make_agentic_response()), image_response())

        response = await _ask(_make_client(wire))

        assert response.message == ModelMessage("Here is your image.")
        assert response.finish_reason == "stop"

    async def test_call_and_result_are_emitted_as_builtin_events(self) -> None:
        wire = WireRecorder(json_response(make_agentic_response()), image_response())
        context = make_call_context()

        await _ask(_make_client(wire), context=context)

        context.send.assert_any_await(
            BuiltinToolCallEvent(id="gen_1", name="image_generation", arguments='{"prompt": "a red circle"}')
        )
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput(GENERATED_URL, kind=BinaryType.IMAGE)),
            )
        )

    async def test_server_executed_call_is_not_returned_for_dispatch(self) -> None:
        """Re-dispatching it would have the agent run a tool it never registered."""
        wire = WireRecorder(json_response(make_agentic_response()), image_response())

        response = await _ask(_make_client(wire))

        assert response.tool_calls.calls == []

    async def test_client_side_call_without_a_result_is_still_dispatched(self) -> None:
        turns = [make_turn(tool_calls=[("tc_1", "search_docs", '{"query": "x"}')])]
        wire = WireRecorder(json_response(make_agentic_response(turns=turns, finish_reason="tool_calls")))

        response = await _ask(_make_client(wire))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]

    async def test_streaming_reports_the_result_mid_stream(self) -> None:
        wire = WireRecorder(
            sse_response(
                make_stream_chunk(tool_calls=[make_tool_call("gen_1", "generate_image", "{}", index=0)]),
                make_stream_chunk(content=f'{{"url": "{GENERATED_URL}"}}', tool_call_id="gen_1"),
                make_stream_chunk(content="Here it is."),
                make_stream_chunk(finish_reason="stop"),
            ),
            image_response(),
        )
        context = make_call_context()

        response = await _ask(_make_client(wire, streaming=True), context=context)

        assert response.message == ModelMessage("Here it is.")
        assert response.tool_calls.calls == []
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput(GENERATED_URL, kind=BinaryType.IMAGE)),
            )
        )

    async def test_generated_image_lands_on_files(self) -> None:
        """``reply.files`` is the cross-provider contract for generated images."""
        wire = WireRecorder(
            json_response(make_agentic_response()),
            image_response(data=b"\xff\xd8jpegbytes", content_type="image/jpeg"),
        )

        response = await _ask(_make_client(wire))

        assert [str(r.url) for r in wire.requests] == [CHAT_URL, GENERATED_URL]
        assert [f.data for f in response.files] == [b"\xff\xd8jpegbytes"]
        assert response.files[0].metadata == IsPartialDict({"media_type": "image/jpeg", "url": GENERATED_URL})
        assert response.files[0].name == "generated.jpg"

    async def test_media_type_falls_back_to_the_url_suffix(self) -> None:
        """Blob storage serves generated images as octet-stream."""
        wire = WireRecorder(
            json_response(make_agentic_response()), image_response(content_type="application/octet-stream")
        )

        response = await _ask(_make_client(wire))

        assert response.files[0].metadata == IsPartialDict({"media_type": "image/jpeg"})
        assert response.files[0].name == "generated.jpg"

    async def test_failed_download_does_not_break_the_turn(self) -> None:
        """The URL is still on the tool-result event, so the turn stays usable."""
        wire = FailingFetches(WireRecorder(json_response(make_agentic_response())), httpx.ConnectError("boom"))
        context = make_call_context()

        response = await _ask(_make_client(wire), context=context)

        assert response.files == []
        assert response.message == ModelMessage("Here is your image.")
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput(GENERATED_URL, kind=BinaryType.IMAGE)),
            )
        )

    async def test_streaming_also_populates_files(self) -> None:
        wire = WireRecorder(
            sse_response(
                make_stream_chunk(tool_calls=[make_tool_call("gen_1", "generate_image", "{}", index=0)]),
                make_stream_chunk(content=f'{{"url": "{GENERATED_URL}"}}', tool_call_id="gen_1"),
                make_stream_chunk(content="done", finish_reason="stop"),
            ),
            image_response(data=b"streamed"),
        )

        response = await _ask(_make_client(wire, streaming=True))

        assert [f.data for f in response.files] == [b"streamed"]

    async def test_plain_turns_fetch_nothing(self) -> None:
        wire = WireRecorder(json_response(make_response(content="hi")))

        response = await _ask(_make_client(wire))

        assert [str(r.url) for r in wire.requests] == [CHAT_URL]
        assert response.files == []

    async def test_tool_result_payload_is_not_treated_as_answer_text(self) -> None:
        """The raw ``{"url": ...}`` JSON must not leak into the reply body."""
        wire = WireRecorder(json_response(make_agentic_response()), image_response())

        response = await _ask(_make_client(wire))

        assert "http" not in (response.message.content if response.message else "")


class TestStreaming:
    async def test_chunks_are_accumulated_and_emitted(self) -> None:
        wire = WireRecorder(
            sse_response(
                make_stream_chunk(content="Hello, "),
                make_stream_chunk(content="world"),
                make_stream_chunk(finish_reason="stop", usage=make_usage(3, 2, 5)),
            )
        )
        context = make_call_context()

        response = await _ask(_make_client(wire, streaming=True), context=context)

        assert response.message == ModelMessage("Hello, world")
        assert response.finish_reason == "stop"
        assert response.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        context.send.assert_any_await(ModelMessageChunk("Hello, "))
        context.send.assert_any_await(ModelMessageChunk("world"))

    async def test_tool_calls_accumulate_by_index(self) -> None:
        wire = WireRecorder(
            sse_response(
                make_stream_chunk(tool_calls=[make_tool_call(index=0, arguments='{"query":')]),
                make_stream_chunk(
                    tool_calls=[make_tool_call(index=0, call_id="tc_1", name="search_docs", arguments='"x"}')]
                ),
                make_stream_chunk(finish_reason="tool_calls"),
            )
        )

        response = await _ask(_make_client(wire, streaming=True))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query":"x"}')]

    async def test_incomplete_tool_call_is_dropped(self) -> None:
        """A call that never receives an id or name cannot be dispatched."""
        wire = WireRecorder(sse_response(make_stream_chunk(tool_calls=[make_tool_call(index=0, call_id="", name="")])))

        response = await _ask(_make_client(wire, streaming=True))

        assert response.tool_calls.calls == []

    async def test_think_chunks_stream_as_reasoning(self) -> None:
        wire = WireRecorder(
            sse_response(
                make_stream_chunk(content=[{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}]),
                make_stream_chunk(content=[{"type": "text", "text": "done"}]),
            )
        )
        context = make_call_context()

        response = await _ask(_make_client(wire, streaming=True), context=context)

        assert response.message == ModelMessage("done")
        context.send.assert_any_await(ModelReasoning("hmm"))

    async def test_falls_back_to_configured_model(self) -> None:
        """The wire always carries ``model``; an empty one falls back to the configured name."""
        wire = WireRecorder(sse_response(make_stream_chunk(content="x", model="")))

        response = await _ask(_make_client(wire, streaming=True))

        assert response.model == "mistral-test"
