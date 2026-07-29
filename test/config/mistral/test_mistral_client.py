# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import httpx
import pytest
from fast_depends.use import SerializerCls
from mistralai.client.models import TextChunk, ThinkChunk, Tool
from pydantic import BaseModel

from ag2.config.mistral import MistralClient
from ag2.events import (
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    TextInput,
    ToolCallEvent,
    ToolResult,
    UrlInput,
    Usage,
)
from ag2.response import PromptedSchema
from test.config._helpers import make_tool
from test.config.mistral._helpers import (
    FakeChat,
    FakeHttpClient,
    FakeMistralClient,
    make_agentic_response,
    make_call_context,
    make_response,
    make_stream_chunk,
    make_tool_call,
    make_usage,
)

pytestmark = pytest.mark.asyncio


class Verdict(BaseModel):
    answer: str


def _make_client(
    chat: FakeChat, *, streaming: bool = False, http: FakeHttpClient | None = None, **options
) -> MistralClient:
    # A fake http client keeps generated-image fetches off the network.
    client = MistralClient(
        "mistral-test",
        streaming=streaming,
        async_client=http or FakeHttpClient(),
        create_options=options or None,
    )
    client._client = FakeMistralClient(chat)
    return client


async def _ask(client: MistralClient, context=None, tools=(), response_schema=None):
    return await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=context if context is not None else make_call_context(),
        tools=tools,
        response_schema=response_schema,
        serializer=SerializerCls,
    )


class TestRequest:
    async def test_model_and_messages_are_sent(self) -> None:
        chat = FakeChat()

        await _ask(_make_client(chat))

        assert chat.kwargs["model"] == "mistral-test"
        assert [m.role for m in chat.kwargs["messages"]] == ["user"]

    async def test_empty_tools_are_omitted(self) -> None:
        chat = FakeChat()

        await _ask(_make_client(chat))

        assert "tools" not in chat.kwargs
        assert "response_format" not in chat.kwargs

    async def test_function_tool_is_forwarded(self) -> None:
        chat = FakeChat()

        await _ask(_make_client(chat), tools=[make_tool().schema])

        [tool] = chat.kwargs["tools"]
        assert isinstance(tool, Tool)
        assert tool.function.name == "search_docs"

    async def test_create_options_are_forwarded(self) -> None:
        chat = FakeChat()

        await _ask(_make_client(chat, temperature=0.25, max_tokens=64))

        assert chat.kwargs["temperature"] == 0.25
        assert chat.kwargs["max_tokens"] == 64

    async def test_none_options_are_dropped(self) -> None:
        chat = FakeChat()

        await _ask(_make_client(chat, temperature=None, max_tokens=0))

        assert "temperature" not in chat.kwargs
        assert chat.kwargs["max_tokens"] == 0

    async def test_prompted_schema_appends_system_prompt(self) -> None:
        chat = FakeChat()
        context = make_call_context(["Be brief."])

        await _ask(_make_client(chat), context=context, response_schema=PromptedSchema(Verdict))

        [system, _user] = chat.kwargs["messages"]
        assert system.role == "system"
        assert system.content.startswith("Be brief.\n")


class TestNonStreaming:
    async def test_text_response(self) -> None:
        chat = FakeChat(make_response(content="hi there"))
        context = make_call_context()

        response = await _ask(_make_client(chat), context=context)

        assert response.message == ModelMessage("hi there")
        assert response.provider == "mistral"
        assert response.model == "mistral-test"
        assert response.finish_reason == "stop"
        context.send.assert_awaited_with(ModelMessage("hi there"))

    async def test_empty_content_yields_no_message(self) -> None:
        response = await _ask(_make_client(FakeChat(make_response(content=""))))

        assert response.message is None

    async def test_think_chunks_become_reasoning(self) -> None:
        content = [
            ThinkChunk(thinking=[TextChunk(text="let me see")]),
            TextChunk(text="the answer"),
        ]
        chat = FakeChat(make_response(content=content))
        context = make_call_context()

        response = await _ask(_make_client(chat), context=context)

        assert response.message == ModelMessage("the answer")
        context.send.assert_any_await(ModelReasoning("let me see"))

    async def test_tool_calls(self) -> None:
        chat = FakeChat(make_response(content=None, tool_calls=[make_tool_call()], finish_reason="tool_calls"))

        response = await _ask(_make_client(chat))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]
        assert response.finish_reason == "tool_calls"

    async def test_dict_arguments_are_json_encoded(self) -> None:
        chat = FakeChat(make_response(tool_calls=[make_tool_call(arguments={"query": "x"})]))

        response = await _ask(_make_client(chat))

        assert response.tool_calls.calls[0].arguments == '{"query": "x"}'

    async def test_usage_is_normalised(self) -> None:
        chat = FakeChat(make_response(usage=make_usage(10, 5, 15, cached_tokens=4)))

        response = await _ask(_make_client(chat))

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
        chat = FakeChat(make_agentic_response())
        context = make_call_context()

        response = await _ask(_make_client(chat), context=context)

        assert response.message == ModelMessage("Here is your image.")
        assert response.finish_reason == "stop"

    async def test_call_and_result_are_emitted_as_builtin_events(self) -> None:
        chat = FakeChat(make_agentic_response())
        context = make_call_context()

        await _ask(_make_client(chat), context=context)

        context.send.assert_any_await(
            BuiltinToolCallEvent(id="gen_1", name="image_generation", arguments='{"prompt": "a red circle"}')
        )
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput("https://example.com/generated.jpg", kind=BinaryType.IMAGE)),
            )
        )

    async def test_server_executed_call_is_not_returned_for_dispatch(self) -> None:
        """Re-dispatching it would have the agent run a tool it never registered."""
        response = await _ask(_make_client(FakeChat(make_agentic_response())))

        assert response.tool_calls.calls == []

    async def test_client_side_call_without_a_result_is_still_dispatched(self) -> None:
        turns = [
            SimpleNamespace(
                content="",
                tool_call_id=None,
                tool_calls=[make_tool_call("tc_1", "search_docs", '{"query": "x"}')],
            )
        ]
        chat = FakeChat(make_agentic_response(turns=turns, finish_reason="tool_calls"))

        response = await _ask(_make_client(chat))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query": "x"}')]

    async def test_streaming_reports_the_result_mid_stream(self) -> None:
        chat = FakeChat(
            stream_chunks=[
                make_stream_chunk(tool_calls=[make_tool_call("gen_1", "generate_image", "{}", index=0)]),
                make_stream_chunk(content='{"url": "https://example.com/generated.jpg"}', tool_call_id="gen_1"),
                make_stream_chunk(content="Here it is."),
                make_stream_chunk(finish_reason="stop"),
            ]
        )
        context = make_call_context()

        response = await _ask(_make_client(chat, streaming=True), context=context)

        assert response.message == ModelMessage("Here it is.")
        assert response.tool_calls.calls == []
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput("https://example.com/generated.jpg", kind=BinaryType.IMAGE)),
            )
        )

    async def test_generated_image_lands_on_files(self) -> None:
        """``reply.files`` is the cross-provider contract for generated images."""
        http = FakeHttpClient(data=b"\xff\xd8jpegbytes", content_type="image/jpeg")

        response = await _ask(_make_client(FakeChat(make_agentic_response()), http=http))

        assert http.urls == ["https://example.com/generated.jpg"]
        assert [f.data for f in response.files] == [b"\xff\xd8jpegbytes"]
        assert response.files[0].metadata["media_type"] == "image/jpeg"
        assert response.files[0].metadata["url"] == "https://example.com/generated.jpg"
        assert response.files[0].name == "generated.jpg"

    async def test_media_type_falls_back_to_the_url_suffix(self) -> None:
        """Blob storage serves generated images as octet-stream."""
        http = FakeHttpClient(content_type="application/octet-stream")

        response = await _ask(_make_client(FakeChat(make_agentic_response()), http=http))

        assert response.files[0].metadata["media_type"] == "image/jpeg"
        assert response.files[0].name == "generated.jpg"

    async def test_failed_download_does_not_break_the_turn(self) -> None:
        """The URL is still on the tool-result event, so the turn stays usable."""
        http = FakeHttpClient(error=httpx.ConnectError("boom"))
        context = make_call_context()

        response = await _ask(_make_client(FakeChat(make_agentic_response()), http=http), context=context)

        assert response.files == []
        assert response.message == ModelMessage("Here is your image.")
        context.send.assert_any_await(
            BuiltinToolResultEvent(
                parent_id="gen_1",
                name="image_generation",
                result=ToolResult(UrlInput("https://example.com/generated.jpg", kind=BinaryType.IMAGE)),
            )
        )

    async def test_streaming_also_populates_files(self) -> None:
        http = FakeHttpClient(data=b"streamed")
        chat = FakeChat(
            stream_chunks=[
                make_stream_chunk(tool_calls=[make_tool_call("gen_1", "generate_image", "{}", index=0)]),
                make_stream_chunk(content='{"url": "https://example.com/generated.jpg"}', tool_call_id="gen_1"),
                make_stream_chunk(content="done", finish_reason="stop"),
            ]
        )

        response = await _ask(_make_client(chat, streaming=True, http=http))

        assert [f.data for f in response.files] == [b"streamed"]

    async def test_plain_turns_fetch_nothing(self) -> None:
        http = FakeHttpClient()

        response = await _ask(_make_client(FakeChat(make_response(content="hi")), http=http))

        assert http.urls == []
        assert response.files == []

    async def test_tool_result_payload_is_not_treated_as_answer_text(self) -> None:
        """The raw ``{"url": ...}`` JSON must not leak into the reply body."""
        response = await _ask(_make_client(FakeChat(make_agentic_response())))

        assert "http" not in (response.message.content if response.message else "")


class TestStreaming:
    async def test_chunks_are_accumulated_and_emitted(self) -> None:
        chat = FakeChat(
            stream_chunks=[
                make_stream_chunk(content="Hello, ", model="mistral-test"),
                make_stream_chunk(content="world"),
                make_stream_chunk(finish_reason="stop", usage=make_usage(3, 2, 5)),
            ]
        )
        context = make_call_context()

        response = await _ask(_make_client(chat, streaming=True), context=context)

        assert response.message == ModelMessage("Hello, world")
        assert response.finish_reason == "stop"
        assert response.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        context.send.assert_any_await(ModelMessageChunk("Hello, "))
        context.send.assert_any_await(ModelMessageChunk("world"))

    async def test_tool_calls_accumulate_by_index(self) -> None:
        chat = FakeChat(
            stream_chunks=[
                make_stream_chunk(tool_calls=[make_tool_call(index=0, arguments='{"query":')]),
                make_stream_chunk(
                    tool_calls=[make_tool_call(index=0, call_id="tc_1", name="search_docs", arguments='"x"}')]
                ),
                make_stream_chunk(finish_reason="tool_calls"),
            ]
        )

        response = await _ask(_make_client(chat, streaming=True))

        assert response.tool_calls.calls == [ToolCallEvent(id="tc_1", name="search_docs", arguments='{"query":"x"}')]

    async def test_incomplete_tool_call_is_dropped(self) -> None:
        """A call that never receives an id or name cannot be dispatched."""
        chat = FakeChat(stream_chunks=[make_stream_chunk(tool_calls=[make_tool_call(index=0, call_id="", name="")])])

        response = await _ask(_make_client(chat, streaming=True))

        assert response.tool_calls.calls == []

    async def test_think_chunks_stream_as_reasoning(self) -> None:
        chat = FakeChat(
            stream_chunks=[
                make_stream_chunk(content=[ThinkChunk(thinking=[TextChunk(text="hmm")])]),
                make_stream_chunk(content=[TextChunk(text="done")]),
            ]
        )
        context = make_call_context()

        response = await _ask(_make_client(chat, streaming=True), context=context)

        assert response.message == ModelMessage("done")
        context.send.assert_any_await(ModelReasoning("hmm"))

    async def test_falls_back_to_configured_model(self) -> None:
        response = await _ask(_make_client(FakeChat(stream_chunks=[make_stream_chunk(content="x")]), streaming=True))

        assert response.model == "mistral-test"
