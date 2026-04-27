# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

import pytest
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Summary

from autogen.beta import MemoryStream
from autogen.beta.config.openai import OpenAIResponsesClient
from autogen.beta.config.openai.events import (
    OpenAIReasoningEvent,
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
)
from autogen.beta.config.openai.mappers import events_to_responses_input
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


async def _process(output: Iterable[Any]) -> tuple[ModelResponse, list[BaseEvent]]:
    client = OpenAIResponsesClient(api_key="test")
    response = Response.model_construct(
        id="r1",
        object="response",
        model="gpt-5",
        output=list(output),
        usage=None,
    )
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    result = await client._process_response(response, context)
    return result, list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_process_response_routes_all_item_types() -> None:
    web = ResponseFunctionWebSearch(
        id="ws_1",
        action=ActionSearch(type="search", query="bitcoin"),
        status="completed",
        type="web_search_call",
    )
    code = ResponseCodeInterpreterToolCall(
        id="ci_1",
        code="print(1)",
        status="completed",
        type="code_interpreter_call",
        outputs=None,
        container_id="c_1",
    )
    image = ImageGenerationCall(
        id="ig_1",
        status="completed",
        type="image_generation_call",
        result="YWJj",  # base64 "abc"
        revised_prompt=None,
        output_format="png",
    )
    msg = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="Done.", annotations=[])],
    )
    user_tool = ResponseFunctionToolCall(
        id="id_1",
        call_id="call_1",
        name="multiply",
        arguments='{"a": 2, "b": 3}',
        type="function_call",
    )

    response, events = await _process([web, code, image, msg, user_tool])

    assert response.message == ModelMessage("Done.")
    assert response.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="call_1", name="multiply", arguments='{"a": 2, "b": 3}'),
    ])
    assert [f.data for f in response.files] == [b"abc"]
    assert events == [
        OpenAIServerToolCallEvent(
            id="ws_1", name=WEB_SEARCH_TOOL_NAME, arguments=web.action.model_dump_json(), item=web
        ),
        OpenAIServerToolResultEvent(parent_id="ws_1", name=WEB_SEARCH_TOOL_NAME, result=ToolResult()),
        OpenAIServerToolCallEvent(
            id="ci_1", name=CODE_EXECUTION_TOOL_NAME, arguments=json.dumps({"code": "print(1)"}), item=code
        ),
        OpenAIServerToolResultEvent(parent_id="ci_1", name=CODE_EXECUTION_TOOL_NAME, result=ToolResult()),
        OpenAIServerToolCallEvent(id="ig_1", name=IMAGE_GENERATION_TOOL_NAME, arguments="", item=image),
        OpenAIServerToolResultEvent(parent_id="ig_1", name=IMAGE_GENERATION_TOOL_NAME, result=ToolResult()),
    ]


@pytest.mark.asyncio
class TestReasoning:
    async def test_persisted_in_history(self) -> None:
        # ModelReasoning is __transient__ by default; the OpenAI subclass must
        # override it because the Responses API requires the reasoning item to
        # accompany subsequent server-side tool calls on the next turn.
        reasoning_item = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[Summary(type="summary_text", text="thinking")],
        )
        reasoning = OpenAIReasoningEvent("thinking", item=reasoning_item)

        stream = MemoryStream()
        context = ConversationContext(stream=stream)

        await stream.send(reasoning, context)

        assert list(await stream.history.get_events()) == [reasoning]

    async def test_round_trips_to_responses_api_input(self) -> None:
        reasoning_item = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[Summary(type="summary_text", text="Looking up bitcoin price")],
        )
        web_item = ResponseFunctionWebSearch(
            id="ws_1",
            action=ActionSearch(type="search", query="bitcoin"),
            status="completed",
            type="web_search_call",
        )
        events = [
            OpenAIReasoningEvent("Looking up bitcoin price", item=reasoning_item),
            OpenAIServerToolCallEvent(
                id="ws_1", name=WEB_SEARCH_TOOL_NAME, arguments=web_item.action.model_dump_json(), item=web_item
            ),
            OpenAIServerToolResultEvent(parent_id="ws_1", name=WEB_SEARCH_TOOL_NAME, result=ToolResult()),
        ]

        api_input = events_to_responses_input(events, serializer=None)  # type: ignore[arg-type]

        assert api_input == [
            reasoning_item.model_dump(exclude_none=True, mode="json"),
            web_item.model_dump(exclude_none=True, mode="json"),
        ]

    async def test_emits_one_event_per_summary(self) -> None:
        reasoning_item = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[
                Summary(type="summary_text", text="step one"),
                Summary(type="summary_text", text="step two"),
            ],
        )

        _, events = await _process([reasoning_item])

        assert events == [
            OpenAIReasoningEvent("step one", item=reasoning_item),
            OpenAIReasoningEvent("step two", item=reasoning_item),
        ]

    async def test_empty_summary_emits_anchor_event(self) -> None:
        # gpt-5 often returns reasoning with only encrypted_content and no
        # summary text; the item must still be persisted to keep round-trip.
        reasoning_item = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[],
        )

        _, events = await _process([reasoning_item])

        assert events == [OpenAIReasoningEvent("", item=reasoning_item)]

    async def test_per_summary_events_serialise_item_once(self) -> None:
        # Per-summary events share one underlying item; mapper must dedupe
        # by id, otherwise the API rejects the duplicate input.
        reasoning_item = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[
                Summary(type="summary_text", text="step one"),
                Summary(type="summary_text", text="step two"),
            ],
        )
        events = [
            OpenAIReasoningEvent("step one", item=reasoning_item),
            OpenAIReasoningEvent("step two", item=reasoning_item),
        ]

        api_input = events_to_responses_input(events, serializer=None)  # type: ignore[arg-type]

        assert api_input == [reasoning_item.model_dump(exclude_none=True, mode="json")]
