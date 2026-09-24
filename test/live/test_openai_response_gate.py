# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("openai")

from openai.types.realtime import RealtimeConversationItemFunctionCall, ResponseOutputItemDoneEvent

from ag2.events import ToolResult, ToolResultEvent
from test.live._helpers import active_response_rejection, created, done, live_agent


def function_call(response_id: str, call_id: str) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent(
        event_id=f"ev-call-{call_id}",
        item=RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id=call_id,
            name="lookup",
            arguments="{}",
        ),
        output_index=0,
        response_id=response_id,
        type="response.output_item.done",
    )


def lookup() -> str:
    """Look the answer up."""
    return "42"


@pytest.mark.asyncio
class TestResponseGate:
    async def test_tool_result_while_idle_requests_response_immediately(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))

            assert conn.response_requests() == 1

    async def test_unacknowledged_request_counts_as_active(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 1

            await conn.emit(created("resp-1"), done("resp-1"))

            assert conn.response_requests() == 2

    async def test_rejected_request_is_retried_at_boundary(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(
                created("resp-vad"),
                active_response_rejection(conn.last_response_request_id()),
                done("resp-vad"),
            )

            assert conn.response_requests() == 2

    async def test_rejected_request_does_not_block_later_requests(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(active_response_rejection(conn.last_response_request_id()))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 2

    async def test_error_for_another_client_event_leaves_request_active(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run() as context:
            await context.send(ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult("42")))
            await conn.emit(active_response_rejection("another-client-event"))
            await context.send(ToolResultEvent(parent_id="call-2", name="lookup", result=ToolResult("43")))

            assert conn.response_requests() == 1

    async def test_tool_result_during_response_waits_for_boundary(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run():
            await conn.emit(created("resp-1"), function_call("resp-1", "call-1"))

            assert (
                "conversation.item.create",
                {"item": {"type": "function_call_output", "call_id": "call-1", "output": "42"}},
            ) in conn.calls
            assert conn.response_requests() == 0

            await conn.emit(done("resp-1"))

            assert conn.response_requests() == 1

    async def test_tool_results_during_one_response_share_one_request(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run():
            await conn.emit(
                created("resp-1"),
                function_call("resp-1", "call-1"),
                function_call("resp-1", "call-2"),
                done("resp-1"),
            )

            assert conn.response_requests() == 1

    async def test_deferred_request_dropped_when_turn_detection_response_is_active(self) -> None:
        agent, conn = live_agent(lookup)

        async with agent.run():
            await conn.emit(
                created("resp-1"),
                function_call("resp-1", "call-1"),
                created("resp-vad"),
                done("resp-1"),
                done("resp-vad"),
            )

            assert conn.response_requests() == 0
