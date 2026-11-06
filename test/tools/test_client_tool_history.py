# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from ag2 import Agent, Context, events, testing, tool
from ag2.middleware import BaseMiddleware, Middleware, ToolExecution, ToolResultType
from ag2.tools.final.client_tool import ClientTool


@pytest.mark.asyncio
@pytest.mark.parametrize("server_outcome", ["none", "success", "error"])
@pytest.mark.parametrize("round_trip", [False, True])
async def test_client_handoff_replays_each_call_and_completed_result_once(
    server_outcome: str, round_trip: bool
) -> None:
    @tool
    def server_step() -> str:
        if server_outcome == "error":
            raise ValueError("server failed")
        return "server evidence"

    client = ClientTool({"function": {"name": "client_step", "description": "client", "parameters": {}}})
    client_call = events.ToolCallEvent("client_step", id="client", vendor_metadata={"caller": {"type": "direct"}})
    calls = [client_call]
    if server_outcome != "none":
        calls.insert(0, events.ToolCallEvent("server_step", id="server"))
    agent = Agent("mixed", config=testing.TestConfig(calls), tools=[server_step, client])

    reply = await agent.ask("work")
    assert reply.response.tool_calls == events.ToolCallsEvent([client_call])
    history = list(await reply.context.stream.history.get_events())
    if round_trip:
        history = [type(event).from_dict(json.loads(json.dumps(event.to_dict()))) for event in history]

    client_result = events.ToolResultEvent.from_call(client_call, "client evidence")
    tracking = testing.TrackingConfig(testing.TestConfig("done", raise_tool_errors=False))
    await agent.resume(*history, events.ToolResultsEvent([client_result]), config=tracking)

    messages = tracking.calls[0].messages
    model_responses = [event for event in messages if isinstance(event, events.ModelResponse)]
    assert model_responses == [events.ModelResponse(tool_calls=events.ToolCallsEvent(calls))]
    results = [result for event in messages if isinstance(event, events.ToolResultsEvent) for result in event.results]
    expected = []
    if server_outcome != "none":
        expected.append(
            events.ToolErrorEvent.from_call(
                calls[0], Exception("server failed") if round_trip else ValueError("server failed")
            )
            if server_outcome == "error"
            else events.ToolResultEvent.from_call(calls[0], "server evidence")
        )
    expected.append(client_result)
    assert results == expected


class SignedCall(events.ToolCallEvent):
    signature: str


class ApproveClientCall(BaseMiddleware):
    async def on_tool_execution(
        self, call_next: ToolExecution, event: events.ToolCallEvent, context: Context
    ) -> ToolResultType:
        approved = events.ToolCallEvent(id=event.id, name="approved_tool", arguments='{"path":"/allowed"}')
        return await call_next(approved, context)


@pytest.mark.asyncio
async def test_handoff_replay_preserves_approved_calls_and_stored_events() -> None:
    call = SignedCall("client_step", id="client", arguments='{"path":"/original"}', signature="provider-signature")
    client = ClientTool({"function": {"name": "client_step", "description": "client", "parameters": {}}})
    agent = Agent("client", config=testing.TestConfig(call), tools=[client], middleware=[Middleware(ApproveClientCall)])
    reply = await agent.ask("work")
    history = list(await reply.context.stream.history.get_events())
    original = [event for event in history if isinstance(event, events.ModelResponse)]
    approved = SignedCall("approved_tool", id="client", arguments='{"path":"/allowed"}', signature="provider-signature")
    tracking = testing.TrackingConfig(testing.TestConfig("done"))

    await agent.resume(
        *history, events.ToolResultsEvent([events.ToolResultEvent.from_call(approved, "done")]), config=tracking
    )

    assert [event for event in tracking.calls[0].messages if isinstance(event, events.ModelResponse)] == [
        events.ModelResponse(tool_calls=events.ToolCallsEvent([approved]))
    ]
    assert original == [
        events.ModelResponse(tool_calls=events.ToolCallsEvent([call])),
        events.ModelResponse(tool_calls=events.ToolCallsEvent([approved]), response_force=True),
    ]
    assert reply.response.tool_calls == events.ToolCallsEvent([approved])


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_previous_call", [False, True])
async def test_handoff_without_pending_original_retains_its_call(completed_previous_call: bool) -> None:
    call = events.ToolCallEvent("client_step", id="reused")
    handoff = events.ModelResponse(tool_calls=events.ToolCallsEvent([call]), response_force=True)
    prefix = []
    if completed_previous_call:
        previous = events.ToolCallEvent("previous_tool", id="reused")
        prefix = [
            events.ModelResponse(tool_calls=events.ToolCallsEvent([previous])),
            events.ToolResultsEvent([events.ToolResultEvent.from_call(previous, "previous result")]),
        ]
    config = testing.TrackingConfig(testing.TestConfig("done"))
    agent = Agent("client", config=config)

    await agent.resume(
        *prefix, handoff, events.ToolResultsEvent([events.ToolResultEvent.from_call(call, "client result")])
    )

    assert [event for event in config.calls[0].messages if isinstance(event, events.ModelResponse)] == [
        *[event for event in prefix if isinstance(event, events.ModelResponse)],
        handoff,
    ]
