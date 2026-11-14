# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

from ag2 import Agent, Context, DataInput, MemoryStream, tool
from ag2.events import ModelRequest, ModelResponse, ToolCallEvent
from ag2.middleware import statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware(persist=True)
async def current_input_calls(context: Context) -> DataInput:
    count = await context.stream.history.since_last_user_input().where(ToolCallEvent).count()
    return DataInput({"calls": count}, metadata={"case": "input-scope"})


@tool
def queue_update(context: Context) -> str:
    """Queue input for delivery after this tool completes."""
    context.enqueue("mid-turn update")
    return "queued"


async def queue_after_completion(event: ModelResponse, context: Context) -> None:
    if event.message is not None and event.message.content == "first completion":
        context.enqueue("final inbox update")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "01_phone_call_limit",
        "02_time_and_timeline",
        "03_todo_plan",
        "04_error_recovery",
        "05_environment",
        "06_user_turn_counters",
        "07_external_multimodal",
        "08_persistent_progress",
        "09_mixed_persistence",
        "10_conditional_progress",
        "11_retry_snapshot",
    ],
)
async def test_statusbar_example_entrypoint(scenario: str) -> None:
    example = importlib.import_module(f"examples.status_middleware_usecases.{scenario}")

    await example.main()


@pytest.mark.asyncio
async def test_persistent_status_scope_survives_run_and_both_inbox_drains() -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent("queue_update"),
            "first completion",
            "final completion",
            "continuation",
        )
    )
    stream = MemoryStream()
    stream.where(ModelResponse).subscribe(queue_after_completion)
    agent = Agent("scope", config=config, tools=[queue_update], middleware=[current_input_calls])

    async with agent.run("question", stream=stream) as run:
        with run.stream.where(ModelResponse).join(max_events=3) as responses:
            reply = await run.result()
            observed = [event async for event in responses]
    assert len(observed) == 3
    assert reply.body == "final completion"
    assert await reply.history.since_last_user_input().where(ToolCallEvent).count() == 1

    await reply.ask("next question")

    snapshots = [
        tuple(
            part.data["calls"]
            for event in call.messages
            for part in getattr(event, "parts", ())
            if isinstance(part, DataInput) and part.metadata.get("case") == "input-scope"
        )
        for call in config.calls
    ]
    assert snapshots == [(0,), (0, 1), (0, 1, 1), (0, 1, 1, 0)]
    assert await reply.history.since_last_user_input().where(ToolCallEvent).count() == 0
    assert [request.origin for request in await reply.history.where(ModelRequest).all()] == [
        "user",
        "service",
        "service",
        "user",
    ]
