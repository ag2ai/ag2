# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from ag2 import Agent, Depends, Inject
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig


class BackendRecord(BaseModel):
    quantity: int


def _backend_record(quantity: int, state: Annotated[dict[str, int], Inject("state")]) -> BackendRecord:
    state["calls"] += 1
    return BackendRecord.model_validate({"quantity": "invalid backend value"})


def _read_backend(quantity: int, state: Annotated[dict[str, int], Inject("state")]) -> dict[str, Any]:
    return _backend_record(quantity, state).model_dump()


async def _read_backend_async(quantity: int, state: Annotated[dict[str, int], Inject("state")]) -> dict[str, Any]:
    return _read_backend(quantity, state)


def _read_backend_dependency(record: Annotated[BackendRecord, Depends(_backend_record)]) -> dict[str, Any]:
    return record.model_dump()


async def _read_backend_dependency_async(record: Annotated[BackendRecord, Depends(_backend_record)]) -> dict[str, Any]:
    return record.model_dump()


def _replacement_backend_record(quantity: int) -> BackendRecord:
    return BackendRecord(quantity=quantity * 2)


async def _call_backend(
    mode: Literal["async", "threaded", "inline"], quantity: int | str, *, dependency: bool = False
) -> tuple[list[dict[str, Any]], dict[str, int], TrackingConfig]:
    state = {"calls": 0}
    handler: Callable[..., Any]
    if dependency:
        handler = _read_backend_dependency_async if mode == "async" else _read_backend_dependency
    else:
        handler = _read_backend_async if mode == "async" else _read_backend
    action = a2ui_action(handler, name="read_backend", sync_to_thread=mode != "inline")
    tracking = TrackingConfig(TestConfig("UNEXPECTED MODEL CALL"))
    app = A2UIServer(
        Agent("ui", config=tracking, dependencies={"state": state}),
        actions=[action],
        protocol_version="v1.0",
        transport=RestTransport(encoding="jsonl"),
    )
    return await _post_function(app, quantity), state, tracking


async def _post_function(app: A2UIServer, quantity: int | str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test") as client:
        response = await client.post(
            "/a2ui",
            json={
                "a2ui": [
                    {
                        "version": "v1.0",
                        "callAgentFunction": {
                            "surfaceId": "s",
                            "functionCallId": "backend-1",
                            "callFunction": {"@call": "read_backend", "args": {"quantity": quantity}},
                        },
                    }
                ]
            },
        )
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines() if line]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["async", "threaded", "inline"])
@pytest.mark.parametrize("dependency", [False, True])
async def test_backend_validation_error_is_a_function_failure(
    mode: Literal["async", "threaded", "inline"], dependency: bool
) -> None:
    messages, state, tracking = await _call_backend(mode, 1, dependency=dependency)

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {
                "functionCallId": "backend-1",
                "error": {"code": "FUNCTION_FAILED", "message": "Function 'read_backend' failed."},
            },
        }
    ]
    assert state == {"calls": 1}
    assert tracking.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["async", "threaded", "inline"])
@pytest.mark.parametrize("dependency", [False, True])
async def test_invalid_function_input_is_rejected_before_handler_runs(
    mode: Literal["async", "threaded", "inline"], dependency: bool
) -> None:
    messages, state, tracking = await _call_backend(mode, "invalid input", dependency=dependency)

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {
                "functionCallId": "backend-1",
                "error": {"code": "INVALID_FUNCTION_CALL", "message": "Invalid arguments for function 'read_backend'."},
            },
        }
    ]
    assert state == {"calls": 0}
    assert tracking.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action_first", [True, False])
async def test_action_and_tool_keep_their_validation_errors_when_sharing_a_dependency(action_first: bool) -> None:
    state = {"calls": 0}
    action = a2ui_action(_read_backend_dependency, name="read_backend") if action_first else None
    agent = Agent(
        "ui",
        config=TestConfig(
            ToolCallEvent(name="_read_backend_dependency", arguments=json.dumps({"quantity": "invalid input"})),
            "unused",
        ),
        dependencies={"state": state},
        tools=[_read_backend_dependency],
    )
    if action is None:
        action = a2ui_action(_read_backend_dependency, name="read_backend")
    app = A2UIServer(agent, actions=[action], protocol_version="v1.0", transport=RestTransport(encoding="jsonl"))

    messages = await _post_function(app, "invalid input")

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {
                "functionCallId": "backend-1",
                "error": {"code": "INVALID_FUNCTION_CALL", "message": "Invalid arguments for function 'read_backend'."},
            },
        }
    ]
    with pytest.raises(ValidationError):
        await agent.ask("Run the tool")
    assert state == {"calls": 0}


@pytest.mark.asyncio
async def test_action_uses_agent_dependency_overrides_with_its_own_dependency_models() -> None:
    action = a2ui_action(_read_backend_dependency, name="read_backend")
    tracking = TrackingConfig(TestConfig("UNEXPECTED MODEL CALL"))
    agent = Agent("ui", config=tracking)
    agent.dependency_provider.override(_backend_record, _replacement_backend_record)
    app = A2UIServer(agent, actions=[action], protocol_version="v1.0", transport=RestTransport(encoding="jsonl"))

    messages = await _post_function(app, 2)

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {"functionCallId": "backend-1", "value": {"quantity": 4}},
        }
    ]
    assert tracking.calls == []
