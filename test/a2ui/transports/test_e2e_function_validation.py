# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest

from ag2 import Agent
from ag2.a2ui import A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.testing import TestConfig, TrackingConfig


def _catalog(allowed_callers: str = "agentOnly") -> dict[str, Any]:
    return {
        "$id": "https://example.com/catalog.json",
        "functions": {
            "lookup": {
                "type": "object",
                "returnType": "string",
                "allowedCallers": allowed_callers,
                "properties": {"@call": {"const": "lookup"}},
                "required": ["@call"],
            },
        },
    }


async def _post(app: A2UIServer, body: dict[str, Any]) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test") as client:
        response = await client.post("/a2ui", json=body)
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines() if line]


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [None, ["cancel"], "cancel", 7, True])
async def test_non_object_function_arguments_do_not_execute_handler(args: Any) -> None:
    calls: list[str] = []

    @a2ui_action
    def submit(choice: str = "default") -> str:
        calls.append(choice)
        return choice

    tracking = TrackingConfig(TestConfig("UNEXPECTED MODEL CALL"))
    app = A2UIServer(
        Agent("ui", config=tracking),
        actions=[submit],
        protocol_version="v1.0",
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(
        app,
        {
            "a2ui": [
                {
                    "version": "v1.0",
                    "callAgentFunction": {
                        "surfaceId": "s",
                        "functionCallId": "fc-1",
                        "callFunction": {"@call": "submit", "args": args},
                    },
                }
            ],
        },
    )

    assert calls == []
    assert tracking.calls == []
    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {
                "functionCallId": "fc-1",
                "error": {"code": "INVALID_FUNCTION_CALL", "message": "Invalid arguments for function 'submit'."},
            },
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_args", [False, True])
async def test_omitted_or_empty_function_arguments_use_handler_defaults(include_args: bool) -> None:
    @a2ui_action
    def submit(choice: str = "default") -> str:
        return choice

    call: dict[str, Any] = {"@call": "submit"}
    if include_args:
        call["args"] = {}
    app = A2UIServer(
        Agent("ui", config=TestConfig("UNEXPECTED MODEL CALL")),
        actions=[submit],
        protocol_version="v1.0",
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(
        app,
        {
            "a2ui": [
                {
                    "version": "v1.0",
                    "callAgentFunction": {"surfaceId": "s", "functionCallId": "fc-1", "callFunction": call},
                }
            ],
        },
    )

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {"functionCallId": "fc-1", "value": "default"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "catalog_id", [None, "https://example.com/catalog.json", "https://unrelated.example/catalog.json"]
)
async def test_function_calls_resolve_only_in_the_active_catalog(catalog_id: str | None) -> None:
    calls: list[str] = []

    @a2ui_action
    def lookup(choice: str = "default") -> str:
        calls.append(choice)
        return choice

    call: dict[str, Any] = {"@call": "lookup", "args": {}}
    if catalog_id is not None:
        call["catalogId"] = catalog_id
    tracking = TrackingConfig(TestConfig("UNEXPECTED MODEL CALL"))
    app = A2UIServer(
        Agent("ui", config=tracking),
        actions=[lookup],
        protocol_version="v1.0",
        custom_catalog=_catalog("rendererOrAgent"),
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(
        app,
        {
            "a2ui": [
                {
                    "version": "v1.0",
                    "callAgentFunction": {"surfaceId": "s", "functionCallId": "fc-1", "callFunction": call},
                }
            ],
        },
    )

    if catalog_id is None or catalog_id == "https://example.com/catalog.json":
        assert calls == ["default"]
        expected = {"functionCallId": "fc-1", "value": "default"}
    else:
        assert calls == []
        expected = {
            "functionCallId": "fc-1",
            "error": {"code": "UNKNOWN_FUNCTION", "message": "Unknown function 'lookup'."},
        }
    assert messages == [{"version": "v1.0", "agentFunctionResponse": expected}]
    assert tracking.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message_key", ["createSurface", "updateComponents"])
@pytest.mark.parametrize("binding", ["property", "action", "nested"])
async def test_agent_only_bindings_trigger_a_corrective_retry(message_key: str, binding: str) -> None:
    call = {"@call": "lookup"}
    components: list[dict[str, Any]]
    if binding == "action":
        components = [
            {"id": "root", "component": "Button", "child": "label", "action": {"functionCall": call}},
            {"id": "label", "component": "Text", "text": "Lookup"},
        ]
    else:
        value = {"@call": "formatString", "args": {"value": call}} if binding == "nested" else call
        components = [{"id": "root", "component": "Text", "text": value}]
    operation = {"version": "v1.0", message_key: {"surfaceId": "s", "components": components}}
    tracking = TrackingConfig(TestConfig(f"<a2ui-json>{json.dumps(operation)}</a2ui-json>", "Corrected."))
    app = A2UIServer(
        Agent("ui", config=tracking),
        protocol_version="v1.0",
        custom_catalog=_catalog(),
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(app, {"messages": [{"role": "user", "content": "Show lookup"}]})

    assert messages == [{"text": "Corrected."}]
    assert len(tracking.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_callers", ["rendererOnly", "rendererOrAgent"])
async def test_renderer_callable_bindings_are_emitted(allowed_callers: str) -> None:
    operation = {
        "version": "v1.0",
        "createSurface": {
            "surfaceId": "s",
            "components": [{"id": "root", "component": "Text", "text": {"@call": "lookup"}}],
        },
    }
    tracking = TrackingConfig(TestConfig(f"<a2ui-json>{json.dumps(operation)}</a2ui-json>"))
    app = A2UIServer(
        Agent("ui", config=tracking),
        protocol_version="v1.0",
        custom_catalog=_catalog(allowed_callers),
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(app, {"messages": [{"role": "user", "content": "Show lookup"}]})

    assert messages == [operation]
    assert len(tracking.calls) == 1


@pytest.mark.asyncio
async def test_agent_only_rpc_and_literal_call_objects_are_emitted() -> None:
    operations = [
        {
            "version": "v1.0",
            "createSurface": {
                "surfaceId": "s",
                "dataModel": {"literal": {"@call": "lookup"}},
                "metadata": {"extensions": {"literal": {"@call": "lookup"}}},
                "components": [
                    {
                        "id": "root",
                        "component": "Text",
                        "text": "Lookup",
                        "metadata": {"extensions": {"literal": {"@call": "lookup"}}},
                    }
                ],
            },
        },
        {
            "version": "v1.0",
            "callRendererFunction": {
                "functionCallId": "fc-1",
                "callFunction": {"@call": "lookup", "catalogId": "https://example.com/catalog.json"},
            },
        },
    ]
    tracking = TrackingConfig(TestConfig(f"<a2ui-json>{json.dumps(operations)}</a2ui-json>"))
    app = A2UIServer(
        Agent("ui", config=tracking),
        protocol_version="v1.0",
        custom_catalog=_catalog(),
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(app, {"messages": [{"role": "user", "content": "Look up"}]})

    assert messages == operations
    assert len(tracking.calls) == 1
