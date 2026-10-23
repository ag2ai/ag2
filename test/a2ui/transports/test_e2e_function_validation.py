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
from ag2.events import ModelRequest, TextInput
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
async def test_unknown_function_argument_does_not_execute_handler_defaults() -> None:
    calls: list[int] = []

    @a2ui_action
    def buy(count: int = 1) -> int:
        calls.append(count)
        return count

    app = A2UIServer(
        Agent("ui", config=TestConfig("UNEXPECTED MODEL CALL")),
        actions=[buy],
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
                        "callFunction": {"@call": "buy", "args": {"coutn": 5}},
                    },
                }
            ]
        },
    )

    assert calls == []
    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {
                "functionCallId": "fc-1",
                "error": {"code": "INVALID_FUNCTION_CALL", "message": "Invalid arguments for function 'buy'."},
            },
        }
    ]


@pytest.mark.asyncio
async def test_function_accepting_extra_keywords_receives_them() -> None:
    @a2ui_action
    def collect(**values: Any) -> str:
        return values["value"]

    app = A2UIServer(
        Agent("ui", config=TestConfig("UNEXPECTED MODEL CALL")),
        actions=[collect],
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
                        "callFunction": {"@call": "collect", "args": {"value": "ok"}},
                    },
                }
            ]
        },
    )

    assert messages == [
        {
            "version": "v1.0",
            "agentFunctionResponse": {"functionCallId": "fc-1", "value": "ok"},
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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["basic_rpc", "basic_binding", "custom_rpc", "custom_binding", "unknown_rpc"])
async def test_function_schema_and_callers_follow_the_explicit_catalog(kind: str) -> None:
    basic_id = "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"
    custom_id = "https://example.com/catalog.json"
    catalog = {
        "$id": custom_id,
        "functions": {
            "openUrl": {
                "type": "object",
                "returnType": "void",
                "allowedCallers": "agentOnly",
                "properties": {
                    "@call": {"const": "openUrl"},
                    "args": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                },
                "required": ["@call", "args"],
            }
        },
    }
    selected = basic_id if kind.startswith("basic") else custom_id
    if kind == "unknown_rpc":
        selected = "https://unknown.example/catalog.json"
    call = {"@call": "openUrl", "catalogId": selected, "args": {"url": "https://example.com"}}
    if kind.endswith("binding"):
        operation = {
            "version": "v1.0",
            "createSurface": {
                "surfaceId": "s",
                "catalogId": custom_id,
                "components": [
                    {"id": "root", "component": "Button", "child": "label", "action": {"functionCall": call}},
                    {"id": "label", "component": "Text", "text": "Open"},
                ],
            },
        }
    else:
        operation = {"version": "v1.0", "callRendererFunction": {"functionCallId": "fc-1", "callFunction": call}}
    tracking = TrackingConfig(TestConfig(f"<a2ui-json>{json.dumps(operation)}</a2ui-json>", "Corrected."))
    app = A2UIServer(
        Agent("ui", config=tracking),
        protocol_version="v1.0",
        custom_catalog=catalog,
        transport=RestTransport(encoding="jsonl"),
    )

    messages = await _post(app, {"messages": [{"role": "user", "content": "Open the page"}]})

    if kind in ("basic_binding", "custom_rpc"):
        assert messages == [operation]
        assert len(tracking.calls) == 1
    else:
        assert messages == [{"text": "Corrected."}]
        assert len(tracking.calls) == 2


@pytest.mark.asyncio
async def test_identical_renderer_call_ids_do_not_mix_function_names_between_clients() -> None:
    catalog = _catalog()
    catalog["functions"]["otherLookup"] = {
        "type": "object",
        "returnType": "string",
        "allowedCallers": "agentOnly",
        "properties": {"@call": {"const": "otherLookup"}},
        "required": ["@call"],
    }
    agent = Agent("ui", config=TestConfig("unused"))
    app = A2UIServer(
        agent,
        protocol_version="v1.0",
        custom_catalog=catalog,
        transport=RestTransport(encoding="jsonl"),
    )
    for name in ("lookup", "otherLookup"):
        operation = {
            "version": "v1.0",
            "callRendererFunction": {
                "functionCallId": "fc-1",
                "callFunction": {"@call": name, "catalogId": catalog["$id"]},
            },
        }
        agent.config = TestConfig(f"<a2ui-json>{json.dumps(operation)}</a2ui-json>")
        await _post(app, {"messages": [{"role": "user", "content": "Look up"}]})
    tracking = TrackingConfig(TestConfig("Recorded."))
    agent.config = tracking

    await _post(
        app,
        {
            "a2ui": [
                {
                    "version": "v1.0",
                    "rendererFunctionResponse": {"functionCallId": "fc-1", "value": 23},
                }
            ]
        },
    )

    sent = tracking.mock.call_args.args[0]
    assert isinstance(sent, ModelRequest)
    text = " ".join(part.content for part in sent.parts if isinstance(part, TextInput))
    assert "fc-1" in text
    assert "23" in text
    assert "function=otherLookup" not in text
