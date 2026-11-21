# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Sequence
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel

from ag2 import Agent
from ag2.a2ui import A2UIAction, A2UIServer, a2ui_action
from ag2.a2ui.transports import RestTransport
from ag2.events import ModelRequest, TextInput
from ag2.testing import TestConfig, TrackingConfig

from .seam import from_renderer

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"


_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)
_HI: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}


async def _serve(
    body: dict[str, Any],
    config: Any,
    actions: Sequence[A2UIAction] = (),
    **server_kwargs: Any,
) -> list[dict[str, Any]]:
    """Post ``body`` to an ``A2UIServer`` whose agent runs on ``config``; return the JSONL frames."""
    app = A2UIServer(
        Agent(name="t", config=config),
        actions=actions,
        transport=RestTransport(encoding="jsonl"),
        **server_kwargs,
    )
    return await _post_turn(app, body)


@pytest.mark.asyncio
class TestStreamTurn:
    async def test_plain_text_yields_single_prose_frame(self) -> None:
        frames = await _serve(_HI, TestConfig("Hello, no UI."), validate_responses=False)

        assert frames == [{"text": "Hello, no UI."}]

    async def test_a2ui_response_yields_prose_then_message(self) -> None:
        frames = await _serve(
            {"messages": [{"role": "user", "content": "show ui"}]}, TestConfig(_A2UI_RESPONSE), validate_responses=True
        )

        assert frames == [
            {"text": "Here is your UI."},
            {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": _CATALOG}},
        ]

    async def test_missing_config_reports_an_error_frame(self) -> None:
        frames = await _serve(_HI, None, validate_responses=False)

        assert frames == [{"error": "Agent.config is not set; cannot serve over REST"}]

    async def test_client_capabilities_injected_into_prompt(self) -> None:
        config = TrackingConfig(TestConfig("ok"))
        supported = ["https://other.example/c.json", "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"]
        body = {**_HI, "a2uiClientCapabilities": {"v0.9": {"supportedCatalogIds": supported}}}

        await _serve(body, config, validate_responses=False)

        joined = "\n".join(config.calls[0].prompt)
        assert "## A2UI Client Capabilities" in joined
        assert "did NOT list" not in joined

    async def test_client_without_the_agents_catalog_gets_no_a2ui(self, caplog: pytest.LogCaptureFixture) -> None:
        # As the official SDK's resolve_catalogs: no catalog in common, nothing to render with.
        config = TrackingConfig(TestConfig("ok"))
        body = {**_HI, "a2uiClientCapabilities": {"v0.9": {"supportedCatalogIds": ["https://other.example/c.json"]}}}

        with caplog.at_level(logging.WARNING):
            await _serve(body, config, validate_responses=False)

        joined = "\n".join(config.calls[0].prompt)
        assert "A2UI Response Format" not in joined
        assert "## A2UI Client Capabilities" not in joined
        assert any("supports none of the agent's catalogs" in r.message for r in caplog.records)

    async def test_no_capabilities_no_negotiation_prompt(self) -> None:
        config = TrackingConfig(TestConfig("ok"))

        await _serve(_HI, config, validate_responses=False)

        joined = "\n".join(config.calls[0].prompt)
        assert "## A2UI Client Capabilities" not in joined

    async def test_client_data_model_injected_into_prompt(self) -> None:
        config = TrackingConfig(TestConfig("ok"))
        app = A2UIServer(
            Agent(name="t", config=config),
            transport=RestTransport(encoding="jsonl"),
            validate_responses=False,
        )

        messages = await _post_turn(
            app,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "a2uiClientDataModel": {"version": "v0.9", "surfaces": {"form": {"name": "Ann"}}},
            },
        )

        assert messages == [{"text": "ok"}]
        joined = "\n".join(config.calls[0].prompt)
        assert '{"form": {"name": "Ann"}}' in joined
        assert "not instructions" in joined

    async def test_no_data_model_no_data_model_prompt(self) -> None:
        config = TrackingConfig(TestConfig("ok"))
        app = A2UIServer(
            Agent(name="t", config=config),
            transport=RestTransport(encoding="jsonl"),
            validate_responses=False,
        )

        messages = await _post_turn(app, {"messages": [{"role": "user", "content": "hi"}]})

        assert messages == [{"text": "ok"}]
        [call] = config.calls
        assert "not instructions" not in "\n".join(call.prompt)

    async def test_renderer_function_response_keeps_its_correlation_id(self) -> None:
        call_block = (
            '<a2ui-json>\n[{"version": "1.0", "callRendererFunction": {"surfaceId": "s1", '
            '"functionCallId": "fc-7", "callFunction": {"@call": "getLocation"}}}]\n</a2ui-json>'
        )
        agent = Agent(name="t", config=TestConfig(call_block))
        app = A2UIServer(
            agent,
            protocol_version="v1.0",
            transport=RestTransport(encoding="jsonl"),
            validate_responses=False,
        )
        await _post_turn(app, {"messages": [{"role": "user", "content": "where am I"}]})
        config = TrackingConfig(TestConfig("Recorded."))
        agent.config = config

        messages = await _post_turn(
            app,
            {"a2ui": [{"version": "1.0", "rendererFunctionResponse": {"functionCallId": "fc-7", "value": "Berlin"}}]},
        )

        assert messages == [{"text": "Recorded."}]
        sent = config.mock.call_args.args[0]
        assert isinstance(sent, ModelRequest)
        prompt = " ".join(part.content for part in sent.parts if isinstance(part, TextInput))
        assert "fc-7" in prompt
        assert "function=" not in prompt
        assert "Berlin" in prompt

    @pytest.mark.parametrize(
        "payload",
        ["null", '"text"', '{"functionCallId": "fc-1", "callFunction": "getLocation"}', '{"functionCallId": 7}'],
    )
    async def test_malformed_renderer_call_does_not_fail_the_turn(self, payload: str) -> None:
        response = f'Checking.\n<a2ui-json>\n[{{"version": "1.0", "callRendererFunction": {payload}}}]\n</a2ui-json>'
        config = TrackingConfig(TestConfig(response))
        app = A2UIServer(
            Agent(name="t", config=config),
            protocol_version="v1.0",
            transport=RestTransport(encoding="jsonl"),
            validate_responses=False,
        )

        messages = await _post_turn(app, {"messages": [{"role": "user", "content": "where am I"}]})

        assert [message for message in messages if "text" in message] == [{"text": "Checking."}]
        assert len(config.calls) == 1

    async def test_history_is_stateless_per_turn(self) -> None:
        # A fresh stream per call: prior history sent in the body, not retained.
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "second"},
            ]
        }

        frames = await _serve(body, TestConfig("ok"), validate_responses=False)

        assert frames == [{"text": "ok"}]


def _server_action_envelope(
    name: str, context: dict[str, Any], *, version: str = "v0.9", **extra: Any
) -> dict[str, Any]:
    return {
        "version": version,
        "action": {
            "name": name,
            "surfaceId": "s",
            "sourceComponentId": "c",
            "timestamp": "2026-06-19T00:00:00Z",
            "context": context,
            **extra,
        },
    }


@pytest.mark.asyncio
class TestServerActions:
    async def test_server_action_click_runs_handler_and_skips_agent(self) -> None:
        @a2ui_action
        def add_to_basket(good_id: str) -> dict:
            return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}}

        config = TrackingConfig(TestConfig("AGENT SHOULD NOT RUN"))
        body = {"a2ui": [_server_action_envelope("add_to_basket", {"good_id": "G1"})]}

        frames = await _serve(body, config, [add_to_basket], validate_responses=False)

        assert config.calls == []
        assert frames == [
            {"version": "v0.9", "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1}},
        ]

    async def test_server_action_value_is_dropped_because_a_click_has_no_response(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> list[str]:
            return ["apple", "application"]

        config = TrackingConfig(TestConfig("AGENT SHOULD NOT RUN"))
        body = {"a2ui": [_server_action_envelope("typeahead", {"prefix": "app"}, version="1.0")]}

        frames = await _serve(body, config, [typeahead], protocol_version="v1.0", validate_responses=False)

        assert config.calls == []
        assert frames == []

    async def test_server_action_alongside_user_message_also_runs_agent(self) -> None:
        @a2ui_action
        def add_to_basket(good_id: str) -> dict:
            return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 2}}

        body = {
            "messages": [{"role": "user", "content": "and what else?"}],
            "a2ui": [_server_action_envelope("add_to_basket", {"good_id": "G1"})],
        }

        frames = await _serve(body, TestConfig("ok"), [add_to_basket], validate_responses=False)

        # Server-action frame first, then the agent's prose.
        assert frames == [
            {"version": "v0.9", "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 2}},
            {"text": "ok"},
        ]


def _call_envelope(name: str, args: dict[str, Any], *, version: str = "1.0", call_id: str = "fc-1") -> dict[str, Any]:
    return {
        "version": version,
        "callAgentFunction": {
            "surfaceId": "s",
            "functionCallId": call_id,
            "callFunction": {"@call": name, "args": args},
        },
    }


async def _rpc_messages(
    envelope: dict[str, Any],
    actions: Sequence[A2UIAction] = (),
    *,
    version: Literal["v0.9", "v1.0"] = "v1.0",
    user_message: str | None = None,
) -> tuple[list[dict[str, Any]], TrackingConfig]:
    config = TrackingConfig(TestConfig("ok"))
    app = A2UIServer(
        Agent(name="t", config=config),
        actions=actions,
        protocol_version=version,
        transport=RestTransport(encoding="jsonl"),
    )
    body: dict[str, Any] = {"a2ui": [envelope]}
    if user_message is not None:
        body["messages"] = [{"role": "user", "content": user_message}]
    return await _post_turn(app, body), config


async def _post_turn(app: A2UIServer, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Post ``body`` from a renderer that supports the basic catalog."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test") as client:
        response = await client.post("/a2ui", json=from_renderer(body))
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines() if line]


@pytest.mark.asyncio
class TestCallAgentFunction:
    async def test_registered_function_answers_with_its_value_and_skips_agent(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> list[str]:
            return ["apple", "application"]

        messages, config = await _rpc_messages(_call_envelope("typeahead", {"prefix": "app"}), [typeahead])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {"functionCallId": "fc-1", "value": ["apple", "application"]},
            },
        ]

    async def test_none_result_is_answered_with_a_null_value(self) -> None:
        @a2ui_action
        def noop() -> None:
            return None

        messages, config = await _rpc_messages(_call_envelope("noop", {}), [noop])

        assert config.calls == []
        assert messages == [
            {"version": "1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": None}},
        ]

    async def test_models_are_serialized_to_json(self) -> None:
        class Provider(BaseModel):
            id: str

        @a2ui_action
        def lookup(provider_id: str) -> Provider:
            return Provider(id=provider_id)

        messages, config = await _rpc_messages(_call_envelope("lookup", {"provider_id": "PRV-1"}), [lookup])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {"functionCallId": "fc-1", "value": {"id": "PRV-1"}},
            },
        ]

    async def test_unknown_function_is_answered_with_an_error(self) -> None:
        messages, config = await _rpc_messages(_call_envelope("missing", {}))

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "error": {"code": "UNKNOWN_FUNCTION", "message": "Unknown function 'missing'."},
                },
            },
        ]

    async def test_invalid_arguments_are_answered_with_an_error(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> str:
            return prefix

        messages, config = await _rpc_messages(_call_envelope("typeahead", {}), [typeahead])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "error": {
                        "code": "INVALID_FUNCTION_CALL",
                        "message": "Invalid arguments for function 'typeahead'.",
                    },
                },
            },
        ]

    async def test_failing_handler_does_not_leak_its_message(self) -> None:
        @a2ui_action
        def boom() -> str:
            raise RuntimeError("backend down")

        messages, config = await _rpc_messages(_call_envelope("boom", {}), [boom])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "error": {"code": "EXECUTION_ERROR", "message": "Function 'boom' failed."},
                },
            },
        ]

    async def test_returned_messages_precede_the_response(self) -> None:
        @a2ui_action
        def refresh() -> dict:
            return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 3}}

        messages, config = await _rpc_messages(_call_envelope("refresh", {}), [refresh])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 3},
            },
            {"version": "1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": None}},
        ]

    async def test_call_before_v1_0_is_left_to_the_agent_unanswered(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> str:
            return prefix

        messages, config = await _rpc_messages(
            _call_envelope("typeahead", {"prefix": "a"}, version="v0.9"), [typeahead], version="v0.9"
        )

        assert messages == [{"text": "ok"}]
        assert len(config.calls) == 1

    async def test_call_alongside_a_user_message_also_runs_the_agent(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> str:
            return prefix

        messages, config = await _rpc_messages(
            _call_envelope("typeahead", {"prefix": "a"}), [typeahead], user_message="and then?"
        )

        assert len(config.calls) == 1
        assert messages == [
            {"version": "1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": "a"}},
            {"text": "ok"},
        ]


async def _click_logs(result: Any, caplog: pytest.LogCaptureFixture) -> tuple[list[dict[str, Any]], str]:
    """Click a server action that returns ``result``; the frames sent and the server-action warnings logged."""

    @a2ui_action
    def handler() -> Any:
        return result

    body = {"a2ui": [_server_action_envelope("handler", {})]}
    with caplog.at_level("WARNING", logger="ag2.a2ui.server_action"):
        frames = await _serve(body, TrackingConfig(TestConfig("unused")), [handler], validate_responses=False)
    return frames, caplog.text


@pytest.mark.asyncio
class TestMalformedHandlerMessages:
    """A handler result that is almost an A2UI message is not sent, and that is logged rather than silent."""

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(
                {
                    "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1},
                    "deleteSurface": {"surfaceId": "old"},
                },
                id="several-message-keys-in-one-object",
            ),
            pytest.param([{"deleteSurface": {"surfaceId": "old"}}, "not a message"], id="list-mixing-messages"),
        ],
    )
    async def test_is_dropped_with_a_warning(self, result: Any, caplog: pytest.LogCaptureFixture) -> None:
        frames, logs = await _click_logs(result, caplog)

        assert frames == []
        assert "'handler'" in logs
        assert "one message per object" in logs

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({"saved": True}, id="plain-object"),
            pytest.param({"deleteSurface": True, "updateDataModel": "ok"}, id="message-keys-without-objects"),
        ],
    )
    async def test_a_plain_value_is_not_warned_about(self, result: Any, caplog: pytest.LogCaptureFixture) -> None:
        frames, logs = await _click_logs(result, caplog)

        assert frames == []
        assert logs == ""


_RELATIVE_PATH_UPDATE: dict[str, Any] = {"updateDataModel": {"surfaceId": "cart", "path": "count", "value": 1}}


@pytest.mark.asyncio
class TestHandlerMessagesAreValidated:
    """Messages a handler returns are server→client messages, so they are held to the spec like the model's."""

    async def test_an_invalid_click_message_is_dropped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        @a2ui_action
        def refresh() -> dict[str, Any]:
            return _RELATIVE_PATH_UPDATE

        body = {"a2ui": [_server_action_envelope("refresh", {})]}
        with caplog.at_level("WARNING", logger="ag2.a2ui.server_action"):
            frames = await _serve(body, TrackingConfig(TestConfig("unused")), [refresh], validate_responses=True)

        assert frames == []
        assert "'refresh'" in caplog.text
        assert "JSON Pointer" in caplog.text

    async def test_validation_off_sends_it_as_is(self) -> None:
        @a2ui_action
        def refresh() -> dict[str, Any]:
            return _RELATIVE_PATH_UPDATE

        body = {"a2ui": [_server_action_envelope("refresh", {})]}
        frames = await _serve(body, TrackingConfig(TestConfig("unused")), [refresh], validate_responses=False)

        assert frames == [{"version": "v0.9", **_RELATIVE_PATH_UPDATE}]

    async def test_an_invalid_function_message_fails_the_call(self) -> None:
        @a2ui_action
        def refresh() -> dict[str, Any]:
            return _RELATIVE_PATH_UPDATE

        messages, config = await _rpc_messages(_call_envelope("refresh", {}), [refresh])

        assert config.calls == []
        assert messages == [
            {
                "version": "1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "error": {"code": "EXECUTION_ERROR", "message": "Function 'refresh' failed."},
                },
            },
        ]


@pytest.mark.asyncio
async def test_a_handler_may_partially_update_a_surface_from_an_earlier_turn() -> None:
    update = {"updateComponents": {"surfaceId": "s1", "components": [{"id": "t1", "component": "Text", "text": "hi"}]}}

    @a2ui_action
    def refresh() -> dict[str, Any]:
        return update

    body = {"a2ui": [_server_action_envelope("refresh", {})]}
    frames = await _serve(body, TrackingConfig(TestConfig("unused")), [refresh], validate_responses=True)

    assert frames == [{"version": "v0.9", **update}]
