# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2ui import A2UIAction, A2UIClientCapabilities, A2UIServer, a2ui_action
from ag2.a2ui._runtime import _A2UIRuntime
from ag2.a2ui.actions import collect_action_declarations, collect_server_actions
from ag2.a2ui.dispatch import A2UIMessageFrame, A2UIProseFrame, stream_turn
from ag2.a2ui.request import parse_request
from ag2.a2ui.transports import RestTransport
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig, TrackingConfig

_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"


class _PromptCaptureConfig(ModelConfig):
    """Records the resolved ``context.prompt`` the agent sends to the model."""

    def __init__(self) -> None:
        self.prompts: list[list[str]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> "_PromptCaptureClient":
        return _PromptCaptureClient(self.prompts)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _PromptCaptureClient(LLMClient):
    def __init__(self, sink: list[list[str]]) -> None:
        self._sink = sink

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self._sink.append(list(context.prompt))
        return ModelResponse(ModelMessage("ok"))


_A2UI_RESPONSE = (
    "Here is your UI.\n<a2ui-json>\n"
    f'[{{"version": "v0.9", "createSurface": {{"surfaceId": "s1", "catalogId": "{_CATALOG}"}}}}]\n'
    "</a2ui-json>"
)


def _agent_and_runtime(config: Any = None, *, validate_responses: bool = True) -> "tuple[Agent, _A2UIRuntime]":
    """Build a plain Agent plus its A2UI runtime (the dispatch path's two inputs)."""
    agent = Agent(name="t", config=config)
    rt = _A2UIRuntime(validate_responses=validate_responses)
    return agent, rt


@pytest.mark.asyncio
class TestStreamTurn:
    async def test_plain_text_yields_single_prose_frame(self) -> None:
        agent, rt = _agent_and_runtime(TestConfig("Hello, no UI."), validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=rt.get_action)

        frames = [f async for f in stream_turn(agent, rt, req)]

        assert frames == [A2UIProseFrame("Hello, no UI.")]

    async def test_a2ui_response_yields_prose_then_message(self) -> None:
        agent, rt = _agent_and_runtime(TestConfig(_A2UI_RESPONSE), validate_responses=True)
        req = parse_request({"messages": [{"role": "user", "content": "show ui"}]}, resolve_action=rt.get_action)

        frames = [f async for f in stream_turn(agent, rt, req)]

        assert frames == [
            A2UIProseFrame("Here is your UI."),
            A2UIMessageFrame({"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": _CATALOG}}),
        ]

    async def test_missing_config_raises(self) -> None:
        agent, rt = _agent_and_runtime(None, validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=rt.get_action)

        with pytest.raises(RuntimeError, match="config is not set"):
            [f async for f in stream_turn(agent, rt, req)]

    async def test_client_capabilities_injected_into_prompt(self) -> None:
        config = _PromptCaptureConfig()
        agent, rt = _agent_and_runtime(config, validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=rt.get_action)
        req.client_capabilities = A2UIClientCapabilities(supported_catalog_ids=["https://other.example/c.json"])

        [_ async for _ in stream_turn(agent, rt, req)]

        joined = "\n".join(config.prompts[0])
        assert "## A2UI Client Capabilities" in joined
        assert "did NOT list" in joined

    async def test_no_capabilities_no_negotiation_prompt(self) -> None:
        config = _PromptCaptureConfig()
        agent, rt = _agent_and_runtime(config, validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=rt.get_action)

        [_ async for _ in stream_turn(agent, rt, req)]

        joined = "\n".join(config.prompts[0])
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
        assert "not instructions" not in "\n".join(config.calls[0].prompt)

    async def test_renderer_function_response_keeps_its_correlation_id(self) -> None:
        call_block = (
            '<a2ui-json>\n[{"version": "v1.0", "callRendererFunction": {"surfaceId": "s1", '
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
            {"a2ui": [{"version": "v1.0", "rendererFunctionResponse": {"functionCallId": "fc-7", "value": "Berlin"}}]},
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
        response = f'Checking.\n<a2ui-json>\n[{{"version": "v1.0", "callRendererFunction": {payload}}}]\n</a2ui-json>'
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
        agent, rt = _agent_and_runtime(TestConfig("ok"), validate_responses=False)
        req = parse_request(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "second"},
                ]
            },
            resolve_action=rt.get_action,
        )

        frames = [f async for f in stream_turn(agent, rt, req)]

        assert frames == [A2UIProseFrame("ok")]


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

        # If the agent ran it would emit this prose; asserting it is absent proves
        # the agent was skipped for a pure server-action turn.
        agent = Agent(name="t", config=TestConfig("AGENT SHOULD NOT RUN"))
        rt = _A2UIRuntime(actions=collect_action_declarations([add_to_basket]), validate_responses=False)
        req = parse_request(
            {"a2ui": [_server_action_envelope("add_to_basket", {"good_id": "G1"})]},
            resolve_action=rt.get_action,
        )

        frames = [f async for f in stream_turn(agent, rt, req, server_actions=collect_server_actions([add_to_basket]))]

        assert frames == [
            A2UIMessageFrame({
                "version": "v0.9",
                "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 1},
            }),
        ]

    async def test_server_action_value_is_dropped_because_a_click_has_no_response(self) -> None:
        @a2ui_action
        def typeahead(prefix: str) -> list[str]:
            return ["apple", "application"]

        agent = Agent(name="t", config=TestConfig("AGENT SHOULD NOT RUN"))
        rt = _A2UIRuntime(
            actions=collect_action_declarations([typeahead]),
            protocol_version="v1.0",
            validate_responses=False,
        )
        req = parse_request(
            {"a2ui": [_server_action_envelope("typeahead", {"prefix": "app"}, version="v1.0")]},
            resolve_action=rt.get_action,
            version_key="v1.0",
        )

        frames = [f async for f in stream_turn(agent, rt, req, server_actions=collect_server_actions([typeahead]))]

        assert frames == []

    async def test_server_action_alongside_user_message_also_runs_agent(self) -> None:
        @a2ui_action
        def add_to_basket(good_id: str) -> dict:
            return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 2}}

        agent = Agent(name="t", config=TestConfig("ok"))
        rt = _A2UIRuntime(actions=collect_action_declarations([add_to_basket]), validate_responses=False)
        req = parse_request(
            {
                "messages": [{"role": "user", "content": "and what else?"}],
                "a2ui": [_server_action_envelope("add_to_basket", {"good_id": "G1"})],
            },
            resolve_action=rt.get_action,
        )

        frames = [f async for f in stream_turn(agent, rt, req, server_actions=collect_server_actions([add_to_basket]))]

        # Server-action frame first, then the agent's prose.
        assert frames == [
            A2UIMessageFrame({
                "version": "v0.9",
                "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 2},
            }),
            A2UIProseFrame("ok"),
        ]


def _call_envelope(name: str, args: dict[str, Any], *, version: str = "v1.0", call_id: str = "fc-1") -> dict[str, Any]:
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
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2ui.test") as client:
        response = await client.post("/a2ui", json=body)
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
                "version": "v1.0",
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
            {"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": None}},
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
                "version": "v1.0",
                "agentFunctionResponse": {"functionCallId": "fc-1", "value": {"id": "PRV-1"}},
            },
        ]

    async def test_unknown_function_is_answered_with_an_error(self) -> None:
        messages, config = await _rpc_messages(_call_envelope("missing", {}))

        assert config.calls == []
        assert messages == [
            {
                "version": "v1.0",
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
                "version": "v1.0",
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
                "version": "v1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "error": {"code": "FUNCTION_FAILED", "message": "Function 'boom' failed."},
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
                "version": "v1.0",
                "updateDataModel": {"surfaceId": "cart", "path": "/count", "value": 3},
            },
            {"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": None}},
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
            {"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": "a"}},
            {"text": "ok"},
        ]
