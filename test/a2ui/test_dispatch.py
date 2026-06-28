# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import pytest
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2ui import A2UIClientCapabilities, a2ui_action
from ag2.a2ui._runtime import _A2UIRuntime
from ag2.a2ui.actions import collect_action_declarations, collect_server_actions
from ag2.a2ui.dispatch import A2UIMessageFrame, A2UIProseFrame, stream_turn
from ag2.a2ui.request import parse_request
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.testing import TestConfig

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

    async def test_server_action_with_want_response_emits_action_response(self) -> None:
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
            {
                "a2ui": [
                    _server_action_envelope(
                        "typeahead", {"prefix": "app"}, version="v1.0", wantResponse=True, actionId="a-1"
                    )
                ]
            },
            resolve_action=rt.get_action,
            version_key="v1.0",
        )

        frames = [f async for f in stream_turn(agent, rt, req, server_actions=collect_server_actions([typeahead]))]

        assert frames == [
            A2UIMessageFrame({
                "version": "v1.0",
                "actionId": "a-1",
                "actionResponse": {"value": ["apple", "application"]},
            }),
        ]

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
