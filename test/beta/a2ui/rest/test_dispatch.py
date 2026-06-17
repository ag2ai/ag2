# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import pytest
from typing_extensions import Self

from autogen.beta import Context
from autogen.beta.a2ui import A2UIAgent, A2UIClientCapabilities
from autogen.beta.a2ui.rest import A2UIMessageFrame, A2UIProseFrame, parse_request, stream_turn
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse
from autogen.beta.testing import TestConfig

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


@pytest.mark.asyncio
class TestStreamTurn:
    async def test_plain_text_yields_single_prose_frame(self) -> None:
        agent = A2UIAgent(name="t", config=TestConfig("Hello, no UI."), validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=agent.get_action)

        frames = [f async for f in stream_turn(agent, req)]

        assert frames == [A2UIProseFrame("Hello, no UI.")]

    async def test_a2ui_response_yields_prose_then_message(self) -> None:
        agent = A2UIAgent(name="t", config=TestConfig(_A2UI_RESPONSE), validate_responses=True)
        req = parse_request({"messages": [{"role": "user", "content": "show ui"}]}, resolve_action=agent.get_action)

        frames = [f async for f in stream_turn(agent, req)]

        assert frames[0] == A2UIProseFrame("Here is your UI.")
        assert isinstance(frames[1], A2UIMessageFrame)
        assert frames[1].message["createSurface"]["surfaceId"] == "s1"
        assert len(frames) == 2

    async def test_missing_config_raises(self) -> None:
        agent = A2UIAgent(name="t", validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=agent.get_action)

        with pytest.raises(RuntimeError, match="config is not set"):
            [f async for f in stream_turn(agent, req)]

    async def test_client_capabilities_injected_into_prompt(self) -> None:
        config = _PromptCaptureConfig()
        agent = A2UIAgent(name="t", config=config, validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=agent.get_action)
        req.client_capabilities = A2UIClientCapabilities(supported_catalog_ids=["https://other.example/c.json"])

        [_ async for _ in stream_turn(agent, req)]

        joined = "\n".join(config.prompts[0])
        assert "## A2UI Client Capabilities" in joined
        assert "did NOT list" in joined

    async def test_no_capabilities_no_negotiation_prompt(self) -> None:
        config = _PromptCaptureConfig()
        agent = A2UIAgent(name="t", config=config, validate_responses=False)
        req = parse_request({"messages": [{"role": "user", "content": "hi"}]}, resolve_action=agent.get_action)

        [_ async for _ in stream_turn(agent, req)]

        joined = "\n".join(config.prompts[0])
        assert "## A2UI Client Capabilities" not in joined

    async def test_history_is_stateless_per_turn(self) -> None:
        # A fresh stream per call: prior history sent in the body, not retained.
        agent = A2UIAgent(name="t", config=TestConfig("ok"), validate_responses=False)
        req = parse_request(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "second"},
                ]
            },
            resolve_action=agent.get_action,
        )

        frames = [f async for f in stream_turn(agent, req)]

        assert frames == [A2UIProseFrame("ok")]
