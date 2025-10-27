# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from dirty_equals import IsPartialDict
from httpx import Response
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from starlette.testclient import TestClient

from autogen import ConversableAgent
from autogen.a2a import A2aRemoteAgent
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_pydantic_a2a() -> None:
    # arrange
    pydantic_a2a_agent = Agent(FakeModel()).to_a2a()
    test_client = CustomTestClient(pydantic_a2a_agent)

    remote_agent_mirror = A2aRemoteAgent(
        url="http://localhost:8000",
        name="custom",
        client=lambda: test_client,
    )

    local_agent = ConversableAgent(name="local")

    # act
    with test_client, TestAgent(local_agent, messages=["Hi, I am mock client!"]):
        result = await local_agent.a_initiate_chat(
            remote_agent_mirror,
            message="Hi, agent!",
            max_turns=2,
        )

    # assert
    assert result.chat_history == [
        IsPartialDict({"content": "Hi, agent!", "name": "local"}),
        IsPartialDict({"content": "Hi, I am pydantic agent!", "name": "custom"}),
        IsPartialDict({"content": "Hi, I am mock client!", "name": "local"}),
        IsPartialDict({"content": "Hi, I am pydantic agent!", "name": "custom"}),
    ]


class CustomTestClient(TestClient):
    async def __aenter__(self) -> "CustomTestClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def get(self, *args: Any, **kwargs: Any) -> Response:
        return super().get(*args, **kwargs)

    async def post(self, *args: Any, **kwargs: Any) -> Response:
        return super().post(*args, **kwargs)


class FakeModel(Model):
    def __init__(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "custom"

    @property
    def system(self) -> str:
        return "openai"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Hi, I am pydantic agent!")])
