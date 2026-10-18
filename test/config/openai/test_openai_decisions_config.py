# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, ResponseSchema
from ag2.config import OpenAIDecisionsConfig
from ag2.config.openai import DecisionRefusedError, OpenAIDecisionsClient
from ag2.events import Usage
from ag2.exceptions import UnsupportedToolError
from ag2.tools import tool

USAGE = {
    "input_tokens": 30,
    "output_tokens": 0,
    "total_tokens": 30,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


class Department(Enum):
    """Which department should handle this complaint?"""

    BILLING = "billing"
    """Payments, invoices, and refunds."""
    TECHNICAL = "technical"
    """Problems using the product."""


@tool
def lookup(order_id: str) -> str:
    return order_id


@dataclass
class _FakeAPI:
    requests: list[httpx2.Request]
    answers: tuple[dict[str, Any], ...]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json={"model": "gpt-6-luna", "answers": list(self.answers), "usage": USAGE})


def _fake_api(requests: list[httpx2.Request], *answers: dict[str, Any]) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(_FakeAPI(requests, answers)))


def test_defaults() -> None:
    config = OpenAIDecisionsConfig()

    assert (config.model, config.boolean_threshold) == ("gpt-6-luna", 0.5)
    assert isinstance(config.copy(api_key="k").create(), OpenAIDecisionsClient)


def test_copy_overrides_without_mutating_original() -> None:
    config = OpenAIDecisionsConfig(boolean_threshold=0.8)

    copy = config.copy(model="gpt-6-luna-2026-10-01")

    assert (copy.model, copy.boolean_threshold) == ("gpt-6-luna-2026-10-01", 0.8)
    assert config.model == "gpt-6-luna"


def test_no_files_api() -> None:
    with pytest.raises(NotImplementedError):
        OpenAIDecisionsConfig().create_files_client()


@pytest.mark.asyncio
async def test_agent_routes_with_enum_schema() -> None:
    requests: list[httpx2.Request] = []
    http_client = _fake_api(
        requests,
        {
            "type": "choice",
            "name": "answer",
            "choice": "billing",
            "confidence": 0.93,
            "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.05}],
        },
    )
    router = Agent(
        "router",
        prompt="You route customer complaints.",
        config=OpenAIDecisionsConfig(api_key="test-key", http_client=http_client, safety_identifier="user-1"),
        response_schema=Department,
    )

    reply = await router.ask("I was charged twice for my order.")

    assert await reply.content() is Department.BILLING
    assert reply.response.metadata == {
        "choice": "billing",
        "confidence": 0.93,
        "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.05}],
    }
    assert reply.response.usage == Usage(
        prompt_tokens=30, completion_tokens=0, total_tokens=30, cache_read_input_tokens=0, thinking_tokens=0
    )

    [request] = requests
    assert request.url.path == "/v1/decisions"
    assert json.loads(request.content) == IsPartialDict({
        "model": "gpt-6-luna",
        "safety_identifier": "user-1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "I was charged twice for my order."}],
            }
        ],
        "questions": [
            {
                "type": "choice",
                "name": "answer",
                "instructions": "You route customer complaints.\n\nWhich department should handle this complaint?",
                "choices": [
                    {"value": "billing", "description": "Payments, invoices, and refunds."},
                    {"value": "technical", "description": "Problems using the product."},
                ],
            }
        ],
    })


@pytest.mark.asyncio
async def test_agent_answers_yes_no_with_bool_schema() -> None:
    requests: list[httpx2.Request] = []
    guard = Agent(
        "guard",
        config=OpenAIDecisionsConfig(
            api_key="test-key",
            http_client=_fake_api(requests, {"type": "predicate", "name": "answer", "probability": 0.92}),
        ),
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice. I want that money back.")

    assert await reply.content() is True
    assert reply.response.metadata == {"probability": 0.92}
    assert "safety_identifier" not in json.loads(requests[0].content)


@pytest.mark.asyncio
async def test_refusal_raises() -> None:
    guard = Agent(
        "guard",
        config=OpenAIDecisionsConfig(
            api_key="test-key", http_client=_fake_api([], {"type": "refusal", "name": "answer"})
        ),
        response_schema=ResponseSchema(bool, description="Is this harmful?"),
    )

    with pytest.raises(DecisionRefusedError):
        await guard.ask("...")


@pytest.mark.asyncio
async def test_tools_are_rejected_before_the_request() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent(
        "router",
        prompt="Route it.",
        config=OpenAIDecisionsConfig(api_key="test-key", http_client=_fake_api(requests)),
        response_schema=Department,
        tools=[lookup],
    )

    with pytest.raises(UnsupportedToolError, match="openai-decisions"):
        await agent.ask("hi")
    assert requests == []
