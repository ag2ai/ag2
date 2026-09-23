# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.config import ModelProvider, TypeSafeConfig
from ag2.config.typesafe import TypeSafeClient


class Department(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


def _fake_api(answer: dict[str, Any], requests: list[dict[str, Any]]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {"answer": answer},
                "usage": {"input_tokens": 30, "output_tokens": 0},
            },
        )

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def test_provider() -> None:
    assert TypeSafeConfig().provider is ModelProvider.TYPESAFE


def test_defaults() -> None:
    config = TypeSafeConfig()

    assert (config.model, config.api_key, config.boolean_threshold) == ("jev-latest", None, 0.5)


def test_copy_overrides_without_mutating_original() -> None:
    config = TypeSafeConfig(boolean_threshold=0.8)

    copy = config.copy(model="jev-2026-09-15")

    assert (copy.model, copy.boolean_threshold) == ("jev-2026-09-15", 0.8)
    assert config.model == "jev-latest"


def test_create_does_not_need_api_key() -> None:
    """The SDK validates the key on construction, so the client must defer building it."""
    assert isinstance(TypeSafeConfig().create(), TypeSafeClient)


def test_no_files_api() -> None:
    with pytest.raises(NotImplementedError):
        TypeSafeConfig().create_files_client()


@pytest.mark.asyncio
async def test_agent_routes_with_enum_schema() -> None:
    requests: list[dict[str, Any]] = []
    http_client = _fake_api(
        {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.93,
            "probabilities": {"billing": 0.07, "technical": 0.93},
        },
        requests,
    )
    router = Agent(
        "router",
        prompt="Which team should handle this ticket?",
        config=TypeSafeConfig(api_key="test-key", http_client=http_client),
        response_schema=Department,
    )

    reply = await router.ask("Help! My webhook integration has been down for 3 days.")

    assert await reply.content() is Department.TECHNICAL
    assert reply.response.metadata == {
        "choice": "technical",
        "confidence": 0.93,
        "probabilities": {"billing": 0.07, "technical": 0.93},
    }
    assert requests == [
        IsPartialDict({
            "model": "jev-latest",
            "state": [{"role": "user", "content": "Help! My webhook integration has been down for 3 days."}],
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {"billing": None, "technical": None},
                }
            },
        })
    ]
