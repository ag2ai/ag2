# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum, IntEnum
from typing import Any

import httpx2
import pytest
from dirty_equals import IsPartialDict, IsStr
from typesafe_sdk import RetryPolicy, TypeSafeAuthenticationError

from ag2 import Agent, AudioInput, ImageInput, ResponseSchema
from ag2.config import TypeSafeConfig
from ag2.config.typesafe import BoundingBox, TypeSafeClient
from ag2.events import Usage
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.tools import tool


class Department(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


class Severity(IntEnum):
    LOW = 0
    """Cosmetic."""
    HIGH = 1
    """Outage."""


@tool
def lookup(order_id: str) -> str:
    return order_id


def _request_json(request: httpx2.Request) -> dict[str, Any]:
    """The JSON body of a request; images make the SDK send it as the `request` part of a multipart form."""
    content_type = request.headers["content-type"]
    if not content_type.startswith("multipart/form-data"):
        return json.loads(request.content)
    boundary = content_type.split("boundary=")[1].encode()
    [part] = [p for p in request.content.split(b"--" + boundary) if b'name="request"' in p]
    return json.loads(part.split(b"\r\n\r\n", 1)[1])


def _fake_api(
    requests: list[dict[str, Any]],
    answers: dict[str, Any],
    *,
    status: int = 200,
) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(_request_json(request))
        if status != 200:
            return httpx2.Response(status, json={"detail": "Invalid API key"})
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": answers,
                "usage": {"input_tokens": 30, "output_tokens": 0},
            },
        )

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


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
        requests,
        {
            "answer": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.93,
                "probabilities": {"billing": 0.07, "technical": 0.93},
            }
        },
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


@pytest.mark.asyncio
async def test_agent_answers_yes_no_with_bool_schema() -> None:
    requests: list[dict[str, Any]] = []
    guard = Agent(
        "guard",
        config=TypeSafeConfig(
            api_key="test-key", http_client=_fake_api(requests, {"answer": {"type": "noul", "noul": 0.8}})
        ),
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice. I want that money back.")

    assert await reply.content() is True
    assert reply.response.usage == Usage(prompt_tokens=30, completion_tokens=0, total_tokens=30)
    assert requests == [
        IsPartialDict({
            "questions": {
                "answer": IsPartialDict({"type": "noul", "instructions": "Is the customer asking for a refund?"})
            },
        })
    ]


@pytest.mark.asyncio
async def test_agent_scores_with_int_enum_schema() -> None:
    requests: list[dict[str, Any]] = []
    answer = {
        "type": "score",
        "score": 0.7,
        "confidence": 0.6,
        "legend": {"0": "Cosmetic.", "1": "Outage."},
        "probabilities": {"0": 0.3, "1": 0.7},
    }
    grader = Agent(
        "grader",
        prompt="How severe is this incident?",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {"answer": answer})),
        response_schema=Severity,
    )

    reply = await grader.ask("Customers can see each other's invoices.")

    assert await reply.content() is Severity.HIGH
    assert requests == [
        IsPartialDict({
            "questions": {
                "answer": IsPartialDict({
                    "type": "score",
                    "instructions": "How severe is this incident?",
                    "criteria": ["Cosmetic.", "Outage."],
                })
            },
        })
    ]


@pytest.mark.asyncio
async def test_agent_locates_with_bounding_box_schema() -> None:
    requests: list[dict[str, Any]] = []
    answer = {"type": "box", "box": {"top": 0.1, "bottom": 0.4, "left": 0.6, "right": 0.85}, "confidence": 0.9}
    finder = Agent(
        "finder",
        prompt="The red square",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {"answer": answer})),
        response_schema=ResponseSchema(BoundingBox),
    )

    reply = await finder.ask("Where is it?", ImageInput(data=b"\x89PNG", media_type="image/png"))

    assert await reply.content() == BoundingBox(top=0.1, bottom=0.4, left=0.6, right=0.85)
    assert reply.response.metadata["confidence"] == 0.9
    assert requests == [
        IsPartialDict({"questions": {"answer": IsPartialDict({"type": "box", "instructions": "The red square"})}})
    ]


@pytest.mark.asyncio
async def test_agent_sends_the_image_and_clamps_a_box_just_outside_it() -> None:
    requests: list[dict[str, Any]] = []
    answer = {"type": "box", "box": {"top": -0.01, "bottom": 0.4, "left": 0.6, "right": 1.02}, "confidence": None}
    finder = Agent(
        "finder",
        prompt="The red square",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {"answer": answer})),
        response_schema=ResponseSchema(BoundingBox),
    )

    reply = await finder.ask("Where is it?", ImageInput(data=b"\x89PNG", media_type="image/png"))

    assert await reply.content() == BoundingBox(top=0, bottom=0.4, left=0.6, right=1)
    [request] = requests
    assert request == IsPartialDict({
        "state": [{"role": "user", "content": ["Where is it?", IsStr()]}],
        "questions": {"answer": {"type": "box", "instructions": "The red square"}},
    })
    assert [asset["type"] for asset in request["assets"].values()] == ["image"]


@pytest.mark.asyncio
async def test_bounding_box_without_an_image_is_rejected_before_any_request() -> None:
    requests: list[dict[str, Any]] = []
    finder = Agent(
        "finder",
        prompt="The red square",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {})),
        response_schema=ResponseSchema(BoundingBox),
    )

    with pytest.raises(ValueError, match="needs an image"):
        await finder.ask("Where is it?")

    assert requests == []


@pytest.mark.asyncio
async def test_agent_rejects_non_image_media_before_any_request() -> None:
    requests: list[dict[str, Any]] = []
    finder = Agent(
        "finder",
        prompt="The red square",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {})),
        response_schema=ResponseSchema(BoundingBox),
    )

    with pytest.raises(UnsupportedInputError):
        await finder.ask("Where is it?", AudioInput(data=b"abc", media_type="audio/wav"))

    assert requests == []


@pytest.mark.asyncio
async def test_agent_with_tools_is_rejected_before_any_request() -> None:
    requests: list[dict[str, Any]] = []
    router = Agent(
        "router",
        config=TypeSafeConfig(api_key="test-key", http_client=_fake_api(requests, {})),
        response_schema=Department,
        tools=[lookup],
    )

    with pytest.raises(UnsupportedToolError, match="typesafe"):
        await router.ask("Where is order 42?")

    assert requests == []


@pytest.mark.asyncio
async def test_missing_answer_is_an_error() -> None:
    router = Agent(
        "router",
        prompt="Which team should handle this ticket?",
        config=TypeSafeConfig(
            api_key="test-key",
            http_client=_fake_api([], {"other": {"type": "noul", "noul": 0.5}}),
        ),
        response_schema=Department,
    )

    with pytest.raises(ValueError, match="no answer for question 'answer'"):
        await router.ask("My payouts are failing.")


@pytest.mark.asyncio
async def test_api_errors_propagate() -> None:
    requests: list[dict[str, Any]] = []
    router = Agent(
        "router",
        prompt="Which team should handle this ticket?",
        config=TypeSafeConfig(
            api_key="test-key",
            retry=RetryPolicy(max_retries=0),
            http_client=_fake_api(requests, {}, status=401),
        ),
        response_schema=Department,
    )

    with pytest.raises(TypeSafeAuthenticationError):
        await router.ask("My payouts are failing.")

    assert len(requests) == 1
