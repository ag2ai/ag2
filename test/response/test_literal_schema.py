# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated, Literal

import pytest
from dirty_equals import IsPartialDict
from pydantic import ValidationError

from ag2 import Agent, Context, MemoryStream
from ag2.response import PromptedSchema, ResponseSchema
from ag2.testing import TestConfig
from ag2.types import ClassInfo


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("type_", "value", "invalid", "expected"),
    [
        (Literal["accept", "reject"], "accept", "other", {"type": "string", "enum": ["accept", "reject"]}),
        (Literal[1, 2], 1, 3, {"type": "integer", "enum": [1, 2]}),
        (
            Annotated[Literal["accept", "reject"], "decision"],
            "reject",
            "other",
            {"type": "string", "enum": ["accept", "reject"]},
        ),
    ],
)
async def test_literal_validation_embedded(
    type_: ClassInfo, value: str | int, invalid: str | int, expected: dict[str, object]
) -> None:
    context = Context(stream=MemoryStream())
    schema = ResponseSchema(type_)

    assert schema.json_schema == IsPartialDict({
        "type": "object",
        "properties": {"data": IsPartialDict(expected)},
        "required": ["data"],
    })
    assert await schema.validate(json.dumps({"data": value}), context) == value
    with pytest.raises(ValidationError):
        await schema.validate(json.dumps({"data": invalid}), context)


@pytest.mark.asyncio
async def test_literal_validation_unembedded() -> None:
    # Local validation only: the root schema is a bare enum, which OpenAI rejects
    # for structured output (its root must be an object), so keep the default ``embed=True``.
    context = Context(stream=MemoryStream())
    schema = ResponseSchema(Literal["accept", "reject"], embed=False)

    assert schema.json_schema == IsPartialDict({"type": "string", "enum": ["accept", "reject"]})
    assert await schema.validate(json.dumps("accept"), context) == "accept"
    with pytest.raises(ValidationError):
        await schema.validate(json.dumps("other"), context)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompted", [True, False])
async def test_agent_literal_response(prompted: bool) -> None:
    schema = ResponseSchema(Literal["accept", "reject"])
    agent = Agent(
        "decision",
        config=TestConfig('{"data": "accept"}'),
        response_schema=PromptedSchema(schema) if prompted else schema,
    )

    reply = await agent.ask("Classify the request.")

    assert await reply.content() == "accept"
