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
    ("type_", "value", "invalid", "schema_type"),
    [
        (Literal["accept", "reject"], "accept", "other", "string"),
        (Literal[1, 2], 1, 3, "integer"),
        (Annotated[Literal["accept", "reject"], "decision"], "reject", "other", "string"),
    ],
)
@pytest.mark.parametrize("embed", [True, False])
async def test_literal_validation(
    type_: ClassInfo, value: str | int, invalid: str | int, schema_type: str, embed: bool
) -> None:
    context = Context(stream=MemoryStream())
    schema = ResponseSchema(type_, embed=embed)
    expected = IsPartialDict({"type": schema_type, "enum": ["accept", "reject"] if schema_type == "string" else [1, 2]})
    assert schema.json_schema == (
        IsPartialDict({"type": "object", "properties": {"data": expected}, "required": ["data"]}) if embed else expected
    )
    assert await schema.validate(json.dumps({"data": value} if embed else value), context) == value
    with pytest.raises(ValidationError):
        await schema.validate(json.dumps({"data": invalid} if embed else invalid), context)


@pytest.mark.asyncio
@pytest.mark.parametrize("type_", [Literal["accept", "reject"]])
@pytest.mark.parametrize("prompted", [True, False])
async def test_agent_literal_response(prompted: bool, type_: ClassInfo) -> None:
    schema = ResponseSchema(type_)
    agent = Agent(
        "decision",
        config=TestConfig('{"data": "accept"}'),
        response_schema=PromptedSchema(schema) if prompted else schema,
    )

    reply = await agent.ask("Classify the request.")

    assert await reply.content() == "accept"
