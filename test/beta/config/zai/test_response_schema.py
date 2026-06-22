# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls
from pydantic import BaseModel

from autogen.beta.config.zai import ZAIClient
from autogen.beta.config.zai.mappers import response_proto_to_format
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.response import PromptedSchema, ResponseSchema
from test.beta.config.zai._helpers import FakeCompletions, FakeZAIClient, make_call_context


class Verdict(BaseModel):
    answer: str
    confidence: float


class Nested(BaseModel):
    verdict: Verdict
    tags: list[str]


def test_none_response_schema_returns_none() -> None:
    assert response_proto_to_format(None) is None
    assert response_proto_to_format(PromptedSchema(Verdict)) is None


def test_json_schema_conversion() -> None:
    result = response_proto_to_format(ResponseSchema(Verdict))

    assert result == IsPartialDict({
        "type": "json_schema",
        "json_schema": IsPartialDict({
            "name": "Verdict",
            "strict": True,
            "schema": IsPartialDict({
                "type": "object",
                "additionalProperties": False,
                "properties": {"answer": IsPartialDict({}), "confidence": IsPartialDict({})},
            }),
        }),
    })


def test_nested_additional_properties() -> None:
    result = response_proto_to_format(ResponseSchema(Nested))

    assert result == IsPartialDict({
        "json_schema": IsPartialDict({
            "schema": IsPartialDict({
                "additionalProperties": False,
                "$defs": IsPartialDict({"Verdict": IsPartialDict({"additionalProperties": False})}),
            })
        })
    })


@pytest.mark.asyncio
async def test_schema_sends_response_format() -> None:
    completions = FakeCompletions()
    client = ZAIClient(create_options={"model": "glm-test"})
    client._client = FakeZAIClient(completions)

    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=ResponseSchema(Verdict),
        serializer=SerializerCls,
    )

    assert completions.kwargs == IsPartialDict({
        "response_format": IsPartialDict({"json_schema": IsPartialDict({"name": "Verdict"})})
    })
