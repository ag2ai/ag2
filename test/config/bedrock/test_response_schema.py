# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Bedrock structured output uses the Converse outputConfig (native) mechanism."""

import json

import pytest
from fast_depends.use import SerializerCls
from pydantic import BaseModel

from ag2.config.bedrock import BedrockClient
from ag2.config.bedrock.mappers import response_proto_to_output_config
from ag2.events import ModelRequest, TextInput
from ag2.response import PromptedSchema, ResponseSchema
from test.config.bedrock._helpers import FakeBedrockRuntime, StubSession, make_call_context


class Verdict(BaseModel):
    answer: str
    confidence: float


class Nested(BaseModel):
    verdict: Verdict
    tags: list[str]


async def _ask(fake: FakeBedrockRuntime, response_schema) -> None:
    client = BedrockClient(session=StubSession(fake), create_options={"model": "m1"})
    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=response_schema,
        serializer=SerializerCls,
    )


def test_output_config_shape() -> None:
    config = response_proto_to_output_config(ResponseSchema(Verdict))

    text_format = config["textFormat"]
    assert text_format["type"] == "json_schema"
    json_schema = text_format["structure"]["jsonSchema"]
    assert json_schema["name"] == "Verdict"
    # Converse expects the schema serialized as a string
    schema = json.loads(json_schema["schema"])
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"answer", "confidence"}


def test_output_config_nested_additional_properties() -> None:
    config = response_proto_to_output_config(ResponseSchema(Nested))

    schema = json.loads(config["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Verdict"]["additionalProperties"] is False


def test_output_config_none_without_json_schema() -> None:
    assert response_proto_to_output_config(None) is None
    assert response_proto_to_output_config(PromptedSchema(Verdict)) is None


@pytest.mark.asyncio
async def test_plain_schema_sends_output_config() -> None:
    fake = FakeBedrockRuntime()

    await _ask(fake, ResponseSchema(Verdict))

    output_config = fake.converse_kwargs["outputConfig"]
    assert output_config["textFormat"]["structure"]["jsonSchema"]["name"] == "Verdict"
    assert "system" not in fake.converse_kwargs


@pytest.mark.asyncio
async def test_prompted_schema_goes_to_system_prompt() -> None:
    fake = FakeBedrockRuntime()
    schema = PromptedSchema(Verdict)

    await _ask(fake, schema)

    [system] = fake.converse_kwargs["system"]
    assert system["text"] == schema.system_prompt
    assert "outputConfig" not in fake.converse_kwargs


@pytest.mark.asyncio
async def test_no_schema_sends_neither() -> None:
    fake = FakeBedrockRuntime()

    await _ask(fake, None)

    assert "outputConfig" not in fake.converse_kwargs
    assert "system" not in fake.converse_kwargs
