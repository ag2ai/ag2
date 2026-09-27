# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Bedrock structured output uses the Converse outputConfig (native) mechanism."""

import pytest
from dirty_equals import IsJson, IsPartialDict
from pydantic import BaseModel

from ag2.response import PromptedSchema, ResponseSchema
from test.config.bedrock._helpers import FakeBedrock, ask


class Verdict(BaseModel):
    answer: str
    confidence: float


class Nested(BaseModel):
    verdict: Verdict
    tags: list[str]


@pytest.mark.asyncio
async def test_plain_schema_sends_output_config(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config(), response_schema=ResponseSchema(Verdict))

    assert bedrock.body == {
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "outputConfig": {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "Verdict",
                        # Converse expects the schema serialized as a string
                        "schema": IsJson({
                            "type": "object",
                            "properties": {
                                "answer": {"title": "Answer", "type": "string"},
                                "confidence": {"title": "Confidence", "type": "number"},
                            },
                            "required": ["answer", "confidence"],
                            "additionalProperties": False,
                        }),
                    },
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_nested_schema_closes_every_object(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config(), response_schema=ResponseSchema(Nested))

    assert bedrock.body == IsPartialDict({
        "outputConfig": {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "Nested",
                        "schema": IsJson(
                            IsPartialDict({
                                "additionalProperties": False,
                                "$defs": {"Verdict": IsPartialDict({"additionalProperties": False})},
                            })
                        ),
                    },
                },
            },
        },
    })


@pytest.mark.asyncio
async def test_prompted_schema_goes_to_system_prompt(bedrock: FakeBedrock) -> None:
    schema = PromptedSchema(Verdict)

    await ask(bedrock.config(), response_schema=schema)

    assert bedrock.body == {
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "system": [{"text": schema.system_prompt}],
    }


@pytest.mark.asyncio
async def test_no_schema_sends_neither(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config())

    assert bedrock.body == {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}
