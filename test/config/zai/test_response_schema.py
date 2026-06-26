# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls
from pydantic import BaseModel

from ag2.config.zai import ZAIClient
from ag2.config.zai.mappers import response_proto_to_format, schema_instruction
from ag2.events import ModelRequest, TextInput
from ag2.response import PromptedSchema, ResponseSchema
from test.config.zai._helpers import FakeCompletions, FakeZAIClient, make_call_context


class Verdict(BaseModel):
    answer: str
    confidence: float


class Nested(BaseModel):
    verdict: Verdict
    tags: list[str]


def test_none_response_schema_returns_none() -> None:
    assert response_proto_to_format(None) is None
    # PromptedSchema carries no native json_schema — it prompts for JSON itself.
    assert response_proto_to_format(PromptedSchema(Verdict)) is None


def test_native_schema_uses_json_mode() -> None:
    # Z.AI only supports JSON mode; the schema travels in the prompt, not response_format.
    assert response_proto_to_format(ResponseSchema(Verdict)) == {"type": "json_object"}


def test_schema_instruction_describes_schema() -> None:
    instruction = schema_instruction(ResponseSchema(Nested))

    assert instruction is not None
    assert '"additionalProperties": false' in instruction
    assert '"verdict"' in instruction
    assert '"tags"' in instruction


def test_schema_instruction_skips_prompted_schema() -> None:
    # PromptedSchema supplies its own system prompt; don't double up.
    assert schema_instruction(PromptedSchema(Verdict)) is None
    assert schema_instruction(None) is None


@pytest.mark.asyncio
async def test_schema_sends_json_mode_and_prompt() -> None:
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

    assert completions.kwargs == IsPartialDict({"response_format": {"type": "json_object"}})
    system = next(m for m in completions.kwargs["messages"] if m["role"] == "system")
    assert "JSON schema" in system["content"]
    assert '"answer"' in system["content"]
