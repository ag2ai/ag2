# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from autogen.beta import PromptedSchema, ResponseSchema, response_schema
from autogen.beta.response import ResponseProto

# --- Construction ---


class TestPromptedSchemaFromType:
    def test_from_primitive_type(self) -> None:
        schema = PromptedSchema(int)

        assert schema.name == "ResponseSchema"
        assert schema.json_schema is None
        assert schema.system_prompt is not None
        assert '"type": "integer"' in schema.system_prompt

    def test_from_dataclass(self) -> None:
        @dataclass
        class User:
            name: str
            age: int

        schema = PromptedSchema(User)

        assert schema.name == "User"
        assert '"name"' in schema.system_prompt
        assert '"age"' in schema.system_prompt

    def test_from_pydantic_model(self) -> None:
        class Item(BaseModel):
            title: str
            price: float

        schema = PromptedSchema(Item)

        assert schema.name == "Item"
        assert '"title"' in schema.system_prompt
        assert '"price"' in schema.system_prompt


class TestPromptedSchemaFromResponseProto:
    def test_wraps_response_schema(self) -> None:
        inner = ResponseSchema(int, name="MyInt")
        schema = PromptedSchema(inner)

        assert schema.name == "MyInt"
        assert schema.description is inner.description
        assert '"type": "integer"' in schema.system_prompt

    def test_wraps_schema_with_description(self) -> None:
        inner = ResponseSchema(int, description="An integer value")
        schema = PromptedSchema(inner)

        assert schema.description == "An integer value"

    def test_inner_schema_preserved(self) -> None:
        inner = ResponseSchema(int, name="MyInt")
        original_schema = inner.json_schema

        schema = PromptedSchema(inner)

        assert schema._json_schema == original_schema
        assert schema.json_schema is None


# --- System prompt ---


class TestSystemPrompt:
    def test_contains_json_schema(self) -> None:
        @dataclass
        class Config:
            debug: bool
            level: int

        schema = PromptedSchema(Config)
        prompt = schema.system_prompt

        assert prompt is not None
        # Should contain the full JSON schema
        inner_schema = ResponseSchema(Config).json_schema
        assert json.dumps(inner_schema, indent=2) in prompt

    def test_uses_default_template(self) -> None:
        schema = PromptedSchema(int)
        prompt = schema.system_prompt

        assert "You must respond with valid JSON" in prompt
        assert "Do not include any text" in prompt

    def test_custom_prompt_template(self) -> None:
        template = "Return JSON: ```{schema}```"
        schema = PromptedSchema(int, prompt_template=template)
        prompt = schema.system_prompt

        assert prompt.startswith("Return JSON: ```")

    def test_no_prompt_when_no_inner_schema(self) -> None:
        """If inner schema has no json_schema, system_prompt should be None."""

        class NoSchemaProto(ResponseProto[str]):
            def __init__(self) -> None:
                self.name = "test"
                self.description = None
                self.json_schema = None

            async def validate(self, response, context, provider=None):
                return response

        schema = PromptedSchema(NoSchemaProto())
        assert schema.system_prompt is None


# --- Validation ---


@pytest.mark.asyncio
class TestValidation:
    async def test_validates_primitive(self) -> None:
        schema = PromptedSchema(int)
        context = AsyncMock()

        result = await schema.validate('{"data": 42}', context)
        assert result == 42

    async def test_validates_dataclass(self) -> None:
        @dataclass
        class Point:
            x: float
            y: float

        schema = PromptedSchema(Point)
        context = AsyncMock()

        result = await schema.validate('{"x": 1.5, "y": 2.5}', context)
        assert result == Point(x=1.5, y=2.5)

    async def test_validates_pydantic_model(self) -> None:
        class User(BaseModel):
            name: str
            email: str

        schema = PromptedSchema(User)
        context = AsyncMock()

        result = await schema.validate('{"name": "Alice", "email": "a@b.com"}', context)
        assert isinstance(result, User)
        assert result.name == "Alice"
        assert result.email == "a@b.com"

    async def test_delegates_to_inner_response_schema(self) -> None:
        inner = ResponseSchema(int, name="MyInt")
        schema = PromptedSchema(inner)
        context = AsyncMock()

        result = await schema.validate('{"data": 42}', context)
        assert result == 42

    async def test_delegates_to_inner_callable_schema(self) -> None:
        @response_schema
        def double(content: str) -> int:
            return int(content) * 2

        schema = PromptedSchema(double)
        context = AsyncMock()

        result = await schema.validate("21", context)
        assert result == 42

    async def test_validation_error_on_invalid_json(self) -> None:
        schema = PromptedSchema(int)
        context = AsyncMock()

        with pytest.raises(Exception):
            await schema.validate("not a number", context)

    async def test_callable_schema_validation_error(self) -> None:
        @response_schema
        def parse_int(content: str) -> int:
            return int(content)

        schema = PromptedSchema(parse_int)
        context = AsyncMock()

        with pytest.raises(Exception):
            await schema.validate("not a number", context)
