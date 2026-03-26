# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel, Field

from autogen.beta.config.gemini.mappers import response_proto_to_config
from autogen.beta.response import ResponseSchema


def _embedded_data_schema(inner: dict) -> dict:  # type: ignore[type-arg]
    """JSON schema for a primitive/union wrapped in ``{\"data\": ...}`` (default ``embed=True``)."""
    return {
        "properties": {
            "data": {
                "description": 'Response with a one-field JSON `"{"data":...}"`',
                "title": "Data",
                **inner,
            }
        },
        "required": ["data"],
        "type": "object",
    }


class TestResponseProtoToConfigNone:
    def test_none_returns_empty(self) -> None:
        assert response_proto_to_config(None) == {}


class TestPrimitiveSchemas:
    @pytest.mark.parametrize(
        ("type_", "name", "expected_inner_schema"),
        [
            pytest.param(int, "IntSchema", {"type": "integer"}, id="int"),
            pytest.param(float, "FloatSchema", {"type": "number"}, id="float"),
            pytest.param(bool, "BoolSchema", {"type": "boolean"}, id="bool"),
        ],
    )
    def test_primitive_type(
        self,
        type_: type,
        name: str,
        expected_inner_schema: dict,  # type: ignore[type-arg]
    ) -> None:
        schema = ResponseSchema(type_, name=name)

        result = response_proto_to_config(schema)

        assert result == {
            "response_mime_type": "application/json",
            "response_json_schema": IsPartialDict(
                **_embedded_data_schema(expected_inner_schema),
                title="ResponseSchema",
            ),
        }


class TestDataclassSchemas:
    def test_simple_dataclass(self) -> None:
        @dataclass
        class User:
            name: str
            age: int

        schema = ResponseSchema(User)

        result = response_proto_to_config(schema)

        assert result == {
            "response_mime_type": "application/json",
            "response_json_schema": IsPartialDict(
                type="object",
                properties=IsPartialDict(
                    name=IsPartialDict(type="string"),
                    age=IsPartialDict(type="integer"),
                ),
            ),
        }


class TestPydanticModelSchemas:
    def test_simple_model(self) -> None:
        class Item(BaseModel):
            name: str
            price: float

        schema = ResponseSchema(Item)

        result = response_proto_to_config(schema)

        assert result == {
            "response_mime_type": "application/json",
            "response_json_schema": IsPartialDict(
                type="object",
                properties=IsPartialDict(
                    name=IsPartialDict(type="string"),
                    price=IsPartialDict(type="number"),
                ),
            ),
        }

    def test_model_with_field_constraints(self) -> None:
        class Bounded(BaseModel):
            value: Annotated[int, Field(ge=0, le=100)]

        schema = ResponseSchema(Bounded)

        result = response_proto_to_config(schema)

        assert result == {
            "response_mime_type": "application/json",
            "response_json_schema": IsPartialDict(
                properties=IsPartialDict(
                    value=IsPartialDict(minimum=0, maximum=100),
                ),
            ),
        }


class TestUnionSchemas:
    def test_union_type(self) -> None:
        schema = ResponseSchema(int | str, name="IntOrStr")

        result = response_proto_to_config(schema)

        assert result == {
            "response_mime_type": "application/json",
            "response_json_schema": IsPartialDict(
                **_embedded_data_schema(
                    {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                    },
                ),
                title="ResponseSchema",
            ),
        }


class TestNoJsonSchema:
    def test_no_schema_returns_empty(self) -> None:
        """ResponseProto with json_schema=None returns empty dict."""

        class FakeProto:
            name = "test"
            description = None
            json_schema = None
            system_prompt = None

        result = response_proto_to_config(FakeProto())  # type: ignore[arg-type]
        assert result == {}
