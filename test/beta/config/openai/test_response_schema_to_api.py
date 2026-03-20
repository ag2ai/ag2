# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel, Field

from autogen.beta.config.openai.mappers import response_proto_to_schema
from autogen.beta.response import ResponseSchema
from autogen.beta.response.schema import RawSchema


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


class TestResponseProtoToSchemaNone:
    def test_none_returns_none(self) -> None:
        assert response_proto_to_schema(None) is None


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

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                schema=IsPartialDict(
                    **_embedded_data_schema(expected_inner_schema),
                    title="ResponseSchema",
                    additionalProperties=False,
                ),
                name=name,
                strict=True,
                # built-in types have docstrings -> description is included
                description=type_.__doc__,
            ),
        }


class TestDataclassSchemas:
    def test_simple_dataclass(self) -> None:
        @dataclass
        class User:
            name: str
            age: int

        schema = ResponseSchema(User)

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                name="User",
                schema=IsPartialDict(
                    type="object",
                    properties=IsPartialDict(
                        name=IsPartialDict(type="string"),
                        age=IsPartialDict(type="integer"),
                    ),
                ),
            ),
        }

    def test_dataclass_with_description(self) -> None:
        @dataclass
        class Response:
            """The structured response."""

            value: int

        schema = ResponseSchema(Response, description="Custom desc")

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(description="Custom desc"),
        }


class TestPydanticModelSchemas:
    def test_simple_model(self) -> None:
        class Item(BaseModel):
            name: str
            price: float

        schema = ResponseSchema(Item)

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                name="Item",
                schema=IsPartialDict(
                    type="object",
                    properties=IsPartialDict(
                        name=IsPartialDict(type="string"),
                        price=IsPartialDict(type="number"),
                    ),
                ),
            ),
        }

    def test_model_with_field_constraints(self) -> None:
        class Bounded(BaseModel):
            value: Annotated[int, Field(ge=0, le=100)]

        schema = ResponseSchema(Bounded)

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                schema=IsPartialDict(
                    properties=IsPartialDict(
                        value=IsPartialDict(minimum=0, maximum=100),
                    ),
                ),
            ),
        }


class TestUnionSchemas:
    def test_union_type(self) -> None:
        schema = ResponseSchema(int | str, name="IntOrStr")

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                name="IntOrStr",
                schema=IsPartialDict(
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
            ),
        }

    def test_tuple_of_types(self) -> None:
        schema = ResponseSchema((int, float), name="Number")

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                name="Number",
                description=(int, float).__doc__,
                schema=IsPartialDict(
                    **_embedded_data_schema(
                        {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                            ],
                        },
                    ),
                    title="ResponseSchema",
                ),
            ),
        }


class TestDescriptionHandling:
    def test_no_description_omitted(self) -> None:
        class Simple(BaseModel):
            x: int

        schema = ResponseSchema(Simple, name="NoDesc")

        result = response_proto_to_schema(schema)

        assert result is not None
        assert "description" not in result["json_schema"]

    def test_description_included(self) -> None:
        schema = ResponseSchema(int, name="WithDesc", description="An integer value")

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(
                name="WithDesc",
                description="An integer value",
            ),
        }

    def test_explicit_name_used(self) -> None:
        class MyModel(BaseModel):
            x: int

        schema = ResponseSchema(MyModel, name="CustomName")

        result = response_proto_to_schema(schema)

        assert result == {
            "type": "json_schema",
            "json_schema": IsPartialDict(name="CustomName"),
        }


class TestRawSchema:
    def test_from_schema_maps_correctly(self) -> None:
        raw = RawSchema(
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            name="Custom",
            description="A custom schema",
        )

        result = response_proto_to_schema(raw)

        assert result == {
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False},
                "name": "Custom",
                "description": "A custom schema",
                "strict": True,
            },
        }

    def test_from_schema_no_description(self) -> None:
        raw = RawSchema(
            {"type": "string"},
            name="Simple",
        )

        result = response_proto_to_schema(raw)

        assert result == {
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "string"},
                "name": "Simple",
                "strict": True,
            },
        }
