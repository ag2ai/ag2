# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel, Field

from autogen.beta.config.anthropic.mappers import response_proto_to_output_config
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


class TestResponseProtoToOutputConfigNone:
    def test_none_returns_none(self) -> None:
        assert response_proto_to_output_config(None) is None


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

        result = response_proto_to_output_config(schema)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": IsPartialDict(
                    **_embedded_data_schema(expected_inner_schema),
                    title="ResponseSchema",
                ),
            },
        }


class TestDataclassSchemas:
    def test_simple_dataclass(self) -> None:
        @dataclass
        class User:
            name: str
            age: int

        schema = ResponseSchema(User)

        result = response_proto_to_output_config(schema)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": IsPartialDict(
                    type="object",
                    additionalProperties=False,
                    properties=IsPartialDict(
                        name=IsPartialDict(type="string"),
                        age=IsPartialDict(type="integer"),
                    ),
                ),
            },
        }

    def test_dataclass_with_description(self) -> None:
        @dataclass
        class Response:
            """The structured response."""

            value: int

        schema = ResponseSchema(Response, description="Custom desc")

        result = response_proto_to_output_config(schema)

        assert result is not None
        assert result["format"]["type"] == "json_schema"


class TestPydanticModelSchemas:
    def test_simple_model(self) -> None:
        class Item(BaseModel):
            name: str
            price: float

        schema = ResponseSchema(Item)

        result = response_proto_to_output_config(schema)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": IsPartialDict(
                    type="object",
                    additionalProperties=False,
                    properties=IsPartialDict(
                        name=IsPartialDict(type="string"),
                        price=IsPartialDict(type="number"),
                    ),
                ),
            },
        }

    def test_model_with_field_constraints(self) -> None:
        class Bounded(BaseModel):
            value: Annotated[int, Field(ge=0, le=100)]

        schema = ResponseSchema(Bounded)

        result = response_proto_to_output_config(schema)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": IsPartialDict(
                    properties=IsPartialDict(
                        value=IsPartialDict(minimum=0, maximum=100),
                    ),
                ),
            },
        }


class TestUnionSchemas:
    def test_union_type(self) -> None:
        schema = ResponseSchema(int | str, name="IntOrStr")

        result = response_proto_to_output_config(schema)

        assert result is not None
        fmt_schema = result["format"]["schema"]
        # Union is embedded in {"data": ...} object
        assert fmt_schema["type"] == "object"
        assert fmt_schema["additionalProperties"] is False

    def test_tuple_of_types(self) -> None:
        schema = ResponseSchema((int, float), name="Number")

        result = response_proto_to_output_config(schema)

        assert result is not None
        fmt_schema = result["format"]["schema"]
        assert fmt_schema["type"] == "object"
        assert fmt_schema["additionalProperties"] is False


class TestAdditionalPropertiesFalse:
    """Anthropic requires additionalProperties: false on all object schemas."""

    def test_added_to_top_level_object(self) -> None:
        @dataclass
        class Simple:
            x: int

        schema = ResponseSchema(Simple)
        result = response_proto_to_output_config(schema)

        assert result is not None
        assert result["format"]["schema"]["additionalProperties"] is False

    def test_added_to_nested_objects(self) -> None:
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        schema = ResponseSchema(Outer)
        result = response_proto_to_output_config(schema)

        assert result is not None
        outer_schema = result["format"]["schema"]
        if "$defs" in outer_schema:
            for def_schema in outer_schema["$defs"].values():
                if def_schema.get("type") == "object":
                    assert def_schema["additionalProperties"] is False

    def test_not_added_to_primitives_raw(self) -> None:
        """RawSchema with a primitive type should not get additionalProperties."""
        raw = RawSchema({"type": "string"}, name="Simple")
        result = response_proto_to_output_config(raw)

        assert result is not None
        assert "additionalProperties" not in result["format"]["schema"]

    def test_embedded_primitive_gets_additional_properties(self) -> None:
        """Embedded primitives produce an object wrapper that should get additionalProperties: false."""
        schema = ResponseSchema(int, name="IntSchema")
        result = response_proto_to_output_config(schema)

        assert result is not None
        assert result["format"]["schema"]["type"] == "object"
        assert result["format"]["schema"]["additionalProperties"] is False


class TestDescriptionHandling:
    def test_no_description_not_in_output(self) -> None:
        """Anthropic output_config has no description field — verify schema still works."""

        class Simple(BaseModel):
            x: int

        schema = ResponseSchema(Simple, name="NoDesc")

        result = response_proto_to_output_config(schema)

        assert result is not None
        assert result["format"]["type"] == "json_schema"

    def test_description_not_passed_to_api(self) -> None:
        """Anthropic output_config format does not include name or description."""
        schema = ResponseSchema(int, name="WithDesc", description="An integer value")

        result = response_proto_to_output_config(schema)

        assert result is not None
        # Anthropic format only has type + schema, no name/description
        assert "name" not in result["format"]
        assert "description" not in result["format"]


class TestRawSchema:
    def test_from_schema_maps_correctly(self) -> None:
        raw = RawSchema(
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            name="Custom",
            description="A custom schema",
        )

        result = response_proto_to_output_config(raw)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False},
            },
        }

    def test_from_schema_no_description(self) -> None:
        raw = RawSchema(
            {"type": "string"},
            name="Simple",
        )

        result = response_proto_to_output_config(raw)

        assert result == {
            "format": {
                "type": "json_schema",
                "schema": {"type": "string"},
            },
        }


class TestNoJsonSchema:
    def test_no_schema_returns_none(self) -> None:
        class FakeProto:
            name = "test"
            description = None
            json_schema = None
            system_prompt = None

        result = response_proto_to_output_config(FakeProto())  # type: ignore[arg-type]
        assert result is None
