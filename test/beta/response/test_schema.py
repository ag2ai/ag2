# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Union

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from autogen.beta.response import ResponseSchema
from autogen.beta.response.schema import RawSchema
from autogen.beta.types import ClassInfo


class TestEmbeddedTypes:
    @pytest.mark.parametrize(
        ("type_", "schema"),
        [
            pytest.param(int, {"type": "integer"}, id="int"),
            pytest.param(float, {"type": "number"}, id="float"),
            pytest.param(bool, {"type": "boolean"}, id="bool"),
            pytest.param(
                [int, float],
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="[T1, T2]",
            ),
            pytest.param(
                (int, float),
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="(T1, T2)",
            ),
            pytest.param(
                int | float,
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="typing.Union[T1, T2]",
            ),
            pytest.param(
                Union[int, float],  # noqa: UP007
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="T1 | T2",
            ),
            pytest.param(
                list[int],
                {"type": "array", "items": {"type": "integer"}},
                id="list[T]",
            ),
        ],
    )
    def test_primitive_types_not_embedded(
        self,
        type_: ClassInfo,
        schema: dict[str, Any],
    ) -> None:
        response_schema = ResponseSchema(type_, embed=False)
        assert not response_schema._embedded_type
        assert response_schema.json_schema == schema

    @pytest.mark.parametrize(
        ("type_", "schema"),
        [
            pytest.param(int, {"type": "integer"}, id="int"),
            pytest.param(float, {"type": "number"}, id="float"),
            pytest.param(bool, {"type": "boolean"}, id="bool"),
            pytest.param(
                [int, float],
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="[T1, T2]",
            ),
            pytest.param(
                (int, float),
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="(T1, T2)",
            ),
            pytest.param(
                int | float,
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="typing.Union[T1, T2]",
            ),
            pytest.param(
                Union[int, float],  # noqa: UP007
                {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                id="T1 | T2",
            ),
            pytest.param(
                list[int],
                {"type": "array", "items": {"type": "integer"}},
                id="list[T]",
            ),
        ],
    )
    def test_primitive_types_embedded(
        self,
        type_: ClassInfo,
        schema: dict[str, Any],
    ) -> None:
        response_schema = ResponseSchema(type_)
        assert response_schema._embedded_type
        assert response_schema.json_schema == {
            "properties": {
                "data": {
                    "description": 'Response with a one-field JSON `"{"data":...}"`',
                    "title": "Data",
                    **schema,
                }
            },
            "required": ["data"],
            "type": "object",
        }

    def test_str_has_no_schema(self) -> None:
        response_schema = ResponseSchema(str)
        assert response_schema._embedded_type is True
        assert response_schema.json_schema is None

    def test_dict_not_embedded(self) -> None:
        response_schema = ResponseSchema(dict[str, int])
        assert response_schema._embedded_type is False
        assert response_schema.json_schema == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_dataclass_not_embedded(self) -> None:
        @dataclass
        class Model:
            field: str

        schema = ResponseSchema(Model)
        assert not schema._embedded_type

    def test_pydantic_model_not_embedded(self) -> None:
        class Model(BaseModel):
            field: str

        schema = ResponseSchema(Model)
        assert not schema._embedded_type

    def test_typed_dict_not_embedded(self) -> None:
        class Model(TypedDict):
            field: str

        schema = ResponseSchema(Model)
        assert not schema._embedded_type


class TestDataclassSchemas:
    def test_simple_dataclass(self) -> None:
        @dataclass
        class User:
            name: str
            age: int

        schema = ResponseSchema(User)

        assert schema.name == "User"
        # dataclasses get auto-generated docstrings
        assert schema.description == User.__doc__
        assert schema.json_schema == {
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "age": {"title": "Age", "type": "integer"},
            },
            "required": ["name", "age"],
            "type": "object",
        }

    def test_dataclass_with_docstring(self) -> None:
        @dataclass
        class Response:
            """Structured response from the model."""

            value: int

        schema = ResponseSchema(Response)

        assert schema.name == "Response"
        assert schema.description == "Structured response from the model."

    def test_dataclass_with_defaults(self) -> None:
        @dataclass
        class Config:
            mode: str
            retries: int = 3

        schema = ResponseSchema(Config)

        assert schema.name == "Config"
        assert schema.json_schema == IsPartialDict({
            "required": ["mode"],
            "properties": IsPartialDict({"retries": IsPartialDict({"default": 3})}),
        })

    def test_nested_dataclass(self) -> None:
        @dataclass
        class Address:
            city: str
            zip_code: str

        @dataclass
        class Person:
            name: str
            address: Address

        schema = ResponseSchema(Person)

        assert schema.name == "Person"
        assert schema.json_schema == IsPartialDict({
            "$defs": IsPartialDict({"Address": IsPartialDict({"type": "object"})}),
            "properties": IsPartialDict({"address": {"$ref": "#/$defs/Address"}}),
        })


class TestPydanticModelSchemas:
    def test_simple_model(self) -> None:
        class Item(BaseModel):
            name: str
            price: float

        schema = ResponseSchema(Item)

        assert schema.name == "Item"
        assert schema.json_schema == {
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "price": {"title": "Price", "type": "number"},
            },
            "required": ["name", "price"],
            "type": "object",
        }

    def test_model_with_field_descriptions(self) -> None:
        class Person(BaseModel):
            name: str = Field(description="Full name")
            age: Annotated[int, Field(description="Age in years")]

        schema = ResponseSchema(Person)

        assert schema.json_schema == IsPartialDict({
            "properties": IsPartialDict({
                "name": IsPartialDict({"description": "Full name"}),
                "age": IsPartialDict({"description": "Age in years"}),
            }),
        })

    def test_model_with_docstring(self) -> None:
        class Result(BaseModel):
            """Analysis result."""

            score: float

        schema = ResponseSchema(Result)

        assert schema.description == "Analysis result."

    def test_model_with_nested_model(self) -> None:
        class Coord(BaseModel):
            x: float
            y: float

        class Shape(BaseModel):
            origin: Coord
            label: str

        schema = ResponseSchema(Shape)

        assert schema.name == "Shape"
        assert schema.json_schema == IsPartialDict({
            "$defs": IsPartialDict({"Coord": IsPartialDict({"type": "object"})}),
        })

    def test_model_with_enum_field(self) -> None:
        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        class Palette(BaseModel):
            primary: Color

        schema = ResponseSchema(Palette)

        assert schema.name == "Palette"
        assert schema.json_schema == IsPartialDict({
            "$defs": {
                "Color": {
                    "enum": ["red", "green", "blue"],
                    "title": "Color",
                    "type": "string",
                }
            },
        })

    def test_model_with_annotated_constraints(self) -> None:
        class Bounded(BaseModel):
            value: Annotated[int, Field(ge=0, le=100)]

        schema = ResponseSchema(Bounded)

        assert schema.json_schema == IsPartialDict({
            "properties": IsPartialDict({
                "value": IsPartialDict({"minimum": 0, "maximum": 100}),
            }),
        })


class TestUnionSchemas:
    @pytest.mark.parametrize(
        ("type_", "expected_any_of"),
        [
            pytest.param(
                int | str,
                [{"type": "integer"}, {"type": "string"}],
                id="union_operator",
            ),
            pytest.param(
                (int, str, float),
                [{"type": "integer"}, {"type": "string"}, {"type": "number"}],
                id="tuple_of_types",
            ),
        ],
    )
    def test_union_schema(self, type_: ClassInfo, expected_any_of: list) -> None:  # type: ignore[type-arg]
        schema = ResponseSchema(type_, name="Union", embed=False)

        assert schema.name == "Union"
        assert schema.json_schema == {"anyOf": expected_any_of}

    def test_union_with_dataclass(self) -> None:
        @dataclass
        class Error:
            message: str

        @dataclass
        class Success:
            value: int

        schema = ResponseSchema((Error, Success), name="Result", embed=False)

        assert schema.name == "Result"
        assert schema.json_schema == IsPartialDict({
            "$defs": IsPartialDict({
                "Error": IsPartialDict({}),
                "Success": IsPartialDict({}),
            }),
        })


class TestNameDescription:
    def test_explicit_name_overrides_title(self) -> None:
        class MyModel(BaseModel):
            x: int

        schema = ResponseSchema(MyModel, name="CustomName")

        assert schema.name == "CustomName"

    def test_name_from_class_title(self) -> None:
        class MyModel(BaseModel):
            x: int

        schema = ResponseSchema(MyModel)

        assert schema.name == "MyModel"

    def test_fallback_name(self) -> None:
        schema = ResponseSchema(int)

        # int schema has no "title" key, so falls back to "ResponseSchema"
        assert schema.name == "ResponseSchema"

    def test_explicit_description(self) -> None:
        class MyModel(BaseModel):
            x: int

        schema = ResponseSchema(MyModel, description="Custom description")

        assert schema.description == "Custom description"

    def test_schema_description_overrides_docstring(self) -> None:
        class MyModel(BaseModel):
            """Model docstring."""

            x: int

        # docstring takes priority over explicit description
        schema = ResponseSchema(MyModel, description="Custom description")

        assert schema.description == "Custom description"

    def test_metadata_popped_from_schema(self) -> None:
        class MyModel(BaseModel):
            """Some description."""

            x: int

        schema = ResponseSchema(MyModel)

        assert "title" not in schema.json_schema
        assert "description" not in schema.json_schema


class TestEnsureSchema:
    def test_none_returns_none(self) -> None:
        assert ResponseSchema.ensure_schema(None) is None

    def test_response_proto_returned_as_is(self) -> None:
        original = ResponseSchema(int, name="Test")

        result = ResponseSchema.ensure_schema(original)

        assert result is original

    def test_type_wrapped_in_response_schema(self) -> None:
        result = ResponseSchema.ensure_schema(int)

        assert isinstance(result, ResponseSchema)
        assert result.json_schema == {
            "properties": {
                "data": {
                    "description": 'Response with a one-field JSON `"{"data":...}"`',
                    "title": "Data",
                    "type": "integer",
                }
            },
            "required": ["data"],
            "type": "object",
        }

    def test_union_type_wrapped(self) -> None:
        result = ResponseSchema.ensure_schema(int | str)

        assert isinstance(result, ResponseSchema)
        assert result.json_schema == {
            "properties": {
                "data": {
                    "description": 'Response with a one-field JSON `"{"data":...}"`',
                    "title": "Data",
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string"},
                    ],
                }
            },
            "required": ["data"],
            "type": "object",
        }


class TestRawSchema:
    def test_creation(self) -> None:
        raw = ResponseSchema.from_schema(
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            name="Custom",
            description="A custom schema",
        )

        assert isinstance(raw, RawSchema)
        assert raw.name == "Custom"
        assert raw.description == "A custom schema"
        assert raw.json_schema == {"type": "object", "properties": {"x": {"type": "integer"}}}

    def test_creation_without_description(self) -> None:
        raw = ResponseSchema.from_schema({"type": "string"}, name="Simple")

        assert raw.description is None

    @pytest.mark.asyncio()
    async def test_validate_returns_raw_string_with_warning(self) -> None:
        raw = ResponseSchema.from_schema({"type": "string"}, name="Raw")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await raw.validate("hello", context=None)  # type: ignore[arg-type]

        assert result == "hello"
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "can't validate" in str(w[0].message)


@pytest.mark.asyncio()
class TestValidation:
    @pytest.mark.parametrize(
        ("type_", "json_input", "expected"),
        [
            pytest.param(int, '{"data": 42}', 42, id="int"),
            pytest.param(float, '{"data": 3.14}', 3.14, id="float"),
            pytest.param(bool, '{"data": true}', True, id="bool"),
        ],
    )
    async def test_validate_primitive(self, type_: type, json_input: str, expected: object) -> None:
        schema = ResponseSchema(type_)
        result = await schema.validate(json_input, context=None)  # type: ignore[arg-type]
        assert result == expected

    @pytest.mark.parametrize(
        ("type_", "json_input", "expected"),
        [
            pytest.param(int, "42", 42, id="int"),
            pytest.param(float, "3.14", 3.14, id="float"),
            pytest.param(bool, "true", True, id="bool"),
        ],
    )
    async def test_validate_not_embedded_primitive(self, type_: type, json_input: str, expected: object) -> None:
        schema = ResponseSchema(type_, embed=False)
        result = await schema.validate(json_input, context=None)  # type: ignore[arg-type]
        assert result == expected

    async def test_validate_union(self) -> None:
        schema = ResponseSchema(int | str)

        assert await schema.validate('{"data": 42}', context=None) == 42  # type: ignore[arg-type]
        assert await schema.validate('{"data": "hello"}', context=None) == "hello"  # type: ignore[arg-type]

    async def test_validate_not_embedded_union(self) -> None:
        schema = ResponseSchema(int | str, embed=False)

        assert await schema.validate("42", context=None) == 42  # type: ignore[arg-type]
        assert await schema.validate('"hello"', context=None) == "hello"  # type: ignore[arg-type]

    async def test_validate_dataclass(self) -> None:
        @dataclass
        class Point:
            x: int
            y: int

        schema = ResponseSchema(Point)

        result = await schema.validate('{"x": 1, "y": 2}', context=None)  # type: ignore[arg-type]

        assert result == Point(x=1, y=2)

    async def test_validate_pydantic_model(self) -> None:
        class Item(BaseModel):
            name: str
            price: float

        schema = ResponseSchema(Item)

        result = await schema.validate('{"name": "Widget", "price": 9.99}', context=None)  # type: ignore[arg-type]

        assert result == Item(name="Widget", price=9.99)

    async def test_validate_invalid_json_raises(self) -> None:
        schema = ResponseSchema(int, embed=False)

        with pytest.raises(Exception):
            await schema.validate("not a number", context=None)  # type: ignore[arg-type]
