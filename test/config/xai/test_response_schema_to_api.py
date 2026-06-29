# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import pytest
from dirty_equals import IsPartialDict
from xai_sdk.chat import chat_pb2

from ag2.config.xai.mappers import response_proto_to_format
from ag2.response import ResponseSchema


def test_none_returns_none() -> None:
    assert response_proto_to_format(None) is None


@pytest.mark.parametrize(
    ("type_", "expected_inner"),
    [
        pytest.param(int, "integer", id="int"),
        pytest.param(float, "number", id="float"),
        pytest.param(bool, "boolean", id="bool"),
    ],
)
def test_primitive_schemas_wrap_into_data_field(type_: type, expected_inner: str) -> None:
    schema = ResponseSchema(type_, name="Num")

    result = response_proto_to_format(schema)

    assert isinstance(result, chat_pb2.ResponseFormat)
    assert result.format_type == chat_pb2.FORMAT_TYPE_JSON_SCHEMA

    assert json.loads(result.schema) == IsPartialDict({
        "type": "object",
        "properties": IsPartialDict({"data": IsPartialDict({"type": expected_inner})}),
        "required": ["data"],
        "additionalProperties": False,
    })


def test_str_schema_returns_none() -> None:
    """``ResponseSchema(str)`` has no json_schema — model can output free text."""
    schema = ResponseSchema(str, name="Text")

    assert response_proto_to_format(schema) is None


def test_dataclass_schema() -> None:
    @dataclass
    class User:
        name: str
        age: int

    result = response_proto_to_format(ResponseSchema(User))

    assert isinstance(result, chat_pb2.ResponseFormat)
    assert json.loads(result.schema) == IsPartialDict({
        "type": "object",
        "properties": IsPartialDict({
            "name": IsPartialDict({"type": "string"}),
            "age": IsPartialDict({"type": "integer"}),
        }),
        "additionalProperties": False,
    })


def test_additional_properties_false_propagates_to_nested_objects() -> None:
    @dataclass
    class Address:
        city: str

    @dataclass
    class User:
        name: str
        address: Address

    result = response_proto_to_format(ResponseSchema(User))
    assert result is not None

    decoded = json.loads(result.schema)
    assert decoded == IsPartialDict({"additionalProperties": False})
    assert decoded.get("$defs"), "expected $defs for nested dataclass"

    [address_def] = decoded["$defs"].values()
    assert address_def == IsPartialDict({"additionalProperties": False})
