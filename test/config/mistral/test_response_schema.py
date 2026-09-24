# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from mistralai.client.models import JSONSchema, ResponseFormat
from pydantic import BaseModel

from ag2.config.mistral.mappers import response_proto_to_format
from ag2.response import PromptedSchema, ResponseSchema


class Verdict(BaseModel):
    """A verdict."""

    answer: str
    score: int


def _format(schema: ResponseSchema[Any]) -> ResponseFormat:
    result = response_proto_to_format(schema)
    assert result is not None
    return result


def _json_schema(schema: ResponseSchema[Any]) -> JSONSchema:
    json_schema = _format(schema).json_schema
    assert isinstance(json_schema, JSONSchema)
    return json_schema


def test_no_schema_returns_none() -> None:
    assert response_proto_to_format(None) is None


def test_prompted_schema_has_no_native_format() -> None:
    """``PromptedSchema`` drives the model via the system prompt, not the API."""
    assert response_proto_to_format(PromptedSchema(Verdict)) is None


def test_schema_is_mapped_to_json_schema_format() -> None:
    json_schema = _json_schema(ResponseSchema(Verdict))

    assert _format(ResponseSchema(Verdict)).type == "json_schema"
    assert json_schema.name == "Verdict"
    assert json_schema.strict is True


def test_schema_body_is_bound_to_the_wire_field() -> None:
    """``schema_definition`` is aliased to ``schema`` on the wire."""
    result = _format(ResponseSchema(Verdict))
    json_schema = _json_schema(ResponseSchema(Verdict))

    assert set(json_schema.schema_definition["properties"]) == {"answer", "score"}
    assert result.model_dump(by_alias=True)["json_schema"]["schema"] == json_schema.schema_definition


def test_additional_properties_is_forced_false() -> None:
    """Mistral's strict mode follows the OpenAI convention."""
    assert _json_schema(ResponseSchema(Verdict)).schema_definition["additionalProperties"] is False


def test_nested_objects_get_additional_properties_false() -> None:
    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        inner: Inner

    assert _json_schema(ResponseSchema(Outer)).schema_definition["$defs"]["Inner"]["additionalProperties"] is False
