# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from ag2.config.mistral.mappers import response_proto_to_format
from ag2.response import PromptedSchema, ResponseSchema


class Verdict(BaseModel):
    """A verdict."""

    answer: str
    score: int


def test_no_schema_returns_none() -> None:
    assert response_proto_to_format(None) is None


def test_prompted_schema_has_no_native_format() -> None:
    """``PromptedSchema`` drives the model via the system prompt, not the API."""
    assert response_proto_to_format(PromptedSchema(Verdict)) is None


def test_schema_is_mapped_to_json_schema_format() -> None:
    result = response_proto_to_format(ResponseSchema(Verdict))

    assert result.type == "json_schema"
    assert result.json_schema.name == "Verdict"
    assert result.json_schema.strict is True


def test_schema_body_is_bound_to_the_wire_field() -> None:
    """``schema_definition`` is aliased to ``schema`` on the wire."""
    result = response_proto_to_format(ResponseSchema(Verdict))

    assert set(result.json_schema.schema_definition["properties"]) == {"answer", "score"}
    assert result.model_dump(by_alias=True)["json_schema"]["schema"] == result.json_schema.schema_definition


def test_additional_properties_is_forced_false() -> None:
    """Mistral's strict mode follows the OpenAI convention."""
    result = response_proto_to_format(ResponseSchema(Verdict))

    assert result.json_schema.schema_definition["additionalProperties"] is False


def test_nested_objects_get_additional_properties_false() -> None:
    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        inner: Inner

    result = response_proto_to_format(ResponseSchema(Outer))

    assert result.json_schema.schema_definition["$defs"]["Inner"]["additionalProperties"] is False
