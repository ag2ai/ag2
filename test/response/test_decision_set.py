# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated

import pytest
from pydantic import AliasPath, BaseModel, Field

from ag2 import Context, MemoryStream, ResponseSchema
from ag2.response import DecisionSet, NotADecisionError, Question


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    TECHNICAL = "technical"


class Urgency(IntEnum):
    LOW = 0
    HIGH = 1


@dataclass
class TicketTriage:
    is_spam: Annotated[bool, Question("Is this message spam?")]
    department: Department
    urgency: Annotated[Urgency, Question("How urgent is this ticket?")]


class AliasedTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="isSpam")
    department: Department = Field(alias="team")


class SharedPathTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(validation_alias=AliasPath("shared"))
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")] = Field(validation_alias=AliasPath("shared"))


class MixedPathTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(validation_alias=AliasPath("shared"))
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")] = Field(alias="shared")


class PathToAnotherField(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(validation_alias=AliasPath("is_urgent"))
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")]


class DistinctPathTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(validation_alias=AliasPath("spam"))
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")] = Field(validation_alias=AliasPath("urgent"))


@dataclass
class Documented:
    """A ticket's triage."""

    is_spam: Annotated[bool, Question("Is this message spam?")]


@dataclass
class Blank:
    is_spam: Annotated[bool, Question("   ")]


def test_scalar_schema_is_one_question_named_answer() -> None:
    decisions = DecisionSet.from_response(ResponseSchema(Department))

    assert decisions.is_scalar
    assert list(decisions) == ["answer"]
    assert decisions.render({"answer": "billing"}) == '{"data": "billing"}'


def test_object_schema_is_one_question_per_field() -> None:
    decisions = DecisionSet.from_response(ResponseSchema(TicketTriage))

    assert not decisions.is_scalar
    assert {name: spec.kind for name, spec in decisions.items()} == {
        "is_spam": "predicate",
        "department": "choice",
        "urgency": "score",
    }
    assert json.loads(decisions.render({"urgency": 1, "is_spam": False, "department": "billing"})) == {
        "is_spam": False,
        "department": "billing",
        "urgency": 1,
    }


def test_aliased_fields_are_named_by_field_and_rendered_by_alias() -> None:
    decisions = DecisionSet.from_response(ResponseSchema(AliasedTriage))

    assert {name: spec.question for name, spec in decisions.items()} == {
        "is_spam": "Is this message spam?",
        "department": "Which team should handle this ticket?",
    }
    assert json.loads(decisions.render({"is_spam": True, "department": "billing"})) == {
        "isSpam": True,
        "team": "billing",
    }


@pytest.mark.parametrize(
    "schema",
    [SharedPathTriage, MixedPathTriage, PathToAnotherField],
    ids=["shared-path", "path-and-alias", "path-to-field-name"],
)
def test_fields_sharing_an_input_key_are_rejected(schema: type[BaseModel]) -> None:
    with pytest.raises(NotADecisionError, match="Fields 'is_spam' and 'is_urgent' .+ both validated from key"):
        DecisionSet.from_response(ResponseSchema(schema))


@pytest.mark.asyncio
async def test_distinct_flat_paths_keep_each_answer() -> None:
    schema = ResponseSchema(DistinctPathTriage)
    decisions = DecisionSet.from_response(schema)

    content = decisions.render({"is_spam": True, "is_urgent": False})

    assert json.loads(content) == {"spam": True, "urgent": False}
    assert await schema.validate(content, Context(stream=MemoryStream())) == DistinctPathTriage.model_validate({
        "spam": True,
        "urgent": False,
    })


def test_blank_question_is_no_question() -> None:
    with pytest.raises(ValueError, match="Field 'is_spam' of `response_schema` has no question"):
        DecisionSet.from_response(ResponseSchema(Blank))


@pytest.mark.parametrize(
    "schema",
    [
        ResponseSchema(TicketTriage, description="Triage this ticket."),
        ResponseSchema(Annotated[TicketTriage, Question("Triage this ticket.")]),
    ],
    ids=["description", "question"],
)
def test_class_takes_no_question_of_its_own(schema: ResponseSchema[TicketTriage]) -> None:
    with pytest.raises(ValueError, match="takes no question of its own"):
        DecisionSet.from_response(schema)


def test_class_docstring_is_not_a_question_of_its_own() -> None:
    assert list(DecisionSet.from_response(ResponseSchema(Documented))) == ["is_spam"]
