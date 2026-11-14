# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from ag2 import ResponseSchema
from ag2.response import DecisionSet, Question


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
