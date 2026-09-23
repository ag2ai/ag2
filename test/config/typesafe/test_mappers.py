# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum, IntEnum
from typing import Any

import pytest
from fast_depends.use import SerializerCls
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from typesafe_sdk import Usage as TypeSafeUsage

from ag2.config.typesafe.mappers import (
    UnsupportedResponseSchemaError,
    answer_metadata,
    answer_to_content,
    convert_state,
    normalize_usage,
    response_proto_to_question,
)
from ag2.events import DataInput, ModelMessage, ModelRequest, ModelResponse, TextInput, Usage
from ag2.response import PromptedSchema, ResponseSchema


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    """Payments, invoicing, refunds."""
    TECHNICAL = "technical"


class Severity(IntEnum):
    LOW = 0
    """Cosmetic."""
    MEDIUM = 1
    """Degraded."""
    HIGH = 2
    """Outage."""


class Undocumented(IntEnum):
    LOW = 0
    HIGH = 1


class OneBased(IntEnum):
    LOW = 1
    """Cosmetic."""
    HIGH = 2
    """Outage."""


def test_bool_maps_to_noul() -> None:
    question = response_proto_to_question(ResponseSchema(bool), instructions="Is this about billing?")

    assert question == Noul(instructions="Is this about billing?")


def test_bool_criteria_override() -> None:
    question = response_proto_to_question(
        ResponseSchema(bool), instructions=None, criteria={"true": "Billing issue", "false": "Anything else"}
    )

    assert question == Noul(criteria={"true": "Billing issue", "false": "Anything else"})


def test_bool_needs_a_question() -> None:
    with pytest.raises(ValueError, match="yes/no question needs asking"):
        response_proto_to_question(ResponseSchema(bool), instructions=None)


def test_explicit_description_is_the_question() -> None:
    schema = ResponseSchema(bool, description="Was 'good' mentioned?")

    assert response_proto_to_question(schema, instructions="You read chat messages.") == Noul(
        instructions="You read chat messages.\n\nWas 'good' mentioned?"
    )


def test_bool_threshold() -> None:
    schema = ResponseSchema(bool)
    answer = NoulAnswer(type="noul", noul=0.6)

    assert json.loads(answer_to_content(schema, answer)) == {"data": True}
    assert json.loads(answer_to_content(schema, answer, boolean_threshold=0.7)) == {"data": False}


def test_unembedded_bool_returns_bare_json() -> None:
    assert answer_to_content(ResponseSchema(bool, embed=False), NoulAnswer(type="noul", noul=0.2)) == "false"


def test_probability_float_maps_to_raw_noul() -> None:
    schema = ResponseSchema.from_schema(
        {"type": "number", "minimum": 0, "maximum": 1}, name="p", description="How likely is churn?"
    )

    assert response_proto_to_question(schema, instructions=None) == Noul(instructions="How likely is churn?")
    assert answer_to_content(schema, NoulAnswer(type="noul", noul=0.25)) == "0.25"


def test_string_literal_maps_to_choice() -> None:
    schema = ResponseSchema.from_schema({"enum": ["calm", "angry"], "type": "string"}, name="tone")

    question = response_proto_to_question(schema, instructions=None, criteria={"angry": "Hostile or upset"})

    assert question == Choice(criteria={"calm": None, "angry": "Hostile or upset"})


@pytest.mark.asyncio
async def test_enum_maps_to_choice_and_round_trips() -> None:
    schema = ResponseSchema(Department)
    answer = ChoiceAnswer(
        type="choice", choice="technical", confidence=0.9, probabilities={"billing": 0.1, "technical": 0.9}
    )

    # The agent prompt frames the question; the Enum docstring is the question.
    assert response_proto_to_question(schema, instructions="You triage support tickets.") == Choice(
        instructions="You triage support tickets.\n\nWhich team should handle this ticket?",
        criteria={"billing": "Payments, invoicing, refunds.", "technical": None},
    )
    assert await schema.validate(answer_to_content(schema, answer), context=None) is Department.TECHNICAL  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_int_enum_maps_to_score_and_snaps_to_nearest_level() -> None:
    schema = ResponseSchema(Severity)
    answer = ScoreAnswer(
        type="score",
        score=1.6,
        confidence=0.7,
        legend={0: "Cosmetic.", 1: "Degraded.", 2: "Outage"},
        probabilities={0: 0.1, 1: 0.2, 2: 0.7},
    )

    assert response_proto_to_question(schema, instructions=None, criteria={"2": "Outage"}) == Score(
        criteria=["Cosmetic.", "Degraded.", "Outage"]
    )
    assert await schema.validate(answer_to_content(schema, answer), context=None) is Severity.HIGH  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(ResponseSchema(Undocumented), id="undocumented"),
        pytest.param(ResponseSchema(OneBased), id="not from zero"),
        pytest.param(
            ResponseSchema.from_schema({"enum": list(range(11)), "type": "integer"}, name="big"), id="11 levels"
        ),
    ],
)
def test_invalid_rubrics_are_rejected(schema: Any) -> None:
    with pytest.raises(UnsupportedResponseSchemaError, match="2-10 levels numbered from 0"):
        response_proto_to_question(schema, instructions=None)


def test_undocumented_rubric_accepts_criteria() -> None:
    question = response_proto_to_question(
        ResponseSchema(Undocumented), instructions=None, criteria={"0": "Cosmetic.", "1": "Outage."}
    )

    assert question == Score(criteria=["Cosmetic.", "Outage."])


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(None, id="no schema"),
        pytest.param(ResponseSchema(str), id="str"),
        pytest.param(ResponseSchema(int), id="int"),
        pytest.param(PromptedSchema(Department), id="prompted"),
        pytest.param(ResponseSchema.from_schema({"enum": [True], "type": "boolean"}, name="yes"), id="Literal[True]"),
    ],
)
def test_non_decision_schemas_are_rejected(schema: Any) -> None:
    with pytest.raises(UnsupportedResponseSchemaError, match="decision-only"):
        response_proto_to_question(schema, instructions=None)


def test_convert_state_serialises_history() -> None:
    state = convert_state(
        [
            ModelRequest([TextInput("I was charged twice.")]),
            ModelResponse(ModelMessage("billing")),
            ModelRequest([DataInput({"order": 42})]),
        ],
        SerializerCls,
    )

    assert state == [
        {"role": "user", "content": "I was charged twice."},
        {"role": "assistant", "content": "billing"},
        {"role": "user", "content": {"order": 42}},
    ]


def test_answer_metadata() -> None:
    answer = ChoiceAnswer(type="choice", choice="calm", confidence=0.8, probabilities={"calm": 0.8, "angry": 0.2})

    assert answer_metadata(answer) == {
        "choice": "calm",
        "confidence": 0.8,
        "probabilities": {"calm": 0.8, "angry": 0.2},
    }


def test_normalize_usage() -> None:
    assert normalize_usage(TypeSafeUsage(input_tokens=12, output_tokens=0)) == Usage(
        prompt_tokens=12, completion_tokens=0, total_tokens=12
    )
    assert normalize_usage(TypeSafeUsage()) == Usage()
