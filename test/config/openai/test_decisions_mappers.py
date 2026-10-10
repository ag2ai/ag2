# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum, IntEnum
from typing import Any

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls
from openai.types import Decision

from ag2.compact import CompactionSummary
from ag2.config.openai.decisions_mappers import (
    DecisionRefusedError,
    UnsupportedResponseSchemaError,
    answer_metadata,
    answer_to_content,
    convert_input,
    find_answer,
    normalize_usage,
    response_proto_to_question,
)
from ag2.events import (
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from ag2.exceptions import UnsupportedInputError
from ag2.response import PromptedSchema, ResponseSchema


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    """Payments, invoicing, refunds."""
    TECHNICAL = "technical"
    """Bugs, outages, integrations."""


class Severity(IntEnum):
    COSMETIC = 0
    """Appearance only."""
    WORKAROUND_AVAILABLE = 1
    FULLY_BLOCKED = 2
    """No way to finish the task."""


class Gapped(IntEnum):
    LOW = 1
    HIGH = 2


USAGE = {
    "input_tokens": 30,
    "output_tokens": 0,
    "total_tokens": 30,
    "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _decision(*answers: dict[str, Any]) -> Decision:
    return Decision.model_validate({"model": "gpt-6-luna", "answers": list(answers), "usage": USAGE})


PROBABILITY = ResponseSchema.from_schema(
    {"type": "number", "minimum": 0, "maximum": 1}, name="churn", description="How likely is churn?"
)


class TestQuestion:
    @pytest.mark.parametrize("embed", [True, False], ids=["embedded", "unembedded"])
    @pytest.mark.parametrize(
        "instructions, expected_instructions",
        [
            (None, "Which team should handle this ticket?"),
            ("You triage tickets.", "You triage tickets.\n\nWhich team should handle this ticket?"),
        ],
        ids=["without-prompt", "with-prompt"],
    )
    def test_enum_is_a_choice_described_by_member_docstrings(
        self, embed: bool, instructions: str | None, expected_instructions: str
    ) -> None:
        question = response_proto_to_question(ResponseSchema(Department, embed=embed), instructions=instructions)

        assert question == {
            "type": "choice",
            "name": "answer",
            "instructions": expected_instructions,
            "choices": [
                {"value": "billing", "description": "Payments, invoicing, refunds."},
                {"value": "technical", "description": "Bugs, outages, integrations."},
            ],
        }

    def test_descriptions_outrank_docstrings(self) -> None:
        question = response_proto_to_question(
            ResponseSchema(Department), instructions=None, descriptions={"billing": "Money."}
        )

        assert question == IsPartialDict({
            "choices": [
                {"value": "billing", "description": "Money."},
                {"value": "technical", "description": "Bugs, outages, integrations."},
            ]
        })

    def test_int_enum_is_a_score_labelled_by_member_names(self) -> None:
        question = response_proto_to_question(
            ResponseSchema(Severity, description="How severe is this issue?"), instructions=None
        )

        assert question == {
            "type": "score",
            "name": "answer",
            "instructions": "How severe is this issue?",
            "levels": [
                {"label": "Cosmetic", "description": "Appearance only."},
                {"label": "Workaround available"},
                {"label": "Fully blocked", "description": "No way to finish the task."},
            ],
        }

    def test_score_levels_must_start_at_zero(self) -> None:
        with pytest.raises(UnsupportedResponseSchemaError, match="numbered from 0"):
            response_proto_to_question(ResponseSchema(Gapped, description="How bad?"), instructions=None)

    @pytest.mark.parametrize(
        "schema",
        [ResponseSchema(bool, description="Is this a refund request?"), PROBABILITY],
        ids=["bool", "probability"],
    )
    def test_bool_and_probability_are_predicates(self, schema: ResponseSchema[Any]) -> None:
        question = response_proto_to_question(schema, instructions=None)

        assert question["type"] == "predicate"
        assert question["instructions"] in {"Is this a refund request?", "How likely is churn?"}

    @pytest.mark.parametrize("embed", [True, False], ids=["embedded", "unembedded"])
    def test_bool_docstring_is_not_a_question(self, embed: bool) -> None:
        with pytest.raises(ValueError, match="needs a question"):
            response_proto_to_question(ResponseSchema(bool, embed=embed), instructions=None)

    @pytest.mark.parametrize(
        "schema",
        [None, ResponseSchema(str), ResponseSchema(int), PromptedSchema(Department)],
        ids=["none", "str", "int", "prompted"],
    )
    def test_non_decision_schemas_are_rejected(self, schema: Any) -> None:
        with pytest.raises(UnsupportedResponseSchemaError):
            response_proto_to_question(schema, instructions="Decide.")


class TestInput:
    def test_conversation_is_user_messages(self) -> None:
        serializer = SerializerCls
        messages = [
            CompactionSummary(summary="Earlier, the user asked about pricing.", event_count=2),
            ModelRequest([TextInput("Charged twice."), DataInput({"order": 7})]),
            ModelResponse(message=ModelMessage("Sorry to hear that.")),
            ToolResultsEvent(results=[ToolResultEvent(parent_id="c1", name="lookup", result=ToolResult("paid"))]),
        ]

        assert convert_input(messages, serializer) == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[Summary of earlier conversation]\nEarlier, the user asked about pricing.",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Charged twice."},
                    {"type": "input_text", "text": '{"order":7}'},
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[Assistant]\nSorry to hear that."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[Tool result]"}, {"type": "input_text", "text": "paid"}],
            },
        ]

    def test_images_become_data_urls(self) -> None:
        image = BinaryInput(
            b"\x89PNG", media_type="image/png", kind=BinaryType.IMAGE, vendor_metadata={"detail": "low"}
        )
        inline = UrlInput("data:image/png;base64,AAAA", kind=BinaryType.IMAGE)

        [message] = convert_input([ModelRequest([image, inline])], SerializerCls)

        assert message["content"] == [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw==", "detail": "low"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]

    @pytest.mark.parametrize(
        "part",
        [
            UrlInput("https://example.com/a.png", kind=BinaryType.IMAGE),
            FileIdInput("file_1"),
            BinaryInput(b"%PDF", media_type="application/pdf", kind=BinaryType.DOCUMENT),
        ],
        ids=["hosted-url", "file-id", "document"],
    )
    def test_unsupported_parts_are_rejected(self, part: Any) -> None:
        with pytest.raises(UnsupportedInputError, match="openai-decisions"):
            convert_input([ModelRequest([part])], SerializerCls)


class TestAnswer:
    def test_bool_applies_threshold(self) -> None:
        answer = find_answer(_decision({"type": "predicate", "name": "answer", "probability": 0.7}))
        schema = ResponseSchema(bool, description="Refund?")

        assert json.loads(answer_to_content(schema, answer)) == {"data": True}
        assert json.loads(answer_to_content(schema, answer, boolean_threshold=0.8)) == {"data": False}

    def test_probability_is_raw(self) -> None:
        answer = find_answer(_decision({"type": "predicate", "name": "answer", "probability": 0.25}))

        assert json.loads(answer_to_content(PROBABILITY, answer)) == 0.25

    def test_score_snaps_to_nearest_level(self) -> None:
        answer = find_answer(
            _decision({
                "type": "score",
                "name": "answer",
                "score": 1.6,
                "confidence": 0.5,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround available", "probability": 0.2},
                    {"value": 2, "label": "Fully blocked", "probability": 0.7},
                ],
            })
        )

        assert json.loads(answer_to_content(ResponseSchema(Severity), answer)) == {"data": 2}
        assert answer_metadata(answer)["score"] == 1.6

    def test_refusal_raises(self) -> None:
        with pytest.raises(DecisionRefusedError, match="'answer'"):
            find_answer(_decision({"type": "refusal", "name": "answer"}))

    def test_unnamed_single_answer_is_used(self) -> None:
        answer = find_answer(_decision({"type": "choice", "choice": "billing", "confidence": 1, "probabilities": []}))

        assert answer_metadata(answer) == {"choice": "billing", "confidence": 1.0, "probabilities": []}

    def test_usage(self) -> None:
        assert normalize_usage(_decision().usage) == Usage(
            prompt_tokens=30,
            completion_tokens=0,
            total_tokens=30,
            cache_read_input_tokens=4,
            thinking_tokens=0,
        )
