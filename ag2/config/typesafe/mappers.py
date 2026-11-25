# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NoReturn

from fast_depends.library.serializer import SerializerProto
from pydantic import BaseModel, Field
from typesafe_sdk import (
    Answer,
    Box,
    BoxAnswer,
    Choice,
    ChoiceAnswer,
    ContentValue,
    Image,
    JSONValue,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
)
from typesafe_sdk import Usage as TypeSafeUsage

from ag2.compact import CompactionSummary
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import DecisionSet, DecisionSpec, NotADecisionError, ResponseProto, ResponseSchema
from ag2.response.decision import ANSWER_KEY, DecisionValue
from ag2.response.schema import strip_annotated
from ag2.tools.schemas import ToolSchema

PROVIDER = "typesafe"


class BoundingBox(BaseModel):
    """Where something is in an image: each edge as a fraction of its height (`top`, `bottom`) or width (`left`, `right`).

    Pass `ResponseSchema(BoundingBox)` to ask TypeSafe to locate what the agent prompt describes in the image of the
    request. The box's confidence is on `reply.response.metadata`.
    """

    top: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)
    left: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)


class UnsupportedResponseSchemaError(NotADecisionError):
    """Raised when a ``response_schema`` cannot be expressed as a TypeSafe question."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"{reason} TypeSafe Jev is decision-only: use `bool`, an `Enum` of strings, a documented `IntEnum`, "
            "a `0..1` number schema from `ResponseSchema.from_schema`, `BoundingBox`, or a flat class whose "
            "required fields are each one of the decision types as `response_schema`."
        )


def tool_to_api(t: ToolSchema) -> NoReturn:
    """Jev does not call tools, so every tool is rejected."""
    raise UnsupportedToolError(t.type, PROVIDER)


def convert_state(messages: Iterable[BaseEvent], serializer: SerializerProto) -> list[dict[str, ContentValue | None]]:
    """Serialise the conversation into Jev ``state``: a JSON array of ``{role, content}``."""
    state: list[dict[str, ContentValue | None]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            state.append({"role": "user", "content": _parts_content(message.parts, serializer)})

        elif isinstance(message, CompactionSummary):
            state.append({"role": "user", "content": f"[Summary of earlier conversation]\n{message.summary}"})

        elif isinstance(message, ModelResponse):
            if message.message and message.message.content:
                state.append({"role": "assistant", "content": message.message.content})

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                state.append({"role": "tool", "content": _parts_content(r.result.parts, serializer)})

    return state


def _parts_content(parts: Sequence[Input], serializer: SerializerProto) -> ContentValue | None:
    values = [_part_value(p, serializer) for p in parts]
    return values[0] if len(values) == 1 else values


def _part_value(part: Input, serializer: SerializerProto) -> ContentValue | None:
    if isinstance(part, TextInput):
        return part.content
    if isinstance(part, DataInput):
        data: JSONValue | None = json.loads(serializer.encode(part.data))
        return data
    if isinstance(part, BinaryInput) and part.kind is BinaryType.IMAGE:
        return Image(part.data)
    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def response_proto_to_questions(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | None,
    criteria: Mapping[str, str] | None = None,
) -> dict[str, Question]:
    """Convert a ``response_schema`` to the named Jev questions of one request.

    A scalar schema or ``BoundingBox`` is one question named ``"answer"``; a flat class is one
    question per field, each led by the shared ``instructions``.
    """
    if _is_box(response):
        return {ANSWER_KEY: response_proto_to_question(response, instructions=instructions, criteria=criteria)}

    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    if not decisions.is_scalar and criteria:
        raise ValueError(
            "`TypeSafeConfig(criteria=...)` is keyed by option value, so it only applies to a single-question "
            "`response_schema`: describe each field's options with `Annotated[..., Question(options=...)]`."
        )
    return {
        name: _spec_to_question(spec, instructions=instructions, criteria=criteria) for name, spec in decisions.items()
    }


def response_proto_to_question(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | None,
    criteria: Mapping[str, str] | None = None,
) -> Question:
    """Convert a scalar ``response_schema`` to the single Jev question: noul, choice, score or box."""
    if _is_box(response):
        # The prompt says what to find; a description is only the caller's own (a docstring is not a question).
        explicit = response.explicit_description if isinstance(response, ResponseSchema) else None
        return Box(instructions="\n\n".join(s for s in (instructions, explicit) if s) or None)

    spec = DecisionSpec.from_response(response, error=UnsupportedResponseSchemaError)
    return _spec_to_question(spec, instructions=instructions, criteria=criteria)


def _spec_to_question(spec: DecisionSpec, *, instructions: str | None, criteria: Mapping[str, str] | None) -> Question:
    instructions = "\n\n".join(s for s in (instructions, spec.question) if s) or None
    criteria = criteria or {}

    if spec.kind == "predicate":
        noul_criteria = NoulCriteria()
        if "true" in criteria:
            noul_criteria["true"] = criteria["true"]
        if "false" in criteria:
            noul_criteria["false"] = criteria["false"]
        if not instructions and not noul_criteria:
            # The API rejects a bare yes/no question; fail before the request with a way out.
            raise ValueError(
                "A yes/no question needs asking: set the agent prompt, pass "
                "`ResponseSchema(bool, description=...)`, or describe the outcomes with "
                "`TypeSafeConfig(criteria={'true': ..., 'false': ...})`."
            )
        if instructions:
            return Noul(instructions=instructions, criteria=noul_criteria or None)
        if (described_true := criteria.get("true")) is not None:
            return Noul(criteria={"true": described_true, "false": criteria.get("false")})
        return Noul(criteria={"true": None, "false": criteria["false"]})

    if spec.kind == "choice":
        return Choice(
            instructions=instructions,
            criteria={str(o.value): criteria.get(str(o.value)) or o.description for o in spec.options},
        )

    # A rubric is levels 0..n-1 in numeric order, each saying what it means.
    levels = [criteria.get(str(o.value)) or o.description for o in spec.options]
    if not spec.is_rubric or not 2 <= len(levels) <= 10 or not all(levels):
        raise UnsupportedResponseSchemaError(
            "A score needs 2-10 levels numbered from 0, each described by a docstring under "
            "the member (unreadable when the source is unavailable, e.g. in a REPL or for a functional `Enum`), "
            "by `Annotated[..., Question(options=...)]` or by `criteria`."
        )
    return Score(instructions=instructions, criteria=levels)


def require_image_for_box(question: Question, state: Iterable[Mapping[str, ContentValue | None]]) -> None:
    """Refuse a box question with nothing to look at: the API would answer with an invented box."""
    if not isinstance(question, Box):
        return
    for message in state:
        content = message["content"]
        if any(isinstance(part, Image) for part in (content if isinstance(content, list) else [content])):
            return
    raise ValueError("A BoundingBox needs an image in the request: send one with `ImageInput`.")


def _is_box(response: ResponseProto[Any] | None) -> bool:
    return isinstance(response, ResponseSchema) and strip_annotated(response.types) is BoundingBox


def answer_to_content(response: ResponseProto[Any] | None, answer: Answer, *, boolean_threshold: float = 0.5) -> str:
    """Render a Jev answer as the JSON the ``response_schema`` validates."""
    if _is_box(response):
        if not isinstance(answer, BoxAnswer):  # pragma: no cover - the API answers the question it was asked
            raise UnsupportedResponseSchemaError(f"TypeSafe answered a box question with a {answer.type!r} answer.")
        # The model's estimate may land just outside the image; the box is clamped rather than the answer lost.
        edges = {edge: min(max(value, 0.0), 1.0) for edge, value in answer.box.model_dump().items()}
        return BoundingBox(**edges).model_dump_json()

    spec = DecisionSpec.from_response(response, error=UnsupportedResponseSchemaError)
    return spec.render(_answer_value(spec, answer, boolean_threshold=boolean_threshold))


def answers_to_reply(
    response: ResponseProto[Any] | None, answers: Mapping[str, Answer], *, boolean_threshold: float = 0.5
) -> tuple[str, dict[str, Any]]:
    """Render the answers to every question as the schema's JSON, with the metadata for ``ModelMessage``.

    The metadata is the answer itself for a single question, and ``{field: answer}`` for a class.
    """
    if _is_box(response):
        answer = _named_answer(answers, ANSWER_KEY)
        return answer_to_content(response, answer), answer_metadata(answer)

    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    values: dict[str, DecisionValue] = {}
    metadata: dict[str, Any] = {}
    for name, spec in decisions.items():
        answer = _named_answer(answers, name)
        values[name] = _answer_value(spec, answer, boolean_threshold=boolean_threshold)
        metadata[name] = answer_metadata(answer)

    return decisions.render(values), metadata[ANSWER_KEY] if decisions.is_scalar else metadata


def _named_answer(answers: Mapping[str, Answer], name: str) -> Answer:
    answer = answers.get(name)
    if answer is None:
        raise ValueError(f"TypeSafe returned no answer for question {name!r}.")
    return answer


def _answer_value(spec: DecisionSpec, answer: Answer, *, boolean_threshold: float) -> DecisionValue:
    if isinstance(answer, NoulAnswer):
        return spec.predicate_value(answer.noul, threshold=boolean_threshold)
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, ScoreAnswer):
        # `score` is the expected value on the 0..n-1 rubric; snap to the nearest level.
        return spec.snap_score(answer.score)
    # A box answer only answers a box question.
    raise UnsupportedResponseSchemaError(  # pragma: no cover
        f"TypeSafe answered a decision with a {answer.type!r} answer."
    )


def normalize_usage(raw: TypeSafeUsage) -> Usage:
    """Normalise TypeSafe's ``Usage`` (either count may be unreported) to AG2 ``Usage``."""
    prompt = float(raw.input_tokens) if raw.input_tokens is not None else None
    completion = float(raw.output_tokens) if raw.output_tokens is not None else None
    total = (prompt or 0) + (completion or 0) if prompt is not None or completion is not None else None

    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def answer_metadata(answer: Answer) -> dict[str, Any]:
    return answer.model_dump(exclude={"type"})
