# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import base64
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NoReturn

from fast_depends.library.serializer import SerializerProto
from openai.types import (
    Decision,
    DecisionInputImageParam,
    DecisionInputMessageParam,
    DecisionInputPartUnionParam,
)
from openai.types.decision import (
    Answer,
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
)
from openai.types.decision import Usage as DecisionUsage
from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoice,
    QuestionQuestionParamChoiceChoice,
    QuestionQuestionParamPredicate,
    QuestionQuestionParamScore,
    QuestionQuestionParamScoreLevel,
)

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
    UrlInput,
    Usage,
)
from ag2.exceptions import AG2Error, UnsupportedInputError, UnsupportedToolError
from ag2.response import DecisionSpec, NotADecisionError, ResponseProto
from ag2.tools.schemas import ToolSchema

from .mappers import _RESPONSES_IMAGE_DETAILS, _image_detail, _kind_label

PROVIDER = "openai-decisions"

ANSWER_KEY = "answer"  # Name of the single question sent per request


class UnsupportedResponseSchemaError(AG2Error):
    """Raised when a ``response_schema`` cannot be expressed as a Decisions API question."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"{reason} The OpenAI Decisions API is decision-only: use `bool`, an `Enum` of strings, "
            "an `IntEnum` numbered from 0, or a `0..1` number schema from `ResponseSchema.from_schema` "
            "as `response_schema`."
        )


class DecisionRefusedError(AG2Error):
    """Raised when the Decisions API declines to answer the question."""

    def __init__(self, name: str | None) -> None:
        self.name = name
        super().__init__(f"The OpenAI Decisions API refused to answer question {name!r}.")


def tool_to_api(t: ToolSchema) -> NoReturn:
    """The Decisions API does not call tools, so every tool is rejected."""
    raise UnsupportedToolError(t.type, PROVIDER)


def convert_input(messages: Iterable[BaseEvent], serializer: SerializerProto) -> list[DecisionInputMessageParam]:
    """Serialise the conversation into Decisions ``input``: user messages of text and inline images.

    The API accepts only the ``user`` role, so earlier assistant turns and tool results
    are passed as user messages tagged with where they came from.
    """
    result: list[DecisionInputMessageParam] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append(_user_message(_parts(message.parts, serializer)))

        elif isinstance(message, CompactionSummary):
            result.append(_user_message([_text(f"[Summary of earlier conversation]\n{message.summary}")]))

        elif isinstance(message, ModelResponse):
            if message.message and message.message.content:
                result.append(_user_message([_text(f"[Assistant]\n{message.message.content}")]))

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append(_user_message([_text("[Tool result]"), *_parts(r.result.parts, serializer)]))

    return result


def _user_message(content: list[DecisionInputPartUnionParam]) -> DecisionInputMessageParam:
    return {"type": "message", "role": "user", "content": content}


def _text(text: str) -> DecisionInputPartUnionParam:
    return {"type": "input_text", "text": text}


def _parts(parts: Sequence[Input], serializer: SerializerProto) -> list[DecisionInputPartUnionParam]:
    return [_part(p, serializer) for p in parts]


def _part(part: Input, serializer: SerializerProto) -> DecisionInputPartUnionParam:
    if isinstance(part, TextInput):
        return _text(part.content)

    if isinstance(part, DataInput):
        return _text(serializer.encode(part.data).decode())

    if isinstance(part, BinaryInput):
        if part.kind != BinaryType.IMAGE:
            raise UnsupportedInputError(f"BinaryInput({_kind_label(part.kind)})", PROVIDER)
        b64 = base64.b64encode(part.data).decode()
        image: DecisionInputImageParam = {
            "type": "input_image",
            "image_url": f"data:{part.media_type};base64,{b64}",
        }
        if detail := _image_detail(part, _RESPONSES_IMAGE_DETAILS, PROVIDER):
            image["detail"] = detail
        return image

    if isinstance(part, UrlInput) and part.kind == BinaryType.IMAGE and part.url.startswith("data:"):
        # Hosted image URLs are rejected by the API; an inline data URL is the one form it takes.
        return {"type": "input_image", "image_url": part.url}

    if isinstance(part, UrlInput):
        raise UnsupportedInputError(f"UrlInput({_kind_label(part.kind)})", PROVIDER)

    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def response_proto_to_question(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | None,
    descriptions: Mapping[str, str] | None = None,
) -> Question:
    """Convert a ``response_schema`` to the single Decisions question: predicate, choice or score."""
    spec = _decision_spec(response)
    instructions = "\n\n".join(s for s in (instructions, spec.question) if s)
    if not instructions:
        # Every question type requires instructions; fail before the request with a way out.
        raise ValueError(
            "A decision needs a question: set the agent prompt or pass `ResponseSchema(..., description=...)`."
        )

    descriptions = descriptions or {}

    if spec.kind == "predicate":
        return QuestionQuestionParamPredicate(type="predicate", name=ANSWER_KEY, instructions=instructions)

    if spec.kind == "choice":
        choices: list[QuestionQuestionParamChoiceChoice] = []
        for option in spec.options:
            choice = QuestionQuestionParamChoiceChoice(value=str(option.value))
            if description := descriptions.get(str(option.value)) or option.description:
                choice["description"] = description
            choices.append(choice)
        return QuestionQuestionParamChoice(type="choice", name=ANSWER_KEY, instructions=instructions, choices=choices)

    # The API numbers levels by position, so the rubric must already be 0..n-1.
    if [o.value for o in spec.options] != list(range(len(spec.options))) or len(spec.options) < 2:
        raise UnsupportedResponseSchemaError("A score needs 2 or more levels numbered from 0.")
    levels: list[QuestionQuestionParamScoreLevel] = []
    for option in spec.options:
        level = QuestionQuestionParamScoreLevel(label=option.name.replace("_", " ").capitalize())
        if description := descriptions.get(str(option.value)) or option.description:
            level["description"] = description
        levels.append(level)
    return QuestionQuestionParamScore(type="score", name=ANSWER_KEY, instructions=instructions, levels=levels)


def find_answer(decision: Decision) -> Answer:
    """The answer to the one question sent, raising if the API declined it."""
    answer = next((a for a in decision.answers if a.name == ANSWER_KEY), None)
    if answer is None:
        if len(decision.answers) != 1:
            raise ValueError(f"OpenAI Decisions returned no answer for question {ANSWER_KEY!r}.")
        answer = decision.answers[0]
    if isinstance(answer, AnswerAnswerResourceRefusal):
        raise DecisionRefusedError(answer.name)
    return answer


def answer_to_content(response: ResponseProto[Any] | None, answer: Answer, *, boolean_threshold: float = 0.5) -> str:
    """Render a Decisions answer as the JSON the ``response_schema`` validates."""
    spec = _decision_spec(response)

    value: bool | int | float | str
    if isinstance(answer, AnswerAnswerResourcePredicate):
        value = spec.predicate_value(answer.probability, threshold=boolean_threshold)
    elif isinstance(answer, AnswerAnswerResourceChoice):
        value = answer.choice
    elif isinstance(answer, AnswerAnswerResourceRefusal):
        raise DecisionRefusedError(answer.name)
    else:
        # `score` is the probability-weighted mean of the level indices; snap to the nearest level.
        value = spec.snap_score(answer.score)

    return spec.render(value)


def answer_metadata(answer: Answer) -> dict[str, Any]:
    return answer.model_dump(mode="json", exclude={"type", "name"})


def _decision_spec(response: ResponseProto[Any] | None) -> DecisionSpec:
    try:
        return DecisionSpec.from_response(response)
    except NotADecisionError as e:
        raise UnsupportedResponseSchemaError(e.reason) from e


def normalize_usage(usage: DecisionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
        thinking_tokens=usage.output_tokens_details.reasoning_tokens,
    )
