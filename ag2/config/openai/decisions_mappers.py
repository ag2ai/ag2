# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import base64
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
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
from ag2.config.decision_schema import decision_node, explicit_description, option_docs
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
from ag2.response import ResponseProto, ResponseSchema
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
    node, _ = _decision_node(response)
    question = explicit_description(response) or node.get("description")
    instructions = "\n\n".join(s for s in (instructions, question) if s)
    if not instructions:
        # Every question type requires instructions; fail before the request with a way out.
        raise ValueError(
            "A decision needs a question: set the agent prompt or pass `ResponseSchema(..., description=...)`."
        )

    descriptions = descriptions or {}
    values = node.get("enum")
    docs = option_docs(response)

    is_bool = node.get("type") == "boolean" and (values is None or set(values) == {True, False})
    is_probability = node.get("type") == "number" and node.get("minimum") == 0 and node.get("maximum") == 1
    if is_bool or is_probability:
        return QuestionQuestionParamPredicate(type="predicate", name=ANSWER_KEY, instructions=instructions)

    if values and all(isinstance(v, str) for v in values):
        choices: list[QuestionQuestionParamChoiceChoice] = []
        for v in values:
            choice = QuestionQuestionParamChoiceChoice(value=v)
            if description := descriptions.get(v) or docs.get(v):
                choice["description"] = description
            choices.append(choice)
        return QuestionQuestionParamChoice(type="choice", name=ANSWER_KEY, instructions=instructions, choices=choices)

    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        # The API numbers levels by position, so the rubric must already be 0..n-1.
        if sorted(values) != list(range(len(values))) or len(values) < 2:
            raise UnsupportedResponseSchemaError("A score needs 2 or more levels numbered from 0.")
        labels = _level_labels(response)
        levels: list[QuestionQuestionParamScoreLevel] = []
        for v in range(len(values)):
            level = QuestionQuestionParamScoreLevel(label=labels.get(v, str(v)))
            if description := descriptions.get(str(v)) or docs.get(v):
                level["description"] = description
            levels.append(level)
        return QuestionQuestionParamScore(type="score", name=ANSWER_KEY, instructions=instructions, levels=levels)

    raise UnsupportedResponseSchemaError("`response_schema` is not a decision type.")


def _level_labels(response: ResponseProto[Any] | None) -> dict[Any, str]:
    """An ``IntEnum``'s member names as level labels, ``SEVERE_OUTAGE`` read as ``Severe outage``."""
    if isinstance(response, ResponseSchema) and isinstance(response.types, type) and issubclass(response.types, Enum):
        return {m.value: m.name.replace("_", " ").capitalize() for m in response.types}
    return {}


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
    node, embedded = _decision_node(response)

    value: bool | int | float | str
    if isinstance(answer, AnswerAnswerResourcePredicate):
        value = answer.probability >= boolean_threshold if node.get("type") == "boolean" else answer.probability
    elif isinstance(answer, AnswerAnswerResourceChoice):
        value = answer.choice
    elif isinstance(answer, AnswerAnswerResourceRefusal):
        raise DecisionRefusedError(answer.name)
    else:
        # `score` is the probability-weighted mean of the level indices; snap to the nearest level.
        value = min(max(round(answer.score), 0), len(node["enum"]) - 1)

    return json.dumps({"data": value} if embedded else value)


def answer_metadata(answer: Answer) -> dict[str, Any]:
    return answer.model_dump(mode="json", exclude={"type", "name"})


def _decision_node(response: ResponseProto[Any] | None) -> tuple[Mapping[str, Any], bool]:
    if response is None:
        raise UnsupportedResponseSchemaError("A `response_schema` is required.")
    if not (root := response.json_schema):
        raise UnsupportedResponseSchemaError(f"`response_schema` {response.name!r} has no JSON schema.")
    return decision_node(root)


def normalize_usage(usage: DecisionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
        thinking_tokens=usage.output_tokens_details.reasoning_tokens,
    )
