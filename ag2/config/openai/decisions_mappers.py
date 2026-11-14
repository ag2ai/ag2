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
from openai.types.shared_params import DecisionQuestionInputPart

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
from ag2.response import DecisionSet, DecisionSpec, NotADecisionError, ResponseProto
from ag2.response.decision import ANSWER_KEY, DecisionValue
from ag2.tools.schemas import ToolSchema

from .mappers import _RESPONSES_IMAGE_DETAILS, _image_detail, _kind_label

PROVIDER = "openai-decisions"
MAX_QUESTIONS = 400  # The API's limit on questions per request


class UnsupportedResponseSchemaError(NotADecisionError):
    """Raised when a ``response_schema`` cannot be expressed as a Decisions API question."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"{reason} The OpenAI Decisions API is decision-only: use `bool`, an `Enum` of strings, "
            "an `IntEnum` numbered from 0, a `0..1` number schema from `ResponseSchema.from_schema`, "
            "or a flat class whose required fields are each one of those as `response_schema`."
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


def with_prompt(
    instructions: str | Sequence[str | Input | DecisionQuestionInputPart] | None, prompt: str
) -> str | list[str | Input | DecisionQuestionInputPart] | None:
    """Lead ``question_instructions`` with the agent prompt, which stays the question's text."""
    if not instructions:
        return prompt or None
    if isinstance(instructions, str):
        return "\n\n".join(s for s in (prompt, instructions) if s)
    return [prompt, *instructions] if prompt else list(instructions)


def _question_instructions(
    instructions: str | Sequence[str | Input | DecisionQuestionInputPart] | None,
    question: str | None,
    serializer: SerializerProto,
) -> str | list[DecisionQuestionInputPart]:
    """The ``instructions`` to send: text joined into one string, or the given items as parts with the question appended.

    Strings and ``Input`` events become parts like the conversation's own inputs; SDK parts pass
    through, which is the only way to send ``input_json``. Parts let a question carry images and
    JSON next to its text, which a joined string cannot; the schema's question text then goes out
    as a trailing ``input_text`` part instead of being joined in.
    """
    if instructions is None or isinstance(instructions, str):
        return "\n\n".join(s for s in (instructions, question) if s)

    parts = [_instruction_part(p, serializer) for p in instructions]
    if question:
        parts.append(_text(question))
    return parts


def _instruction_part(
    item: str | Input | DecisionQuestionInputPart, serializer: SerializerProto
) -> DecisionQuestionInputPart:
    if isinstance(item, str):
        return _text(item)
    if isinstance(item, UrlInput) and item.kind == BinaryType.IMAGE:
        # Question instructions, unlike ``input``, take a hosted image URL.
        return {"type": "input_image", "image_url": item.url}
    if isinstance(item, Input):
        return _part(item, serializer)
    return item


def response_proto_to_questions(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | Sequence[str | Input | DecisionQuestionInputPart] | None,
    serializer: SerializerProto,
    descriptions: Mapping[str, str] | None = None,
) -> list[Question]:
    """Convert a ``response_schema`` to the questions of one request.

    A scalar schema is one question named ``"answer"``; a flat class is one question per field,
    each led by the shared ``instructions``.
    """
    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    if not decisions.is_scalar and descriptions:
        raise ValueError(
            "`OpenAIDecisionsConfig(descriptions=...)` is keyed by option value, so it only applies to a "
            "single-question `response_schema`: describe each field's options with "
            "`Annotated[..., Question(options=...)]`."
        )
    if len(decisions) > MAX_QUESTIONS:
        raise UnsupportedResponseSchemaError(
            f"`response_schema` has {len(decisions)} fields; one request takes at most {MAX_QUESTIONS} questions."
        )
    return [
        _spec_to_question(spec, name=name, instructions=instructions, serializer=serializer, descriptions=descriptions)
        for name, spec in decisions.items()
    ]


def _spec_to_question(
    spec: DecisionSpec,
    *,
    name: str,
    instructions: str | Sequence[str | Input | DecisionQuestionInputPart] | None,
    serializer: SerializerProto,
    descriptions: Mapping[str, str] | None,
) -> Question:
    instructions_param = _question_instructions(instructions, spec.question, serializer)
    if not instructions_param:
        # Every question type requires instructions; fail before the request with a way out.
        raise ValueError(
            "A decision needs a question: set the agent prompt, pass `question_instructions`, "
            "or `ResponseSchema(..., description=...)`."
        )

    descriptions = descriptions or {}

    if spec.kind == "predicate":
        return QuestionQuestionParamPredicate(type="predicate", name=name, instructions=instructions_param)

    if spec.kind == "choice":
        choices: list[QuestionQuestionParamChoiceChoice] = []
        for option in spec.options:
            choice = QuestionQuestionParamChoiceChoice(value=str(option.value))
            if description := descriptions.get(str(option.value)) or option.description:
                choice["description"] = description
            choices.append(choice)
        return QuestionQuestionParamChoice(type="choice", name=name, instructions=instructions_param, choices=choices)

    # The API numbers levels by position, so the rubric must already be 0..n-1.
    if not spec.is_rubric or len(spec.options) < 2:
        raise UnsupportedResponseSchemaError("A score needs 2 or more levels numbered from 0.")
    levels: list[QuestionQuestionParamScoreLevel] = []
    for option in spec.options:
        level = QuestionQuestionParamScoreLevel(label=option.name.replace("_", " ").capitalize())
        if description := descriptions.get(str(option.value)) or option.description:
            level["description"] = description
        levels.append(level)
    return QuestionQuestionParamScore(type="score", name=name, instructions=instructions_param, levels=levels)


def find_answer(decision: Decision) -> Answer:
    """The answer to the one question a scalar schema sends, raising if the API declined it."""
    answer = next((a for a in decision.answers if a.name == ANSWER_KEY), None)
    if answer is None:
        if len(decision.answers) != 1:
            raise ValueError(f"OpenAI Decisions returned no answer for question {ANSWER_KEY!r}.")
        answer = decision.answers[0]
    if isinstance(answer, AnswerAnswerResourceRefusal):
        raise DecisionRefusedError(answer.name)
    return answer


def answers_to_reply(
    response: ResponseProto[Any] | None, decision: Decision, *, boolean_threshold: float = 0.5
) -> tuple[str, dict[str, Any]]:
    """Render the answers to every question as the schema's JSON, with the metadata for ``ModelMessage``.

    A refusal of any question fails the whole reply with ``DecisionRefusedError``. The metadata
    is the answer itself for a single question, and ``{field: answer}`` for a class.
    """
    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    if decisions.is_scalar:
        answer = find_answer(decision)
        value = _answer_value(decisions[ANSWER_KEY], answer, boolean_threshold=boolean_threshold)
        return decisions.render({ANSWER_KEY: value}), answer_metadata(answer)

    by_name = {a.name: a for a in decision.answers if a.name in decisions}
    # A declined question has no value of the schema, so the whole reply fails, whatever else is missing.
    if refused := next((a for a in by_name.values() if isinstance(a, AnswerAnswerResourceRefusal)), None):
        raise DecisionRefusedError(refused.name)

    values: dict[str, DecisionValue] = {}
    metadata: dict[str, Any] = {}
    for name, spec in decisions.items():
        if (named := by_name.get(name)) is None:
            raise ValueError(f"OpenAI Decisions returned no answer for question {name!r}.")
        values[name] = _answer_value(spec, named, boolean_threshold=boolean_threshold)
        metadata[name] = answer_metadata(named)
    return decisions.render(values), metadata


def _answer_value(spec: DecisionSpec, answer: Answer, *, boolean_threshold: float) -> DecisionValue:
    if isinstance(answer, AnswerAnswerResourcePredicate):
        return spec.predicate_value(answer.probability, threshold=boolean_threshold)
    if isinstance(answer, AnswerAnswerResourceChoice):
        return answer.choice
    if isinstance(answer, AnswerAnswerResourceRefusal):
        raise DecisionRefusedError(answer.name)
    # `score` is the probability-weighted mean of the level indices; snap to the nearest level.
    return spec.snap_score(answer.score)


def answer_metadata(answer: Answer) -> dict[str, Any]:
    return answer.model_dump(mode="json", exclude={"type", "name"})


def normalize_usage(usage: DecisionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
        cache_creation_input_tokens=usage.input_tokens_details.cache_write_tokens,
        thinking_tokens=usage.output_tokens_details.reasoning_tokens,
    )
