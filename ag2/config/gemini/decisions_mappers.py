# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure conversions between ag2 and the Gemini Decisions API (`google-genai>=3.2`)."""

import base64
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from fast_depends.library.serializer import SerializerProto
from google.genai.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestionParam,
    CreateDecisionResponse,
    DecisionInputContentParam,
    NoulAnswer,
    NoulCriteriaParam,
    NoulQuestionParam,
    QuestionParam,
    ScoreAnswer,
    ScoreQuestionParam,
)
from google.genai.interactions import (
    AudioContentParam,
    DocumentContentParam,
    ImageContentParam,
    TextContentParam,
    VideoContentParam,
)
from google.genai.interactions import (
    Usage as DecisionUsage,
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
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.response import DecisionSet, DecisionSpec, ResponseProto
from ag2.response.decision import ANSWER_KEY, DecisionValue
from ag2.tools.schemas import ToolSchema

from .decisions_errors import DecisionBlockedError, UnsupportedResponseSchemaError

PROVIDER = "gemini-decisions"
MAX_QUESTIONS = 1000  # The API's limit on questions per request

# `descriptions` keys for the two outcomes of a yes/no question.
_TRUE_KEY = "true"
_FALSE_KEY = "false"


def tool_to_api(t: ToolSchema) -> NoReturn:
    """The Decisions API does not call tools, so every tool is rejected."""
    raise UnsupportedToolError(t.type, PROVIDER)


def convert_input(messages: Iterable[BaseEvent], serializer: SerializerProto) -> list[DecisionInputContentParam]:
    """Serialise the conversation into Decisions ``input``: one flat list of content blocks.

    The API has no roles, so earlier assistant turns and tool results are text blocks
    tagged with where they came from.
    """
    result: list[DecisionInputContentParam] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.extend(_part(p, serializer) for p in message.parts)

        elif isinstance(message, CompactionSummary):
            result.append(_text(f"[Summary of earlier conversation]\n{message.summary}"))

        elif isinstance(message, ModelResponse):
            if message.message and message.message.content:
                result.append(_text(f"[Assistant]\n{message.message.content}"))

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result.append(_text("[Tool result]"))
                result.extend(_part(p, serializer) for p in r.result.parts)

    return result


def _text(text: str) -> TextContentParam:
    return {"type": "text", "text": text}


def _part(part: Input, serializer: SerializerProto) -> DecisionInputContentParam:
    if isinstance(part, TextInput):
        return _text(part.content)

    if isinstance(part, DataInput):
        return _text(serializer.encode(part.data).decode())

    if isinstance(part, BinaryInput):
        return _media(part, data=base64.b64encode(part.data).decode(), mime_type=part.media_type)

    if isinstance(part, UrlInput):
        return _media(part, uri=part.url)

    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def _media(part: BinaryInput | UrlInput, **fields: str) -> DecisionInputContentParam:
    """An image, audio, video or document block; the API has no block for other binaries."""
    kind = part.kind
    if kind == BinaryType.IMAGE:
        image: ImageContentParam = {"type": "image", **fields}  # type: ignore[typeddict-item]
        return image
    if kind == BinaryType.AUDIO:
        audio: AudioContentParam = {"type": "audio", **fields}  # type: ignore[typeddict-item]
        return audio
    if kind == BinaryType.VIDEO:
        video: VideoContentParam = {"type": "video", **fields}  # type: ignore[typeddict-item]
        return video
    if kind == BinaryType.DOCUMENT:
        document: DocumentContentParam = {"type": "document", **fields}  # type: ignore[typeddict-item]
        return document
    label = kind.value if isinstance(kind, BinaryType) else str(kind)
    raise UnsupportedInputError(f"{type(part).__name__}({label})", PROVIDER)


def question_instructions(instructions: str | None, prompt: str) -> str | None:
    """Lead ``question_instructions`` with the agent prompt, which stays the question's text."""
    return "\n\n".join(s for s in (prompt, instructions) if s) or None


def response_proto_to_questions(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | None,
    descriptions: Mapping[str, str] | None = None,
) -> dict[str, QuestionParam]:
    """Convert a ``response_schema`` to the questions of one request, keyed by question name.

    A scalar schema is one question named ``"answer"``; a flat class is one question per field,
    each led by the shared ``instructions``.
    """
    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    if not decisions.is_scalar and descriptions:
        raise ValueError(
            "`GeminiDecisionsConfig(descriptions=...)` is keyed by option value, so it only applies to a "
            "single-question `response_schema`: describe each field's options with "
            "`Annotated[..., Question(options=...)]`."
        )
    if len(decisions) > MAX_QUESTIONS:
        raise UnsupportedResponseSchemaError(
            f"`response_schema` has {len(decisions)} fields; one request takes at most {MAX_QUESTIONS} questions."
        )
    return {
        name: _spec_to_question(spec, instructions=instructions, descriptions=descriptions or {})
        for name, spec in decisions.items()
    }


def _spec_to_question(
    spec: DecisionSpec, *, instructions: str | None, descriptions: Mapping[str, str]
) -> QuestionParam:
    text = "\n\n".join(s for s in (instructions, spec.question) if s)
    if not text:
        # Every question type requires instructions; fail before the request with a way out.
        raise ValueError(
            "A decision needs a question: set the agent prompt, pass `question_instructions`, "
            "or `ResponseSchema(..., description=...)`."
        )

    if spec.kind == "predicate":
        noul = NoulQuestionParam(type="noul", instructions=text)
        criteria = NoulCriteriaParam()
        if true_description := descriptions.get(_TRUE_KEY):
            criteria["true_description"] = true_description
        if false_description := descriptions.get(_FALSE_KEY):
            criteria["false_description"] = false_description
        if criteria:
            noul["criteria"] = criteria
        return noul

    if spec.kind == "choice":
        options = {
            str(option.value): descriptions.get(str(option.value)) or option.description or ""
            for option in spec.options
        }
        return ChoiceQuestionParam(type="choice", instructions=text, criteria=options)

    # The API numbers levels by position, so the rubric must already be 0..n-1.
    if not spec.is_rubric or len(spec.options) < 2:
        raise UnsupportedResponseSchemaError("A score needs 2 or more levels numbered from 0.")
    levels = [_level(option.name, descriptions.get(str(option.value)) or option.description) for option in spec.options]
    return ScoreQuestionParam(type="score", instructions=text, criteria=levels)


def _level(name: str, description: str | None) -> str:
    """A level's rubric text: its label from the member name, then its description."""
    label = name.replace("_", " ").capitalize()
    return f"{label}: {description}" if description else label


def check_blocked(decision: CreateDecisionResponse) -> None:
    """Raise ``DecisionBlockedError`` when the input filters returned no answers."""
    feedback = decision.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise DecisionBlockedError(str(feedback.block_reason), feedback.block_reason_message)


def answers_to_reply(
    response: ResponseProto[Any] | None, decision: CreateDecisionResponse, *, boolean_threshold: float = 0.5
) -> tuple[str, dict[str, Any]]:
    """Render the answers to every question as the schema's JSON, with the metadata for ``ModelMessage``.

    The metadata is the answer itself for a single question, and ``{field: answer}`` for a class.
    """
    check_blocked(decision)
    decisions = DecisionSet.from_response(response, error=UnsupportedResponseSchemaError)
    answers = decision.answers or {}

    values: dict[str, DecisionValue] = {}
    metadata: dict[str, Any] = {}
    for name, spec in decisions.items():
        if (answer := answers.get(name)) is None:
            raise ValueError(f"Gemini Decisions returned no answer for question {name!r}.")
        values[name] = _answer_value(spec, answer, name=name, boolean_threshold=boolean_threshold)
        metadata[name] = answer_metadata(answer)

    if decisions.is_scalar:
        return decisions.render(values), metadata[ANSWER_KEY]
    return decisions.render(values), metadata


def _answer_value(spec: DecisionSpec, answer: Answer, *, name: str, boolean_threshold: float) -> DecisionValue:
    if isinstance(answer, NoulAnswer) and spec.kind == "predicate":
        return spec.predicate_value(answer.noul, threshold=boolean_threshold)
    if isinstance(answer, ChoiceAnswer) and spec.kind == "choice":
        return answer.choice
    if isinstance(answer, ScoreAnswer) and spec.kind == "score":
        # `score` is the probability-weighted mean of the level indices; snap to the nearest level.
        return spec.snap_score(answer.score)
    raise ValueError(f"Gemini Decisions answered question {name!r} ({spec.kind}) with {answer.type!r}.")


def answer_metadata(answer: Answer) -> dict[str, Any]:
    dumped: dict[str, Any] = answer.model_dump(mode="json", exclude={"type"})
    return dumped


def normalize_usage(usage: DecisionUsage | None) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=usage.total_input_tokens,
        completion_tokens=usage.total_output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.total_cached_tokens,
        thinking_tokens=usage.total_thought_tokens,
    )
