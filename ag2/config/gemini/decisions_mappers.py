# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure conversions between ag2 and the Gemini Decisions API."""

import base64
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any, NoReturn
from urllib.parse import urlparse

from fast_depends.library.serializer import SerializerProto
from google.genai.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestionParam,
    CreateDecisionResponse,
    DecisionInputContentParam,
    NoulAnswer,
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
    FileIdInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from ag2.exceptions import UnsupportedInputError, UnsupportedToolError
from ag2.files.types import FileProvider, UploadedFile
from ag2.response import DecisionSet, DecisionSpec, ResponseProto
from ag2.response.decision import ANSWER_KEY, DecisionValue
from ag2.tools.schemas import ToolSchema

from .decisions_errors import DecisionBlockedError, UnsupportedResponseSchemaError

PROVIDER = "gemini-decisions"

# The API's request limits.
MAX_QUESTIONS = 1000
MAX_OPTIONS = 255
MAX_OPTION_ID_LENGTH = 128
_QUESTION_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")

# `error.code` of the HTTP 400 the API answers when its input filters block a request.
BLOCK_CODES = frozenset(("safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"))

# Text blocks are concatenated with no separator, so adjacent ones get this between them.
_TEXT_SEPARATOR = "\n\n"

_GENERIC_MIME = "application/octet-stream"
_UNSUPPORTED_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")
_FILES_API_BASE = "https://generativelanguage.googleapis.com/v1beta/"

# The MIME type the API takes for each file extension, by block kind.
_MIME_BY_KIND: dict[BinaryType, dict[str, str]] = {
    BinaryType.IMAGE: {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    },
    BinaryType.AUDIO: {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".aif": "audio/aiff",
        ".aiff": "audio/aiff",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
    },
    BinaryType.VIDEO: {
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/mov",
        ".avi": "video/avi",
        ".webm": "video/webm",
        ".wmv": "video/wmv",
        ".flv": "video/x-flv",
        ".3gp": "video/3gpp",
    },
    BinaryType.DOCUMENT: {".pdf": "application/pdf"},
}


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

    return _separate_text(result)


def _separate_text(blocks: list[DecisionInputContentParam]) -> list[DecisionInputContentParam]:
    """Drop empty text, and start each text block that follows another with a separator.

    The API concatenates adjacent text blocks, so without one they would run together.
    """
    result: list[DecisionInputContentParam] = []
    for block in blocks:
        if block["type"] == "text" and not block["text"].strip():
            continue
        if block["type"] == "text" and result and result[-1]["type"] == "text":
            block = _text(_TEXT_SEPARATOR + block["text"])
        result.append(block)
    return result


def _text(text: str) -> TextContentParam:
    return {"type": "text", "text": text}


def _part(part: Input, serializer: SerializerProto) -> DecisionInputContentParam:
    if isinstance(part, TextInput):
        return _text(part.content)

    if isinstance(part, DataInput):
        return _text(serializer.encode(part.data).decode())

    if isinstance(part, BinaryInput):
        if part.media_type == _GENERIC_MIME:
            # The API rejects a generic type; it needs the real file format.
            raise UnsupportedInputError(f"BinaryInput({_GENERIC_MIME})", PROVIDER)
        return _media(part, data=base64.b64encode(part.data).decode(), mime_type=part.media_type)

    if isinstance(part, UrlInput):
        return _url_media(part)

    if isinstance(part, FileIdInput):
        return _file_media(part)

    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def _mime_type(kind: BinaryType | str, path: str) -> str | None:
    """The MIME type the API takes for a ``kind`` block whose file is named ``path``."""
    by_extension = _MIME_BY_KIND.get(kind) if isinstance(kind, BinaryType) else None
    return (by_extension or {}).get(PurePosixPath(path).suffix.lower())


def _file_media(part: FileIdInput) -> DecisionInputContentParam:
    """A media block for a file uploaded with Gemini's Files API, typed by its ``filename``."""
    if isinstance(part, UploadedFile) and part.provider is not None and part.provider is not FileProvider.GEMINI:
        raise UnsupportedInputError(f"file uploaded via '{part.provider.value}'", PROVIDER)
    kind = _kind_of(part.filename or "")
    if kind is None or (mime_type := _mime_type(kind, part.filename or "")) is None:
        raise UnsupportedInputError("FileIdInput without a supported `filename` extension", PROVIDER)
    uri = part.file_id if part.file_id.startswith("https://") else _FILES_API_BASE + part.file_id
    if (block := _block(kind, uri=uri, mime_type=mime_type)) is None:
        raise UnsupportedInputError(f"FileIdInput({kind.value})", PROVIDER)
    return block


def _kind_of(filename: str) -> BinaryType | None:
    """The block kind a file extension names; ``.webm`` is taken as video."""
    suffix = PurePosixPath(filename).suffix.lower()
    for kind in (BinaryType.IMAGE, BinaryType.VIDEO, BinaryType.AUDIO, BinaryType.DOCUMENT):
        if suffix in _MIME_BY_KIND[kind]:
            return kind
    return None


def _url_media(part: UrlInput) -> DecisionInputContentParam:
    """A media block by URI: an HTTPS URL whose file extension gives its MIME type."""
    url = urlparse(part.url)
    if url.scheme != "https" or url.hostname in _UNSUPPORTED_HOSTS:
        # The API takes HTTPS and Files API URIs, not `gs://` or YouTube.
        raise UnsupportedInputError(f"UrlInput({url.scheme}://{url.hostname or ''})", PROVIDER)
    if (mime_type := _mime_type(part.kind, url.path)) is None:
        raise UnsupportedInputError(
            "UrlInput without a file extension the API supports for its kind; pass the bytes instead", PROVIDER
        )
    return _media(part, uri=part.url, mime_type=mime_type)


def _media(part: BinaryInput | UrlInput, **fields: str) -> DecisionInputContentParam:
    """An image, audio, video or document block; the API has no block for other binaries."""
    if (block := _block(part.kind, **fields)) is not None:
        return block
    label = part.kind.value if isinstance(part.kind, BinaryType) else str(part.kind)
    raise UnsupportedInputError(f"{type(part).__name__}({label})", PROVIDER)


def _block(kind: BinaryType | str, **fields: str) -> DecisionInputContentParam | None:
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
    return None


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
    questions: dict[str, QuestionParam] = {}
    for name, spec in decisions.items():
        if not _QUESTION_KEY.fullmatch(name):
            raise UnsupportedResponseSchemaError(
                f"Field {name!r} of `response_schema` cannot be a question key: "
                "keys are 1 to 64 ASCII letters, digits, `_` or `-`."
            )
        questions[name] = _spec_to_question(spec, instructions=instructions, descriptions=descriptions or {})
    return questions


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
        return NoulQuestionParam(type="noul", instructions=text)

    if not 2 <= len(spec.options) <= MAX_OPTIONS:
        raise UnsupportedResponseSchemaError(f"A {spec.kind} needs 2 to {MAX_OPTIONS} options.")

    if spec.kind == "choice":
        options: dict[str, str] = {}
        for option in spec.options:
            option_id = str(option.value)
            _check_option_id(option_id)
            options[option_id] = descriptions.get(option_id) or option.description or ""
        return ChoiceQuestionParam(type="choice", instructions=text, criteria=options)

    # The API numbers levels by position, so the rubric must already be 0..n-1.
    if not spec.is_rubric:
        raise UnsupportedResponseSchemaError("A score needs levels numbered from 0.")
    levels = [_level(option.name, descriptions.get(str(option.value)) or option.description) for option in spec.options]
    return ScoreQuestionParam(type="score", instructions=text, criteria=levels)


def _check_option_id(option_id: str) -> None:
    """Fail on a choice value the API rejects as an option ID."""
    if (
        not option_id
        or len(option_id) > MAX_OPTION_ID_LENGTH
        or option_id != option_id.strip()
        or any(unicodedata.category(c) == "Cc" or (c.isspace() and c != " ") for c in option_id)
    ):
        raise UnsupportedResponseSchemaError(
            f"Choice {option_id!r} cannot be an option ID: IDs are 1 to {MAX_OPTION_ID_LENGTH} characters, "
            "without leading or trailing spaces or control characters."
        )


def _level(name: str, description: str | None) -> str:
    """A level's rubric text: its label from the member name, then its description."""
    label = name.replace("_", " ").capitalize()
    return f"{label}: {description}" if description else label


def blocked_error(error: Exception) -> DecisionBlockedError | None:
    """The ``DecisionBlockedError`` for an SDK error that is the API's HTTP 400 for blocked input, else ``None``."""
    body = getattr(error, "body", None)
    if getattr(error, "status_code", None) != 400 or not isinstance(body, Mapping):
        return None
    detail = body.get("error")
    if not isinstance(detail, Mapping) or detail.get("code") not in BLOCK_CODES:
        return None
    return DecisionBlockedError(str(detail["code"]), detail.get("message"))


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
        if answer.choice not in {str(o.value) for o in spec.options}:
            raise ValueError(
                f"Gemini Decisions answered question {name!r} with {answer.choice!r}, not one of its options."
            )
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
