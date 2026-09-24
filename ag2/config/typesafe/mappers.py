# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import json
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import cache
from typing import Any, NoReturn

from fast_depends.library.serializer import SerializerProto
from typesafe_sdk import Answer, Choice, ChoiceAnswer, JSONValue, Noul, NoulAnswer, Question, Score
from typesafe_sdk import Usage as TypeSafeUsage

from ag2.compact import CompactionSummary
from ag2.events import (
    BaseEvent,
    DataInput,
    Input,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
    Usage,
)
from ag2.exceptions import AG2Error, UnsupportedInputError, UnsupportedToolError
from ag2.response import ResponseProto, ResponseSchema
from ag2.tools.schemas import ToolSchema

PROVIDER = "typesafe"

ANSWER_KEY = "answer"  # Name of the single question sent per request


class UnsupportedResponseSchemaError(AG2Error):
    """Raised when a ``response_schema`` cannot be expressed as a TypeSafe question."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"{reason} TypeSafe Jev is decision-only: use `bool`, an `Enum` of strings, a documented `IntEnum`, "
            "or a `0..1` number schema from `ResponseSchema.from_schema` as `response_schema`."
        )


def tool_to_api(t: ToolSchema) -> NoReturn:
    """Jev does not call tools, so every tool is rejected."""
    raise UnsupportedToolError(t.type, PROVIDER)


def convert_state(messages: Iterable[BaseEvent], serializer: SerializerProto) -> list[dict[str, JSONValue | None]]:
    """Serialise the conversation into Jev ``state``: a JSON array of ``{role, content}``."""
    state: list[dict[str, JSONValue | None]] = []

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


def _parts_content(parts: Sequence[Input], serializer: SerializerProto) -> JSONValue | None:
    values = [_part_value(p, serializer) for p in parts]
    return values[0] if len(values) == 1 else values


def _part_value(part: Input, serializer: SerializerProto) -> JSONValue | None:
    if isinstance(part, TextInput):
        return part.content
    if isinstance(part, DataInput):
        data: JSONValue | None = json.loads(serializer.encode(part.data))
        return data
    raise UnsupportedInputError(type(part).__name__, PROVIDER)


def response_proto_to_question(
    response: ResponseProto[Any] | None,
    *,
    instructions: str | None,
    criteria: Mapping[str, str] | None = None,
) -> Question:
    """Convert a ``response_schema`` to the single Jev question: noul, choice or score."""
    node, _ = _decision_node(response)
    question = _explicit_description(response) or node.get("description")
    instructions = "\n\n".join(s for s in (instructions, question) if s) or None
    criteria = criteria or {}
    values = node.get("enum")
    docs = _option_docs(response)

    is_bool = node.get("type") == "boolean" and (values is None or set(values) == {True, False})
    is_probability = node.get("type") == "number" and node.get("minimum") == 0 and node.get("maximum") == 1
    if is_bool or is_probability:
        noul_criteria = {k: criteria[k] for k in ("true", "false") if k in criteria}
        if not instructions and not noul_criteria:
            # The API rejects a bare yes/no question; fail before the request with a way out.
            raise ValueError(
                "A yes/no question needs asking: set the agent prompt, pass "
                "`ResponseSchema(bool, description=...)`, or describe the outcomes with "
                "`TypeSafeConfig(criteria={'true': ..., 'false': ...})`."
            )
        return Noul(instructions=instructions, criteria=noul_criteria or None)
    if values and all(isinstance(v, str) for v in values):
        return Choice(instructions=instructions, criteria={v: criteria.get(v) or docs.get(v) for v in values})
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        # A rubric is levels 0..n-1 in numeric order, each saying what it means.
        levels = [criteria.get(str(v)) or docs.get(v) for v in range(len(values))]
        if sorted(values) != list(range(len(values))) or not 2 <= len(values) <= 10 or not all(levels):
            raise UnsupportedResponseSchemaError(
                "A score needs 2-10 levels numbered from 0, each described by a docstring under "
                "the member or by `criteria`."
            )
        return Score(instructions=instructions, criteria=levels)

    raise UnsupportedResponseSchemaError("`response_schema` is not a decision type.")


def answer_to_content(response: ResponseProto[Any] | None, answer: Answer, *, boolean_threshold: float = 0.5) -> str:
    """Render a Jev answer as the JSON the ``response_schema`` validates."""
    node, embedded = _decision_node(response)

    value: bool | int | float | str
    if isinstance(answer, NoulAnswer):
        value = answer.noul >= boolean_threshold if node.get("type") == "boolean" else answer.noul
    elif isinstance(answer, ChoiceAnswer):
        value = answer.choice
    else:
        # `score` is the expected value on the 0..n-1 rubric; snap to the nearest level.
        value = min(max(round(answer.score), 0), len(node["enum"]) - 1)

    return json.dumps({"data": value} if embedded else value)


def _explicit_description(response: ResponseProto[Any] | None) -> str | None:
    """A ``description=`` the user gave, ignoring ``ResponseSchema``'s fallback to the type's docstring."""
    if response is None or (
        isinstance(response, ResponseSchema) and response.description == getattr(response.types, "__doc__", None)
    ):
        return None
    return response.description


def _option_docs(response: ResponseProto[Any] | None) -> dict[Any, str]:
    if isinstance(response, ResponseSchema) and isinstance(response.types, type) and issubclass(response.types, Enum):
        return _member_docstrings(response.types)
    return {}


@cache
def _member_docstrings(enum_type: type[Enum]) -> dict[Any, str]:
    """Map each member's value to the string literal under it, read from the class source."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(enum_type)))
    except (OSError, TypeError, SyntaxError):
        return {}

    body = tree.body[0].body if tree.body and isinstance(tree.body[0], ast.ClassDef) else []
    docs = {
        stmt.targets[0].id: inspect.cleandoc(doc.value.value)
        for stmt, doc in zip(body, body[1:])
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(doc, ast.Expr)
        and isinstance(doc.value, ast.Constant)
        and isinstance(doc.value.value, str)
    }
    return {member.value: docs[member.name] for member in enum_type if member.name in docs}


def _decision_node(response: ResponseProto[Any] | None) -> tuple[Mapping[str, Any], bool]:
    """Return the schema node to decide on, and whether ``ResponseSchema`` wrapped it as ``{"data": ...}``."""
    if response is None:
        raise UnsupportedResponseSchemaError("A `response_schema` is required.")
    if not (root := response.json_schema):
        raise UnsupportedResponseSchemaError(f"`response_schema` {response.name!r} has no JSON schema.")

    node: Mapping[str, Any] = root
    properties = root.get("properties")
    embedded = root.get("type") == "object" and isinstance(properties, Mapping) and set(properties) == {"data"}
    if embedded:
        # The envelope's description is ag2 boilerplate, not a question for Jev.
        node = {k: v for k, v in properties["data"].items() if k != "description"}  # type: ignore[index]

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        node = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
    return node, embedded


def normalize_usage(raw: TypeSafeUsage) -> Usage:
    """Normalise TypeSafe's ``Usage`` (either count may be unreported) to AG2 ``Usage``."""
    prompt = float(raw.input_tokens) if raw.input_tokens is not None else None
    completion = float(raw.output_tokens) if raw.output_tokens is not None else None
    total = (prompt or 0) + (completion or 0) if prompt is not None or completion is not None else None

    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def answer_metadata(answer: Answer) -> dict[str, Any]:
    return answer.model_dump(exclude={"type"})
