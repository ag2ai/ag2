# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Read a decision question out of a ``response_schema``, for the decision-only providers.

TypeSafe's Jev and OpenAI's Decisions API both answer one typed question instead of
generating text, and both take the question from the agent's ``response_schema``. The
schema is classified once here; how it becomes a request is up to the provider.
"""

import ast
import inspect
import json
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Annotated, Any, Literal, get_args, get_origin

from ag2.exceptions import AG2Error

from .proto import ResponseProto
from .schema import ResponseSchema, strip_annotated

__all__ = ("DecisionKind", "DecisionOption", "DecisionSpec", "NotADecisionError", "Question")

DecisionKind = Literal["predicate", "choice", "score"]

# Python 3.10 gives an undocumented ``Enum`` this docstring; later versions give ``None``.
_PY310_ENUM_DOC = "An enumeration."


class NotADecisionError(AG2Error):
    """Raised when a ``response_schema`` is not a yes/no, choice or score question."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class Question:
    """Metadata for a decision type, attached with ``Annotated``.

    ``Annotated[bool, Question("Is this a refund request?")]`` sets the question;
    ``Annotated[Dept, Question(options={"billing": "Payments"})]`` describes options by their schema value.
    """

    question: str | None = None
    options: Mapping[str | int, str] | None = None


@dataclass(frozen=True, slots=True)
class DecisionOption:
    """One answer of a choice or one level of a score."""

    value: str | int
    name: str  # ``Enum`` member name, else the value as text
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionSpec:
    kind: DecisionKind
    question: str | None
    options: tuple[DecisionOption, ...]
    boolean: bool  # a predicate that answers ``bool`` rather than a raw probability
    embedded: bool  # ``ResponseSchema`` wrapped the value as ``{"data": ...}``

    @classmethod
    def from_response(cls, response: ResponseProto[Any] | None) -> "DecisionSpec":
        if response is None:
            raise NotADecisionError("A `response_schema` is required.")
        if not (root := response.json_schema):
            raise NotADecisionError(f"`response_schema` {response.name!r} has no JSON schema.")

        node, embedded = _decision_node(root)
        enum_type = _enum_type(response)
        markers = _markers(response)
        question = _question(response, node, markers)
        values = node.get("enum")

        is_bool = node.get("type") == "boolean" and (values is None or set(values) == {True, False})
        is_probability = node.get("type") == "number" and node.get("minimum") == 0 and node.get("maximum") == 1
        if is_bool or is_probability:
            return cls("predicate", question, (), is_bool, embedded)

        if values and all(isinstance(v, str) for v in values):
            return cls("choice", question, _options(values, enum_type, markers), False, embedded)

        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return cls("score", question, _options(values, enum_type, markers), False, embedded)

        raise NotADecisionError("`response_schema` is not a decision type.")

    def predicate_value(self, probability: float, *, threshold: float) -> bool | float:
        """A predicate's probability as the schema's value: thresholded for ``bool``, raw otherwise."""
        return probability >= threshold if self.boolean else probability

    def snap_score(self, score: float) -> int:
        """The nearest valid level of a score, clamped to the rubric."""
        return min(max(round(score), 0), len(self.options) - 1)

    def render(self, value: bool | int | float | str) -> str:
        """The JSON the ``response_schema`` validates."""
        return json.dumps({"data": value} if self.embedded else value)


def _decision_node(root: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """The node of the JSON schema ``root`` to decide on, and whether it was wrapped as ``{"data": ...}``."""
    node: Mapping[str, Any] = root
    properties = root.get("properties")
    embedded = root.get("type") == "object" and isinstance(properties, Mapping) and set(properties) == {"data"}
    if isinstance(properties, Mapping) and embedded:
        # The envelope's description is ag2 boilerplate, not a question for the model.
        node = {k: v for k, v in properties["data"].items() if k != "description"}

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        node = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
    return node, embedded


def _enum_type(response: ResponseProto[Any]) -> type[Enum] | None:
    if isinstance(response, ResponseSchema):
        bare = strip_annotated(response.types)
        if isinstance(bare, type) and issubclass(bare, Enum):
            return bare
    return None


def _markers(response: ResponseProto[Any]) -> list[Question]:
    if not isinstance(response, ResponseSchema) or get_origin(response.types) is not Annotated:
        return []
    return [m for m in get_args(response.types)[1:] if isinstance(m, Question)]


def _question(response: ResponseProto[Any], node: Mapping[str, Any], markers: list[Question]) -> str | None:
    """The closest statement of the question: ``description=``, then ``Question``, then the ``Enum`` docstring."""
    description = response.description
    docstring = None
    if isinstance(response, ResponseSchema):
        if enum_type := _enum_type(response):
            docstring = None if description == _PY310_ENUM_DOC else description
            if description == inspect.cleandoc(enum_type.__doc__ or ""):
                description = None
        elif description == getattr(strip_annotated(response.types), "__doc__", None):
            description = None
    marked = next((m.question for m in reversed(markers) if m.question), None)
    return description or marked or docstring or node.get("description")


def _options(
    values: list[str] | list[int], enum_type: type[Enum] | None, markers: list[Question]
) -> tuple[DecisionOption, ...]:
    names = {m.value: m.name for m in enum_type} if enum_type else {}
    docs = member_docstrings(enum_type) if enum_type else {}
    marked = {k: v for m in markers for k, v in (m.options or {}).items()}
    return tuple(DecisionOption(v, names.get(v, str(v)), marked.get(v) or docs.get(v)) for v in values)


@cache
def member_docstrings(enum_type: type[Enum]) -> dict[Any, str]:
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
