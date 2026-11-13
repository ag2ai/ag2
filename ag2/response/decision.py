# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Read decision questions out of a ``response_schema``, for the decision-only providers.

TypeSafe's Jev and OpenAI's Decisions API both answer typed questions instead of
generating text, and both take the questions from the agent's ``response_schema``: a
scalar is one question, a flat class is one question per field, all sent in one request.
The schema is classified once here; how it becomes a request is up to the provider.
"""

import ast
import inspect
import json
import textwrap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Annotated, Any, Literal, NotRequired, Required, get_args, get_origin, get_type_hints

from pydantic import AliasChoices
from pydantic.fields import FieldInfo
from typing_extensions import ReadOnly

from ag2.exceptions import AG2Error

from .proto import ResponseProto
from .schema import ResponseSchema, strip_annotated

__all__ = (
    "ANSWER_KEY",
    "DecisionKind",
    "DecisionOption",
    "DecisionSet",
    "DecisionSpec",
    "DecisionValue",
    "NotADecisionError",
    "Question",
)

DecisionKind = Literal["predicate", "choice", "score"]
DecisionValue = bool | int | float | str

ANSWER_KEY = "answer"  # Name of the question a scalar ``response_schema`` is sent under


class NotADecisionError(AG2Error):
    """Raised when a ``response_schema`` is not a yes/no, choice or score question."""


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
    """A ``response_schema`` read as one question; ``from_response`` builds it, providers translate it."""

    kind: DecisionKind
    question: str | None
    options: tuple[DecisionOption, ...]
    boolean: bool  # a predicate that answers ``bool`` rather than a raw probability
    embedded: bool  # ``ResponseSchema`` wrapped the value as ``{"data": ...}``

    @classmethod
    def from_response(
        cls, response: ResponseProto[Any] | None, *, error: type[NotADecisionError] = NotADecisionError
    ) -> "DecisionSpec":
        """Read ``response``, raising ``error`` (a provider's own subclass, say) if it is not a decision."""
        if response is None:
            raise error("A `response_schema` is required.")
        if not (root := response.json_schema):
            raise error(f"`response_schema` {response.name!r} is not a decision type: it has no JSON schema.")

        node, embedded = _decision_node(root)
        enum_type = _enum_type(response)
        markers = _markers(response)
        question = _question(response, node, enum_type, markers)
        return cls._classify(node, question, enum_type, markers, embedded=embedded, error=error)

    @classmethod
    def _classify(
        cls,
        node: Mapping[str, Any],
        question: str | None,
        enum_type: type[Enum] | None,
        markers: list[Question],
        *,
        embedded: bool,
        error: type[NotADecisionError],
        subject: str = "`response_schema`",
    ) -> "DecisionSpec":
        values = node.get("enum")
        _check_option_keys(markers, values or [])

        is_bool = node.get("type") == "boolean" and (values is None or set(values) == {True, False})
        is_probability = node.get("type") == "number" and node.get("minimum") == 0 and node.get("maximum") == 1
        if is_bool or is_probability:
            return cls("predicate", question, (), is_bool, embedded)

        if values and all(isinstance(v, str) for v in values):
            return cls("choice", question, _options(values, enum_type, markers), False, embedded)

        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            # The score APIs number levels by position, so order them by value, not by declaration.
            return cls("score", question, _options(sorted(values), enum_type, markers), False, embedded)

        raise error(f"{subject} is not a decision type.")

    @property
    def is_rubric(self) -> bool:
        """Whether the options are the levels ``0..n-1`` in order, as the score APIs number them."""
        return [o.value for o in self.options] == list(range(len(self.options)))

    def predicate_value(self, probability: float, *, threshold: float) -> bool | float:
        """For a predicate: its probability as the schema's value, thresholded for ``bool``, raw otherwise."""
        return probability >= threshold if self.boolean else probability

    def snap_score(self, score: float) -> int:
        """For a score: the nearest valid level, clamped to the rubric."""
        return min(max(round(score), 0), len(self.options) - 1)

    def render(self, value: DecisionValue) -> str:
        """The JSON the ``response_schema`` validates."""
        return json.dumps({"data": value} if self.embedded else value)


class DecisionSet(Mapping[str, DecisionSpec]):
    """A ``response_schema`` read as the named questions of one request.

    A scalar schema (``bool``, a string ``Enum``, an ``IntEnum``, a ``0..1`` number) is one
    question named ``"answer"``. A flat class (a dataclass, a ``TypedDict``, a Pydantic model)
    is one question per field, named after the field and asked by its ``Question(...)``,
    ``Field(description=...)`` or ``Enum`` docstring.
    """

    __slots__ = ("_keys", "_scalar", "_specs")

    def __init__(
        self, specs: Mapping[str, DecisionSpec], *, scalar: bool, keys: Mapping[str, str] | None = None
    ) -> None:
        self._specs = dict(specs)
        self._scalar = scalar
        # The JSON key of each field where it differs from the field name: a Pydantic alias.
        self._keys = dict(keys or {})

    @classmethod
    def from_response(
        cls, response: ResponseProto[Any] | None, *, error: type[NotADecisionError] = NotADecisionError
    ) -> "DecisionSet":
        """Read ``response``, raising ``error`` (a provider's own subclass, say) if it is not a decision."""
        root = response.json_schema if response is not None else None
        if response is None or not root or not _is_object_schema(response, root):
            return cls({ANSWER_KEY: DecisionSpec.from_response(response, error=error)}, scalar=True)
        if _class_question(response):
            # Each field is its own question; a question on the class would have nowhere to go.
            raise ValueError(
                "A class `response_schema` is one question per field, so the class takes no question of its "
                "own: put what the fields share in the agent prompt and each field's in `Question(...)`."
            )
        specs, keys = _field_specs(response, root, error)
        return cls(specs, scalar=False, keys=keys)

    @property
    def is_scalar(self) -> bool:
        """Whether the schema is one value asked as ``"answer"``, rather than a class of fields."""
        return self._scalar

    def render(self, values: Mapping[str, DecisionValue]) -> str:
        """The JSON the ``response_schema`` validates, from each question's value."""
        if self._scalar:
            return self._specs[ANSWER_KEY].render(values[ANSWER_KEY])
        return json.dumps({self._keys.get(name, name): values[name] for name in self._specs})

    def __getitem__(self, name: str) -> DecisionSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"DecisionSet({self._specs!r}, scalar={self._scalar!r}, keys={self._keys!r})"


def _is_object_schema(response: ResponseProto[Any], root: Mapping[str, Any]) -> bool:
    """Whether ``root`` is a class of fields, as opposed to a scalar or ``ResponseSchema``'s ``{"data": ...}`` envelope."""
    if root.get("type") != "object" or not isinstance(root.get("properties"), Mapping):
        return False
    if isinstance(response, ResponseSchema):
        # A class with a single ``data`` field looks like the envelope; the schema knows which it built.
        return not response.embedded
    return set(root["properties"]) != {"data"}


def _class_question(response: ResponseProto[Any]) -> bool:
    """Whether the caller asked the class itself a question: ``description=`` or an ``Annotated`` ``Question``."""
    if not isinstance(response, ResponseSchema):
        return bool(response.description)
    return bool(response.explicit_description or _markers(response))


def _field_specs(
    response: ResponseProto[Any], root: Mapping[str, Any], error: type[NotADecisionError]
) -> tuple[dict[str, DecisionSpec], dict[str, str]]:
    """Each field's question by field name, and the JSON key of each field whose key is an alias."""
    if root.get("additionalProperties") not in (None, False):
        raise error("`response_schema` allows additional properties; each question must be a declared field.")

    if not root["properties"]:
        # Both APIs reject a request without questions.
        raise error("`response_schema` has no fields; a class needs at least one question.")

    required = set(root.get("required", ()))
    fields = _fields_by_key(response)
    specs: dict[str, DecisionSpec] = {}
    keys: dict[str, str] = {}
    for key, prop in root["properties"].items():
        field: tuple[str, Any] = fields.get(key, (key, None))
        name, hint = field
        subject = f"Field {name!r} of `response_schema`"
        if key not in required:
            raise error(f"{subject} is optional or has a default; every question must be a required field.")

        bare = strip_annotated(hint) if hint is not None else None
        enum_type = bare if isinstance(bare, type) and issubclass(bare, Enum) else None
        markers = [m for m in get_args(hint)[1:] if isinstance(m, Question)] if get_origin(hint) is Annotated else []

        node = _resolve_ref(root, prop)
        marked = next((m.question for m in reversed(markers) if m.question), None)
        # ``Field(description=)`` sits on the property, an ``Enum`` docstring on the definition it refers to.
        question = marked or prop.get("description") or node.get("description")
        spec = DecisionSpec._classify(node, question, enum_type, markers, embedded=False, error=error, subject=subject)
        if not question or not question.strip():
            # OpenAI requires instructions on every question, and the prompt is shared by all fields.
            raise ValueError(
                f"{subject} has no question: annotate it with `Annotated[..., Question('...')]` "
                "or `Field(description=...)`, or document its `Enum`."
            )
        specs[name] = spec
        if key != name:
            keys[name] = key
    return specs, keys


def _fields_by_key(response: ResponseProto[Any]) -> dict[str, tuple[str, Any]]:
    """Each field of the class by its JSON key, as its name and its annotation with ``Annotated`` kept.

    The ``Question`` marker does not reach the JSON schema, so it is read from the annotations,
    which are keyed by field name; the schema is keyed by alias where a field has one.
    """
    if not isinstance(response, ResponseSchema):
        return {}
    cls = strip_annotated(response.types)
    if not isinstance(cls, type):
        return {}

    pydantic_fields: Mapping[str, FieldInfo] = getattr(cls, "__pydantic_fields__", None) or {}
    fields: dict[str, tuple[str, Any]] = {}
    for name, hint in get_type_hints(cls, include_extras=True).items():
        hint = _strip_qualifiers(hint)
        info = pydantic_fields.get(name) or _annotated_field_info(hint)
        fields[_json_key(name, info)] = (name, hint)
    return fields


def _strip_qualifiers(hint: Any) -> Any:
    """The annotation under ``TypedDict`` qualifiers, which ``get_type_hints`` keeps around ``Annotated``."""
    while get_origin(hint) in (Required, NotRequired, ReadOnly):
        hint = get_args(hint)[0]
    return hint


def _annotated_field_info(hint: Any) -> FieldInfo | None:
    if get_origin(hint) is not Annotated:
        return None
    return next((m for m in reversed(get_args(hint)[1:]) if isinstance(m, FieldInfo)), None)


def _json_key(name: str, info: FieldInfo | None) -> str:
    """The key Pydantic validates the field under, as its validation-mode JSON schema names it."""
    if info is None:
        return name
    alias = info.validation_alias
    if isinstance(alias, AliasChoices):
        alias = alias.choices[0]
    if isinstance(alias, str):
        return alias
    return info.alias or name


def _resolve_ref(root: Mapping[str, Any], node: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved: Mapping[str, Any] = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
        return resolved
    return node


def _decision_node(root: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """The node of the JSON schema ``root`` to decide on, and whether it was wrapped as ``{"data": ...}``."""
    node: Mapping[str, Any] = root
    properties = root.get("properties")
    embedded = root.get("type") == "object" and isinstance(properties, Mapping) and set(properties) == {"data"}
    if isinstance(properties, Mapping) and embedded:
        # The envelope's description is ag2 boilerplate, not a question for the model.
        node = {k: v for k, v in properties["data"].items() if k != "description"}

    return _resolve_ref(root, node), embedded


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


def _question(
    response: ResponseProto[Any], node: Mapping[str, Any], enum_type: type[Enum] | None, markers: list[Question]
) -> str | None:
    """The closest statement of the question: ``description=``, then ``Question``, then the ``Enum`` docstring."""
    if isinstance(response, ResponseSchema):
        explicit = response.explicit_description
        # An ``Enum`` docstring is the question; other types' docstrings are not, but a description
        # Pydantic lifted out of the schema (``RootModel`` + ``Field(description=)``) is.
        type_doc = None if enum_type else getattr(strip_annotated(response.types), "__doc__", None)
        docstring = None if response.description == type_doc else response.description
    else:
        explicit, docstring = response.description, None
    marked = next((m.question for m in reversed(markers) if m.question), None)
    return explicit or marked or docstring or node.get("description")


def _check_option_keys(markers: list[Question], values: list[str] | list[int]) -> None:
    """Fail on ``Question(options=)`` keys that name no option, which would otherwise be silently ignored."""
    # ``True == 1``, so a ``bool`` key must not match a score level.
    unknown = [k for m in markers for k in (m.options or {}) if isinstance(k, bool) or k not in values]
    if unknown:
        raise ValueError(
            f"`Question(options=...)` names {unknown!r}, which is not an option of the schema: "
            "use the values as the schema has them (`'billing'`, or `0` and `1` for a score)."
        )


def _options(
    values: list[str] | list[int], enum_type: type[Enum] | None, markers: list[Question]
) -> tuple[DecisionOption, ...]:
    names = {m.value: m.name for m in enum_type} if enum_type else {}
    docs = _member_docstrings(enum_type) if enum_type else {}
    marked = {k: v for m in markers for k, v in (m.options or {}).items()}
    return tuple(DecisionOption(v, names.get(v, str(v)), marked.get(v) or docs.get(v)) for v in values)


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
