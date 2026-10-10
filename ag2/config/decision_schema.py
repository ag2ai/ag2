# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Read a decision question out of a ``response_schema``, for the decision-only providers.

TypeSafe's Jev and OpenAI's Decisions API both answer one typed question instead of
generating text, and both take the question from the agent's ``response_schema``. The
provider mappers share how that schema is read; how it becomes a request is theirs.
"""

import ast
import inspect
import textwrap
from collections.abc import Mapping
from enum import Enum
from functools import cache
from typing import Any

from ag2.response import ResponseProto, ResponseSchema

# Python 3.10 gives an undocumented ``Enum`` this docstring; later versions give ``None``.
_PY310_ENUM_DOC = "An enumeration."

__all__ = ("decision_node", "explicit_description", "member_docstrings", "option_docs")


def explicit_description(response: ResponseProto[Any] | None) -> str | None:
    """The question description, retaining enum docstrings but ignoring other type docstring fallbacks."""
    if isinstance(response, ResponseSchema) and isinstance(response.types, type) and issubclass(response.types, Enum):
        return None if response.description == _PY310_ENUM_DOC else response.description
    if response is None or (
        isinstance(response, ResponseSchema) and response.description == getattr(response.types, "__doc__", None)
    ):
        return None
    return response.description


def option_docs(response: ResponseProto[Any] | None) -> dict[Any, str]:
    """Each ``Enum`` member's value mapped to the docstring under it, if the schema was built from an ``Enum``."""
    if isinstance(response, ResponseSchema) and isinstance(response.types, type) and issubclass(response.types, Enum):
        return member_docstrings(response.types)
    return {}


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


def decision_node(root: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Return the node of the JSON schema ``root`` to decide on, and whether ``ResponseSchema`` wrapped it as ``{"data": ...}``."""
    node: Mapping[str, Any] = root
    properties = root.get("properties")
    embedded = root.get("type") == "object" and isinstance(properties, Mapping) and set(properties) == {"data"}
    if embedded:
        # The envelope's description is ag2 boilerplate, not a question for the model.
        node = {k: v for k, v in properties["data"].items() if k != "description"}  # type: ignore[index]

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        node = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
    return node, embedded
