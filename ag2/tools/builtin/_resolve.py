# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from types import EllipsisType
from typing import Any, Protocol

from ag2.annotations import Variable


class HasVariables(Protocol):
    """Anything a ``Variable`` resolves against: a conversation, or one MCP request."""

    @property
    def variables(self) -> Mapping[str, Any]: ...


def resolve_variable(
    value: Any,
    context: HasVariables,
    *,
    param_name: str = "",
) -> Any:
    """If value is a Variable marker, resolve from context.variables. Otherwise return as-is."""
    if not isinstance(value, Variable):
        return value

    key = value.name or param_name
    if key in context.variables:
        return context.variables[key]
    if value.default is not Ellipsis:
        return value.default
    # `is not Ellipsis` is the same test, but only `isinstance` narrows the
    # `EllipsisType` out of the union for the call below.
    if not isinstance(value.default_factory, EllipsisType):
        return value.default_factory()

    raise KeyError(f"Context variable {key!r} not found and no default provided")
