# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.beta.annotations import Variable
from autogen.beta.context import ConversationContext


def resolve_variable(
    value: Any,
    context: ConversationContext,
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
    if value.default_factory is not Ellipsis:
        return value.default_factory()

    raise KeyError(f"Context variable {key!r} not found and no default provided")
