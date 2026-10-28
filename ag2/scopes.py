# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Invocation variables with explicit delegation ownership and sharing.

Shared bindings provide visibility, not synchronization. Authors supply locks
for compound concurrent changes, including changes from synchronous tools.
"""

from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class Scope(str, Enum):
    LOCAL = "local"
    COPIED = "copied"
    SHARED = "shared"


class ScopeError(ValueError):
    """A variable declaration conflicts with its ownership or inherited scope."""


@dataclass(frozen=True, slots=True, init=False)
class Var:
    value: Any
    scope: Scope = Scope.COPIED

    def __init__(self, value: Any, scope: Scope | str = Scope.COPIED) -> None:
        try:
            resolved = Scope(scope)
        except ValueError as exc:
            raise ScopeError(f"Unknown variable scope: {scope!r}") from exc
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "scope", resolved)


class ScopedVariables(MutableMapping[str, Any]):
    """A variable view with branch-owned values and tree-owned shared bindings."""

    def __init__(
        self,
        own: dict[str, Any],
        shared: dict[str, Any],
        declared: dict[str, Scope],
        *,
        root: bool,
    ) -> None:
        self._own = own
        self._shared = shared
        self._declared = declared
        self._root = root

    @property
    def declared(self) -> Mapping[str, Scope]:
        return MappingProxyType(self._declared)

    def __getitem__(self, key: str) -> Any:
        if self._declared.get(key) == Scope.SHARED:
            return self._shared[key]
        return self._own[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Var):
            self._declare(key, value.scope)
            value = value.value
        if self._declared.get(key) == Scope.SHARED:
            self._shared[key] = value
        else:
            self._own[key] = value

    def __delitem__(self, key: str) -> None:
        if self._declared.get(key) == Scope.SHARED:
            del self._shared[key]
        else:
            del self._own[key]

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys((*self._own, *self._shared)))

    def __len__(self) -> int:
        return len(dict.fromkeys((*self._own, *self._shared)))

    def copy(self) -> dict[str, Any]:
        """Return a shallow value snapshot, without exporting scope declarations."""
        return dict(self)

    def _declare(self, key: str, scope: Scope) -> None:
        current = self._declared.get(key)
        if scope == Scope.SHARED and key.startswith(("ag:", "a2a:")):
            raise ScopeError(f"Reserved variable {key!r} cannot be shared")
        if current is not None and current != scope:
            raise ScopeError(f"Variable {key!r} cannot be redeclared differently: {current.value} -> {scope.value}")
        if not self._root and scope == Scope.SHARED and current != Scope.SHARED:
            raise ScopeError(f"Cannot declare new shared variable {key!r} below the root")
        self._declared[key] = scope


def merge_variables(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any] | ScopedVariables:
    """Build a fresh root view, or overlay agent defaults on an inherited view."""
    if isinstance(overrides, ScopedVariables):
        for key, value in defaults.items():
            if isinstance(value, Var):
                overrides._declare(key, value.scope)
            if key not in overrides:
                overrides[key] = deepcopy(value)
        return overrides

    merged = dict(defaults)
    for key, value in (overrides or {}).items():
        previous = merged.get(key)
        if isinstance(previous, Var):
            if isinstance(value, Var) and previous.scope != value.scope:
                raise ScopeError(f"Variable {key!r} cannot be redeclared differently")
            if not isinstance(value, Var):
                value = Var(value, previous.scope)
        merged[key] = value
    merged = deepcopy(merged)
    if not any(isinstance(value, Var) for value in merged.values()):
        return merged
    result = ScopedVariables({}, {}, {}, root=True)
    result.update(merged)
    return result


def overlay_variables(variables: MutableMapping[str, Any], values: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply continuation or plugin values while preserving scope declarations."""
    if isinstance(variables, ScopedVariables):
        variables.update(values)
        return variables
    if any(isinstance(value, Var) for value in values.values()):
        result = ScopedVariables(dict(variables), {}, {}, root=True)
        result.update(deepcopy(values))
        return result
    variables.update(values)
    return variables


def derive_child_variables(variables: Mapping[str, Any]) -> ScopedVariables:
    """Deep-copy a branch's values, excluding local and control-plane keys."""
    declared = dict(variables.declared) if isinstance(variables, ScopedVariables) else {}
    shared = variables._shared if isinstance(variables, ScopedVariables) else {}
    own = {
        key: value
        for key, value in variables.items()
        if declared.get(key) not in (Scope.LOCAL, Scope.SHARED) and not key.startswith(("ag:", "a2a:"))
    }
    child_declarations = {key: scope for key, scope in declared.items() if scope == Scope.SHARED or key in own}
    return ScopedVariables(deepcopy(own), shared, child_declarations, root=False)


__all__ = (
    "Scope",
    "ScopeError",
    "ScopedVariables",
    "Var",
    "derive_child_variables",
    "merge_variables",
    "overlay_variables",
)
