# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC
from typing import Any, Callable, get_type_hints

from fast_depends import Depends as FastDepends
from fast_depends import inject

__all__ = [
    "BaseContext",
    "ChatContext",
    "Depends",
    "Field",
    "inject_params",
]


class BaseContext(ABC):
    pass


class ChatContext(BaseContext):
    messages: list[str] = []


def Depends(x: Any) -> Any:
    if isinstance(x, BaseContext):
        return FastDepends(lambda: x)

    return FastDepends(x)


def _remove_injected_params_from_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    remove_from_signature = []
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        param_annotation = param.annotation.__args__[0] if hasattr(param.annotation, "__args__") else param.annotation
        if isinstance(param_annotation, type) and issubclass(param_annotation, BaseContext):
            remove_from_signature.append(param.name)

    new_signature = sig.replace(parameters=[p for p in sig.parameters.values() if p.name not in remove_from_signature])
    func.__signature__ = new_signature  # type: ignore[attr-defined]
    return func


class Field:
    def __init__(self, description: str) -> None:
        self._description = description

    @property
    def description(self) -> str:
        return self._description


def _string_metadata_to_description_field(func: Callable[..., Any]) -> Callable[..., Any]:
    type_hints = get_type_hints(func, include_extras=True)

    for _, annotation in type_hints.items():
        if hasattr(annotation, "__metadata__"):
            metadata = annotation.__metadata__
            if metadata and isinstance(metadata[0], str):
                # Replace string metadata with DescriptionField
                annotation.__metadata__ = (Field(description=metadata[0]),)
    return func


def inject_params(f: Callable[..., Any]) -> Callable[..., Any]:
    f = _string_metadata_to_description_field(f)
    f = inject(f)
    f = _remove_injected_params_from_signature(f)

    return f
