# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC
from typing import Any, Callable, Iterable, get_type_hints

from fast_depends import Depends as FastDepends
from fast_depends import inject
from fast_depends.dependencies import model

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


def _is_base_context_param(param: inspect.Parameter) -> bool:
    # param.annotation.__args__[0] is used to handle Annotated[MyContext, Depends(MyContext(b=2))]
    param_annotation = param.annotation.__args__[0] if hasattr(param.annotation, "__args__") else param.annotation
    return isinstance(param_annotation, type) and issubclass(param_annotation, BaseContext)


def _is_depends_param(param: inspect.Parameter) -> bool:
    return isinstance(param.default, model.Depends) or (
        hasattr(param.annotation, "__metadata__")
        and type(param.annotation.__metadata__) == tuple
        and isinstance(param.annotation.__metadata__[0], model.Depends)
    )


def _remove_params(func: Callable[..., Any], sig: inspect.Signature, params: Iterable[str]) -> None:
    new_signature = sig.replace(parameters=[p for p in sig.parameters.values() if p.name not in params])
    func.__signature__ = new_signature  # type: ignore[attr-defined]


def _remove_injected_params_from_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    sig = inspect.signature(func)
    params_to_remove = [p.name for p in sig.parameters.values() if _is_base_context_param(p) or _is_depends_param(p)]
    _remove_params(func, sig, params_to_remove)
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
