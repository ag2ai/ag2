# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC
from contextlib import contextmanager
from typing import Any, Callable, Generator, get_type_hints

from fast_depends import Depends as FastDepends
from pydantic import Field

__all__ = ["BaseContext", "ChatContext", "Depends", "remove_injected_params_from_signature", "annotation_context"]


class BaseContext(ABC):
    pass


class ChatContext(BaseContext):
    messages: list[str] = []


def Depends(x: Any) -> Any:
    if isinstance(x, BaseContext):
        return FastDepends(lambda: x)

    return FastDepends(x)


def remove_injected_params_from_signature(func: Callable[..., Any]) -> None:
    remove_from_signature = []
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        param_annotation = param.annotation.__args__[0] if hasattr(param.annotation, "__args__") else param.annotation
        if isinstance(param_annotation, type) and issubclass(param_annotation, BaseContext):
            remove_from_signature.append(param.name)

    new_signature = sig.replace(parameters=[p for p in sig.parameters.values() if p.name not in remove_from_signature])
    func.__signature__ = new_signature  # type: ignore[attr-defined]


@contextmanager
def annotation_context(func: Callable[..., Any]) -> Generator[Callable[..., Any], None, None]:
    original_annotations = get_type_hints(func, include_extras=True).copy()

    try:
        for _, annotation in original_annotations.items():
            if hasattr(annotation, "__metadata__"):
                metadata = annotation.__metadata__
                if metadata and isinstance(metadata[0], str):
                    # Replace string metadata with Pydantic's Field
                    annotation.__metadata__ = (Field(description=metadata[0]),)
        yield func
    finally:
        func.__annotations__ = original_annotations
