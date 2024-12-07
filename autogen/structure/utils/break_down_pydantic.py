# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from inspect import isclass
from typing import Any, Dict, TypeVar, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def break_down_pydantic_annotation(
    model: type[T], max_depth: int | None = None, current_depth: int = 0
) -> Dict[str, Any]:
    """
    Break down the type annotations of a Pydantic model into a dictionary.

    This function recursively processes Pydantic models, converting their
    field annotations into a dictionary structure. It handles nested models
    and lists of models.

    Args:
        model: The Pydantic model class to break down.
        max_depth: Maximum depth for recursion. None for no limit.
        current_depth: Current recursion depth (used internally).

    Returns:
        A dictionary representing the structure of the model's annotations.

    Raises:
        TypeError: If the input is not a Pydantic model.
        RecursionError: If max recursion depth is reached.

    Example:
        >>> from pydantic import BaseModel
        >>> class SubModel(BaseModel):
        ...     field1: int
        ...     field2: str
        >>> class MainModel(BaseModel):
        ...     sub: SubModel
        ...     items: list[SubModel]
        >>> result = break_down_annotation(MainModel)
        >>> print(result)
        {
            'sub': {'field1': <class 'int'>, 'field2': <class 'str'>},
            'items': [{'field1': <class 'int'>, 'field2': <class 'str'>}]
        }
    """

    if not _is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: Dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if _is_pydantic_model(v):
            out[k] = break_down_pydantic_annotation(v, max_depth, current_depth + 1)
        elif origin is list:
            args = get_args(v)
            if args and _is_pydantic_model(args[0]):
                out[k] = [break_down_pydantic_annotation(args[0], max_depth, current_depth + 1)]
            else:
                out[k] = [args[0] if args else Any]
        else:
            out[k] = v

    return out


def _is_pydantic_model(x: Any) -> bool:
    return isclass(x) and issubclass(x, BaseModel)


__all__ = ["break_down_pydantic_annotation"]
