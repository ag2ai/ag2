# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["export_module"]

from typing import Callable, TypeVar

T = TypeVar("T")


def export_module(module: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        setattr(cls, "__module__", module)
        return cls

    return decorator
