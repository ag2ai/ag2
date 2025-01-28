# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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
