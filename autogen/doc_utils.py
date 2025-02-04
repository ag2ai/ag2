# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["export_module"]

from typing import Callable, TypeVar

T = TypeVar("T")

_PDOC_PLACEHOLDER = "__ADD_MODULE_TO_PDOC_AND_SET_FALSE__"


def export_module(module: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        if hasattr(cls, "_set__exported_module__"):
            cls._set__exported_module__(module)
            if not cls._get__exported_module__():
                original_module = getattr(cls, "__module__")
                setattr(cls, "__module__", f"{_PDOC_PLACEHOLDER}{original_module}")
            return cls
        setattr(cls, "__exported_module__", module)
        return cls

    return decorator
