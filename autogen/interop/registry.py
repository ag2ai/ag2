# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Generic, List, Type, TypeVar

from .interoperable import Interoperable

__all__ = ["register_interoperable_class", "InteroperableRegistry"]

InteroperableClass = TypeVar("InteroperableClass", bound=type[Interoperable])


class InteroperableRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[Interoperable]] = {}

    def register(self, short_name: str, cls: InteroperableClass) -> InteroperableClass:
        if short_name in self._registry:
            raise ValueError(f"Duplicate registration for {short_name}")

        self._registry[short_name] = cls

        return cls

    def get_short_names(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_supported_types(self) -> list[str]:
        short_names = self.get_short_names()
        supported_types = [name for name in short_names if self._registry[name].get_unsupported_reason() is None]
        return supported_types

    def get_class(self, short_name: str) -> type[Interoperable]:
        return self._registry[short_name]

    @classmethod
    def get_instance(cls) -> "InteroperableRegistry":
        return _register


# global registry
_register = InteroperableRegistry()


# register decorator
def register_interoperable_class(short_name: str) -> Callable[[InteroperableClass], InteroperableClass]:
    """Register an Interoperable class in the global registry.

    Returns:
        Callable[[InteroperableClass], InteroperableClass]: Decorator function

    Example:
        ```python
        @register_interoperable_class("myinterop")
        class MyInteroperability(Interoperable):
            def convert_tool(self, tool: Any) -> Tool:
                # implementation
                ...
        ```
    """

    def inner(cls: InteroperableClass) -> InteroperableClass:
        global _register
        return _register.register(short_name, cls)

    return inner
