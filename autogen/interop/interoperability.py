# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Type

from ..tools import Tool
from .helpers import get_all_interoperability_classes
from .interoperable import Interoperable


class Interoperability:
    _interoperability_classes: Dict[str, Type[Interoperable]] = get_all_interoperability_classes()

    def __init__(self) -> None:
        pass

    def convert_tool(self, *, tool: Any, type: str) -> Tool:
        interop_cls = self.get_interoperability_class(type)
        interop = interop_cls()
        return interop.convert_tool(tool)

    @classmethod
    def get_interoperability_class(cls, type: str) -> Type[Interoperable]:
        if type not in cls._interoperability_classes:
            raise ValueError(f"Interoperability class {type} not found")
        return cls._interoperability_classes[type]

    @classmethod
    def supported_types(cls) -> List[str]:
        return sorted(cls._interoperability_classes.keys())

    @classmethod
    def register_interoperability_class(cls, name: str, interoperability_class: Type[Interoperable]) -> None:
        if not issubclass(interoperability_class, Interoperable):
            raise ValueError(
                f"Expected a class implementing `Interoperable` protocol, got {type(interoperability_class)}"
            )

        cls._interoperability_classes[name] = interoperability_class
