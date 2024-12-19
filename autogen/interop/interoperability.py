# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Type

from ..tools import Tool
from .helpers import get_all_interoperability_classes
from .interoperable import Interoperable

__all__ = ["Interoperable"]


class Interoperability:
    """
    A class to handle interoperability between different tool types.

    This class allows the conversion of tools to various interoperability classes and provides functionality
    for retrieving and registering interoperability classes.
    """

    _interoperability_classes: Dict[str, Type[Interoperable]] = get_all_interoperability_classes()

    def __init__(self) -> None:
        """
        Initializes an instance of the Interoperability class.

        This constructor does not perform any specific actions as the class is primarily used for its class
        methods to manage interoperability classes.
        """
        pass

    def convert_tool(self, *, tool: Any, type: str, **kwargs: Any) -> Tool:
        """
        Converts a given tool to an instance of a specified interoperability type.

        Args:
            tool (Any): The tool object to be converted.
            type (str): The type of interoperability to convert the tool to.
            **kwargs (Any): Additional arguments to be passed during conversion.

        Returns:
            Tool: The converted tool.

        Raises:
            ValueError: If the interoperability class for the provided type is not found.
        """
        interop_cls = self.get_interoperability_class(type)
        interop = interop_cls()
        return interop.convert_tool(tool, **kwargs)

    @classmethod
    def get_interoperability_class(cls, type: str) -> Type[Interoperable]:
        """
        Retrieves the interoperability class corresponding to the specified type.

        Args:
            type (str): The type of the interoperability class to retrieve.

        Returns:
            Type[Interoperable]: The interoperability class type.

        Raises:
            ValueError: If no interoperability class is found for the provided type.
        """
        if type not in cls._interoperability_classes:
            raise ValueError(f"Interoperability class {type} not found")
        return cls._interoperability_classes[type]

    @classmethod
    def supported_types(cls) -> List[str]:
        """
        Returns a sorted list of all supported interoperability types.

        Returns:
            List[str]: A sorted list of strings representing the supported interoperability types.
        """
        return sorted(cls._interoperability_classes.keys())

    @classmethod
    def register_interoperability_class(cls, name: str, interoperability_class: Type[Interoperable]) -> None:
        """
        Registers a new interoperability class with the given name.

        Args:
            name (str): The name to associate with the interoperability class.
            interoperability_class (Type[Interoperable]): The class implementing the Interoperable protocol.

        Raises:
            ValueError: If the provided class does not implement the Interoperable protocol.
        """
        if not issubclass(interoperability_class, Interoperable):
            raise ValueError(
                f"Expected a class implementing `Interoperable` protocol, got {type(interoperability_class)}"
            )

        cls._interoperability_classes[name] = interoperability_class
