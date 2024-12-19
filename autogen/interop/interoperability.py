# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Type

from ..tools import Tool
from .interoperable import Interoperable
from .registry import InteroperableRegistry

__all__ = ["Interoperable"]


class Interoperability:
    """
    A class to handle interoperability between different tool types.

    This class allows the conversion of tools to various interoperability classes and provides functionality
    for retrieving and registering interoperability classes.
    """

    def __init__(self) -> None:
        """
        Initializes an instance of the Interoperability class.

        This constructor does not perform any specific actions as the class is primarily used for its class
        methods to manage interoperability classes.
        """
        self.registry = InteroperableRegistry.get_instance()

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

    def get_interoperability_class(self, type: str) -> Type[Interoperable]:
        """
        Retrieves the interoperability class corresponding to the specified type.

        Args:
            type (str): The type of the interoperability class to retrieve.

        Returns:
            Type[Interoperable]: The interoperability class type.

        Raises:
            ValueError: If no interoperability class is found for the provided type.
        """
        supported_types = self.registry.get_supported_types()
        if type not in supported_types:
            supported_types_formated = ", ".join(["'t'" for t in supported_types])
            raise ValueError(
                f"Interoperability class {type} is not supported, supported types: {supported_types_formated}"
            )

        return self.registry.get_class(type)

    def get_supported_types(self) -> List[str]:
        """
        Returns a sorted list of all supported interoperability types.

        Returns:
            List[str]: A sorted list of strings representing the supported interoperability types.
        """
        return sorted(self.registry.get_supported_types())
