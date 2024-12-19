# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from ..tools import Tool

__all__ = ["Interoperable"]


@runtime_checkable
class Interoperable(Protocol):
    """
    A Protocol defining the interoperability interface for tool conversion.

    This protocol ensures that any class implementing it provides the method
    `convert_tool` to convert a given tool into a desired format or type.
    """

    def convert_tool(self, tool: Any, **kwargs: Any) -> Tool:
        """
        Converts a given tool to a desired format or type.

        This method should be implemented by any class adhering to the `Interoperable` protocol.

        Args:
            tool (Any): The tool object to be converted.
            **kwargs (Any): Additional parameters to pass during the conversion process.

        Returns:
            Tool: The converted tool in the desired format or type.
        """
        ...