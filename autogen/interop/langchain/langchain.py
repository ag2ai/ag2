# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from langchain_core.tools import BaseTool as LangchainTool

from ...tools import Tool
from ..interoperable import Interoperable

__all__ = ["LangchainInteroperability"]


class LangchainInteroperability(Interoperable):
    """
    A class implementing the `Interoperable` protocol for converting Langchain tools
    into a general `Tool` format.

    This class takes a `LangchainTool` and converts it into a standard `Tool` object,
    ensuring compatibility between Langchain tools and other systems that expect
    the `Tool` format.
    """

    def convert_tool(self, tool: Any, **kwargs: Any) -> Tool:
        """
        Converts a given Langchain tool into a general `Tool` format.

        This method verifies that the provided tool is a valid `LangchainTool`,
        processes the tool's input and description, and returns a standardized
        `Tool` object.

        Args:
            tool (Any): The tool to convert, expected to be an instance of `LangchainTool`.
            **kwargs (Any): Additional arguments, which are not supported by this method.

        Returns:
            Tool: A standardized `Tool` object converted from the Langchain tool.

        Raises:
            ValueError: If the provided tool is not an instance of `LangchainTool`, or if
                        any additional arguments are passed.
        """
        if not isinstance(tool, LangchainTool):
            raise ValueError(f"Expected an instance of `langchain_core.tools.BaseTool`, got {type(tool)}")
        if kwargs:
            raise ValueError(f"The LangchainInteroperability does not support any additional arguments, got {kwargs}")

        # needed for type checking
        langchain_tool: LangchainTool = tool  # type: ignore[no-any-unimported]

        def func(tool_input: langchain_tool.args_schema) -> Any:  # type: ignore[no-any-unimported]
            return langchain_tool.run(tool_input.model_dump())

        return Tool(
            name=langchain_tool.name,
            description=langchain_tool.description,
            func=func,
        )
