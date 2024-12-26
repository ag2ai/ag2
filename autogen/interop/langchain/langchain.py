# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Optional

from ...tools import Tool
from ..registry import register_interoperable_class

__all__ = ["LangChainInteroperability"]


@register_interoperable_class("langchain")
class LangChainInteroperability:
    """
    A class implementing the `Interoperable` protocol for converting Langchain tools
    into a general `Tool` format.

    This class takes a `LangchainTool` and converts it into a standard `Tool` object,
    ensuring compatibility between Langchain tools and other systems that expect
    the `Tool` format.
    """

    @classmethod
    def convert_tool(cls, tool: Any, **kwargs: Any) -> Tool:
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
        from langchain_core.tools import BaseTool as LangchainTool

        if not isinstance(tool, LangchainTool):
            raise ValueError(f"Expected an instance of `langchain_core.tools.BaseTool`, got {type(tool)}")
        if kwargs:
            raise ValueError(f"The LangchainInteroperability does not support any additional arguments, got {kwargs}")

        # needed for type checking
        langchain_tool: LangchainTool = tool  # type: ignore

        def func(tool_input: langchain_tool.args_schema) -> Any:  # type: ignore
            return langchain_tool.run(tool_input.model_dump())

        return Tool(
            name=langchain_tool.name,
            description=langchain_tool.description,
            func=func,
        )

    @classmethod
    def get_unsupported_reason(cls) -> Optional[str]:
        if sys.version_info < (3, 9):
            return "This submodule is only supported for Python versions 3.9 and above"

        try:
            import langchain_core.tools
        except ImportError:
            return (
                "Please install `interop-langchain` extra to use this module:\n\n\tpip install ag2[interop-langchain]"
            )

        return None
