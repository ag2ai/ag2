# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from langchain_core.tools import BaseTool as LangchainTool

from ...tools import Tool
from ..interoperable import Interoperable

__all__ = ["LangchainInteroperability"]


class LangchainInteroperability(Interoperable):
    def convert_tool(self, tool: Any) -> Tool:
        if not isinstance(tool, LangchainTool):
            raise ValueError(f"Expected an instance of `langchain_core.tools.BaseTool`, got {type(tool)}")

        # needed for type checking
        langchain_tool: LangchainTool = tool  # type: ignore[no-any-unimported]

        def func(tool_input: langchain_tool.args_schema) -> Any:  # type: ignore[no-any-unimported]
            return langchain_tool.run(tool_input.model_dump())

        return Tool(
            name=langchain_tool.name,
            description=langchain_tool.description,
            func=func,
        )
