# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, cast

from crewai.tools import BaseTool as CrewAITool

from ...tools import Tool
from ..interoperability import Interoperable

__all__ = ["CrewAIInteroperability"]


def _sanitize_name(s: str) -> str:
    return re.sub(r"\W|^(?=\d)", "_", s)


class CrewAIInteroperability(Interoperable):
    def convert_tool(self, tool: Any) -> Tool:
        if not isinstance(tool, CrewAITool):
            raise ValueError(f"Expected an instance of `crewai.tools.BaseTool`, got {type(tool)}")

        # needed for type checking
        crewai_tool: CrewAITool = tool  # type: ignore[no-any-unimported]

        name = _sanitize_name(crewai_tool.name)
        description = crewai_tool.description.split("Tool Description: ")[-1]

        def func(args: crewai_tool.args_schema) -> Any:  # type: ignore[no-any-unimported]
            return crewai_tool.run(**args.model_dump())

        return Tool(
            name=name,
            description=description,
            func=func,
        )
