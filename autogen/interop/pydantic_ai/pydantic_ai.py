# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


from functools import wraps
from inspect import signature
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai.tools import Tool as PydanticAITool

from ...tools import Tool
from ..interoperability import Interoperable

__all__ = ["PydanticAIInteroperability"]


class PydanticAIInteroperability(Interoperable):
    @staticmethod
    def inject_params(f: Callable[..., Any], ctx: Any) -> Callable[..., Any]:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("ctx", None)
            return f(**kwargs, ctx=ctx)

        sig = signature(f)
        new_params = [param for name, param in sig.parameters.items() if name != "ctx"]

        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

        return wrapper

    def convert_tool(self, tool: Any, deps: Any = None) -> Tool:
        if not isinstance(tool, PydanticAITool):
            raise ValueError(f"Expected an instance of `pydantic_ai.tools.Tool`, got {type(tool)}")

        # needed for type checking
        pydantic_ai_tool: PydanticAITool = tool  # type: ignore[no-any-unimported]

        if deps is not None:
            ctx = RunContext(
                deps=deps,
                retry=pydantic_ai_tool.max_retries,  # TODO: check what to do with this
                messages=[],  # TODO: check what to do with this
                tool_name=pydantic_ai_tool.name,
            )
            func = PydanticAIInteroperability.inject_params(f=pydantic_ai_tool.function, ctx=ctx)
        else:
            func = pydantic_ai_tool.function

        return Tool(
            name=pydantic_ai_tool.name,
            description=pydantic_ai_tool.description,
            func=func,
        )
