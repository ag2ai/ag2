# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional

from pydantic_ai import RunContext
from pydantic_ai.tools import Tool as PydanticAITool

from ...tools import PydanticAITool as AG2PydanticAITool
from ..interoperability import Interoperable

__all__ = ["PydanticAIInteroperability"]


class PydanticAIInteroperability(Interoperable):
    @staticmethod
    def inject_params(  # type: ignore[no-any-unimported]
        ctx: Optional[RunContext[Any]],
        tool: PydanticAITool,
    ) -> Callable[..., Any]:
        max_retries = tool.max_retries if tool.max_retries is not None else 1
        f = tool.function

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if tool.current_retry >= max_retries:
                raise ValueError(f"{tool.name} failed after {max_retries} retries")

            try:
                if ctx is not None:
                    kwargs.pop("ctx", None)
                    ctx.retry = tool.current_retry
                    result = f(**kwargs, ctx=ctx)
                else:
                    result = f(**kwargs)
                tool.current_retry = 0
            except Exception as e:
                tool.current_retry += 1
                raise e

            return result

        sig = signature(f)
        if ctx is not None:
            new_params = [param for name, param in sig.parameters.items() if name != "ctx"]
        else:
            new_params = list(sig.parameters.values())

        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

        return wrapper

    def convert_tool(self, tool: Any, deps: Any = None) -> AG2PydanticAITool:
        if not isinstance(tool, PydanticAITool):
            raise ValueError(f"Expected an instance of `pydantic_ai.tools.Tool`, got {type(tool)}")

        # needed for type checking
        pydantic_ai_tool: PydanticAITool = tool  # type: ignore[no-any-unimported]

        if deps is not None:
            ctx = RunContext(
                deps=deps,
                retry=0,
                messages=None,  # TODO: check what to do with this
                tool_name=pydantic_ai_tool.name,
            )
        else:
            ctx = None

        func = PydanticAIInteroperability.inject_params(
            ctx=ctx,
            tool=pydantic_ai_tool,
        )

        return AG2PydanticAITool(
            name=pydantic_ai_tool.name,
            description=pydantic_ai_tool.description,
            func=func,
            parameters_json_schema=pydantic_ai_tool._parameters_json_schema,
        )
