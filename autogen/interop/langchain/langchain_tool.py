# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from ...doc_utils import export_module
from ...import_utils import optional_import_block, require_optional_import
from ...tools import Tool
from ..registry import register_interoperable_class

__all__ = ["LangChainInteroperability"]

with optional_import_block():
    from langchain_core.tools import BaseTool as LangchainTool


def _is_async_langchain_tool(tool: Any) -> bool:
    """Return True when the Langchain tool implements an async path.

    Two cases qualify:
    1. Langchain's StructuredTool / Tool exposes a `coroutine` attribute that
       holds the original async function. It is `None` when the wrapped
       function is sync, and a coroutine function when it is async.
    2. BaseTool subclasses that override `_arun` directly (without going via
       StructuredTool's `coroutine` indirection) are async-native; the default
       `BaseTool._arun` only delegates back to `_run` in a worker thread, so
       routing those through `arun` would not buy us anything.
    """
    if hasattr(tool, "coroutine"):
        # StructuredTool / Tool path: the only reliable signal is the coroutine
        # field. The class always overrides `_arun`, so the class-level check
        # below would mis-classify sync tools as async.
        return getattr(tool, "coroutine") is not None

    base_arun = getattr(LangchainTool, "_arun", None)
    cls_arun = getattr(type(tool), "_arun", None)
    return cls_arun is not base_arun and asyncio.iscoroutinefunction(cls_arun)


@register_interoperable_class("langchain")
@export_module("autogen.interop")
class LangChainInteroperability:
    """A class implementing the `Interoperable` protocol for converting Langchain tools
    into a general `Tool` format.

    This class takes a `LangchainTool` and converts it into a standard `Tool` object,
    ensuring compatibility between Langchain tools and other systems that expect
    the `Tool` format.
    """

    @classmethod
    @require_optional_import("langchain_core", "interop-langchain")
    def convert_tool(cls, tool: Any, **kwargs: Any) -> Tool:
        """Converts a given Langchain tool into a general `Tool` format.

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

        model_type = langchain_tool.get_input_schema()

        if _is_async_langchain_tool(langchain_tool):
            # Use Langchain's async entry point so that async-native tools
            # don't block the event loop when invoked from `a_initiate_chat`
            # or any other async caller. AG2's Tool already accepts coroutine
            # functions as `func_or_tool`. See issue #1402.
            async def func(tool_input: model_type) -> Any:  # type: ignore[valid-type]
                return await langchain_tool.arun(tool_input.model_dump())  # type: ignore[attr-defined]
        else:

            def func(tool_input: model_type) -> Any:  # type: ignore[valid-type, no-redef]
                return langchain_tool.run(tool_input.model_dump())  # type: ignore[attr-defined]

        return Tool(
            name=langchain_tool.name,
            description=langchain_tool.description,
            func_or_tool=func,
        )

    @classmethod
    def get_unsupported_reason(cls) -> str | None:
        with optional_import_block() as result:
            import langchain_core.tools  # noqa: F401

        if not result.is_successful:
            return (
                "Please install `interop-langchain` extra to use this module:\n\n\tpip install ag2[interop-langchain]"
            )

        return None
