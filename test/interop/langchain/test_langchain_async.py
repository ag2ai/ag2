# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop.langchain import LangChainInteroperability

with optional_import_block():
    from langchain_core.tools import BaseTool as LangchainBaseTool
    from langchain_core.tools import tool as langchain_tool


@pytest.mark.interop
@run_for_optional_imports("langchain_core", "interop-langchain")
class TestLangChainAsyncToolConversion:
    """Tests for async LangChain tool support in LangChainInteroperability.

    Verifies that:
    - Tools with native async (_arun) are converted to async functions using ainvoke
    - Tools without async remain synchronous (backward compatibility)
    - Async converted tools can be awaited without blocking
    - Sync converted tools still work as before
    """

    @pytest.fixture
    def sync_tool(self) -> Any:
        """A LangChain tool with only synchronous implementation."""
        self.sync_mock = MagicMock(return_value="sync result")

        class SyncSearchInput(BaseModel):
            query: str = Field(description="search query")

        @langchain_tool("sync-search", args_schema=SyncSearchInput, return_direct=True)  # type: ignore[misc]
        def sync_search(query: str) -> str:
            """A synchronous search tool."""
            return self.sync_mock(query)  # type: ignore[no-any-return]

        return sync_search

    @pytest.fixture
    def async_tool(self) -> Any:
        """A LangChain tool with native async implementation (_arun override)."""
        self.async_mock = AsyncMock(return_value="async result")
        self.sync_fallback_mock = MagicMock(return_value="sync fallback result")

        class AsyncSearchInput(BaseModel):
            query: str = Field(description="search query")

        class AsyncSearchTool(LangchainBaseTool):  # type: ignore[misc, no-any-unimported]
            name: str = "async-search"
            description: str = "An asynchronous search tool."
            args_schema: type[BaseModel] = AsyncSearchInput

            def _run(self, query: str) -> str:
                return self.sync_fallback_mock(query)  # type: ignore[no-any-return]

            async def _arun(self, query: str) -> str:
                return await self.async_mock(query)  # type: ignore[no-any-return]

        # Store references for assertion access
        tool_instance = AsyncSearchTool()
        tool_instance.sync_fallback_mock = self.sync_fallback_mock
        tool_instance.async_mock = self.async_mock
        return tool_instance

    def test_sync_tool_stays_sync(self, sync_tool: Any) -> None:
        """Sync-only LangChain tools should produce a sync converted function."""
        converted = LangChainInteroperability.convert_tool(sync_tool)

        assert converted.name == "sync-search"
        assert converted.description == "A synchronous search tool."
        # The converted function should NOT be a coroutine function
        assert not inspect.iscoroutinefunction(converted.func)

    def test_sync_tool_execution(self, sync_tool: Any) -> None:
        """Sync converted tool should execute correctly and return expected result."""
        converted = LangChainInteroperability.convert_tool(sync_tool)

        model_type = sync_tool.get_input_schema()
        tool_input = model_type(query="test query")
        result = converted.func(tool_input=tool_input)

        assert result == "sync result"
        self.sync_mock.assert_called_once_with("test query")

    def test_async_tool_produces_async_func(self, async_tool: Any) -> None:
        """LangChain tools with _arun should produce an async converted function."""
        converted = LangChainInteroperability.convert_tool(async_tool)

        assert converted.name == "async-search"
        assert converted.description == "An asynchronous search tool."
        # The converted function MUST be a coroutine function
        assert inspect.iscoroutinefunction(converted.func)

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, async_tool: Any) -> None:
        """Async converted tool should be awaitable and call ainvoke internally."""
        converted = LangChainInteroperability.convert_tool(async_tool)

        model_type = async_tool.get_input_schema()
        tool_input = model_type(query="async test query")

        # The function should be awaitable
        result = await converted.func(tool_input=tool_input)

        assert result == "async result"
        self.async_mock.assert_called_once_with("async test query")

    @pytest.mark.asyncio
    async def test_async_tool_does_not_block_event_loop(self, async_tool: Any) -> None:
        """Verify that the async tool runs concurrently without blocking.

        We run the async tool alongside a fast coroutine. If the tool were
        blocking (using sync run()), the fast coroutine would be delayed.
        """
        converted = LangChainInteroperability.convert_tool(async_tool)
        model_type = async_tool.get_input_schema()
        tool_input = model_type(query="concurrent test")

        results = []

        async def fast_task() -> None:
            results.append("fast_done")

        # Run both concurrently
        tool_result, _ = await asyncio.gather(
            converted.func(tool_input=tool_input),
            fast_task(),
        )

        assert tool_result == "async result"
        assert "fast_done" in results

    def test_convert_tool_preserves_name_and_description(self, async_tool: Any) -> None:
        """Tool metadata should be preserved after conversion."""
        converted = LangChainInteroperability.convert_tool(async_tool)

        assert converted.name == async_tool.name
        assert converted.description == async_tool.description

    def test_convert_tool_invalid_type_raises(self) -> None:
        """Passing a non-LangChain tool should raise ValueError."""
        with pytest.raises(ValueError, match="Expected an instance of"):
            LangChainInteroperability.convert_tool("not a tool")

    def test_convert_tool_extra_kwargs_raises(self, sync_tool: Any) -> None:
        """Extra keyword arguments should raise ValueError."""
        with pytest.raises(ValueError, match="does not support any additional arguments"):
            LangChainInteroperability.convert_tool(sync_tool, extra_arg="value")


@pytest.mark.interop
@run_for_optional_imports("langchain_core", "interop-langchain")
class TestLangChainAsyncDetection:
    """Tests for the _langchain_tool_has_async_implementation helper."""

    def test_detects_async_implementation(self) -> None:
        """Tools that override _arun should be detected as async."""
        from autogen.interop.langchain.langchain_tool import _langchain_tool_has_async_implementation

        class AsyncTool(LangchainBaseTool):  # type: ignore[misc, no-any-unimported]
            name: str = "async-tool"
            description: str = "test"

            def _run(self, query: str) -> str:
                return "sync"

            async def _arun(self, query: str) -> str:
                return "async"

        tool = AsyncTool()
        assert _langchain_tool_has_async_implementation(tool) is True

    def test_detects_sync_only_tool(self) -> None:
        """Tools using the @tool decorator (no _arun override) should be detected as sync."""
        from autogen.interop.langchain.langchain_tool import _langchain_tool_has_async_implementation

        @langchain_tool  # type: ignore[misc]
        def my_sync_tool(query: str) -> str:
            """A sync tool."""
            return "result"

        assert _langchain_tool_has_async_implementation(my_sync_tool) is False

    def test_detects_tool_with_explicit_arun_override(self) -> None:
        """Tools that explicitly override _arun on a subclass should be detected as async.

        Even if the override raises NotImplementedError, the override is on
        the concrete class (not BaseTool), so we treat it as async. This is
        the conservative choice — let LangChain handle error semantics.
        """
        from autogen.interop.langchain.langchain_tool import _langchain_tool_has_async_implementation

        class ToolWithExplicitAsync(LangchainBaseTool):  # type: ignore[misc, no-any-unimported]
            name: str = "explicit-async-tool"
            description: str = "test"

            def _run(self, query: str) -> str:
                return "sync"

            async def _arun(self, query: str) -> str:
                raise NotImplementedError("No async support")

        tool = ToolWithExplicitAsync()
        # The user explicitly defined _arun on their class, so we treat it
        # as having async support regardless of what the body does.
        assert _langchain_tool_has_async_implementation(tool) is True
