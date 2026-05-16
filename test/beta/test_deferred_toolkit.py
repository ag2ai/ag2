# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeferredToolkit."""

import json
from unittest.mock import AsyncMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools import DeferredToolkit, MemoryToolkit
from autogen.beta.tools.final import tool
from autogen.beta.tools.toolkits.deferred import _CatalogEntry

# ---------------------------------------------------------------------------
# Helpers — simple catalog tools for tests
# ---------------------------------------------------------------------------


@tool(name="add", description="Add two integers and return the sum.")
def _add(a: int, b: int) -> int:
    return a + b


@tool(name="greet", description="Return a greeting for a given name.")
def _greet(name: str) -> str:
    return f"Hello, {name}!"


@tool(name="ping", description="Return pong. No parameters needed.")
def _ping() -> str:
    return "pong"


# ---------------------------------------------------------------------------
# _CatalogEntry
# ---------------------------------------------------------------------------


def test_catalog_entry_metadata() -> None:
    entry = _CatalogEntry(_add)
    assert entry.name == "add"
    assert "Add two integers" in entry.description


def test_catalog_entry_summary() -> None:
    entry = _CatalogEntry(_greet)
    assert "[greet]" in entry.summary()
    assert "greeting" in entry.summary()


def test_catalog_entry_full_description_contains_schema() -> None:
    desc = _CatalogEntry(_add).full_description()
    assert "add" in desc
    assert "parameters" in desc.lower()


@pytest.mark.asyncio
async def test_catalog_entry_invoke_sync() -> None:
    entry = _CatalogEntry(_add)
    result = await entry.invoke(a=3, b=4)
    assert result == "7"


@pytest.mark.asyncio
async def test_catalog_entry_invoke_no_args() -> None:
    entry = _CatalogEntry(_ping)
    result = await entry.invoke()
    assert result == "pong"


# ---------------------------------------------------------------------------
# DeferredToolkit construction
# ---------------------------------------------------------------------------


def test_only_two_tools_exposed() -> None:
    dt = DeferredToolkit(_add, _greet, _ping)
    assert set(dt._tools.keys()) == {"search_tools", "use_tool"}


def test_catalog_names() -> None:
    dt = DeferredToolkit(_add, _greet, _ping)
    assert dt.catalog_names == ("add", "greet", "ping")


def test_toolkit_name() -> None:
    assert DeferredToolkit().name == "deferred_toolkit"


def test_empty_catalog() -> None:
    dt = DeferredToolkit()
    assert dt.catalog_names == ()


def test_add_to_catalog() -> None:
    dt = DeferredToolkit(_add)
    assert "greet" not in dt.catalog_names
    dt.add_to_catalog(_greet)
    assert "greet" in dt.catalog_names


def test_from_toolkit_tools() -> None:
    """Accept tools from another toolkit's .tools property."""
    memory = MemoryToolkit()
    dt = DeferredToolkit(*memory.tools)
    assert set(dt.catalog_names) == {"remember", "recall", "forget", "list_memories"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schemas(async_mock: AsyncMock) -> None:
    dt = DeferredToolkit(_add, _greet)
    schemas = list(await dt.schemas(Context(async_mock)))
    names = {s.function.name for s in schemas}
    assert names == {"search_tools", "use_tool"}


# ---------------------------------------------------------------------------
# search_tools via agent call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_finds_matching_tool() -> None:
    dt = DeferredToolkit(_add, _greet, _ping)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="search_tools",
                arguments=json.dumps({"query": "greeting", "max_results": 5}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("find tools")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "greet" in result_text
    assert "greeting" in result_text


@pytest.mark.asyncio
async def test_search_no_match_lists_all() -> None:
    dt = DeferredToolkit(_add, _ping)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="search_tools",
                arguments=json.dumps({"query": "zucchini"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("find tools")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "No tools matched" in result_text
    assert "add" in result_text
    assert "ping" in result_text


# ---------------------------------------------------------------------------
# use_tool via agent call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_tool_invokes_sync_function() -> None:
    dt = DeferredToolkit(_add)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "add", "arguments": '{"a": 10, "b": 32}'}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("add numbers")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "42" in result_text


@pytest.mark.asyncio
async def test_use_tool_no_args() -> None:
    dt = DeferredToolkit(_ping)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "ping"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("ping")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "pong" in result_text


@pytest.mark.asyncio
async def test_use_tool_unknown_name() -> None:
    dt = DeferredToolkit(_ping)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "ghost_tool"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("use unknown tool")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "not found" in result_text.lower()


@pytest.mark.asyncio
async def test_use_tool_bad_json_arguments() -> None:
    dt = DeferredToolkit(_add)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "add", "arguments": "not json {{{"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("use with bad args")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "Invalid JSON" in result_text


@pytest.mark.asyncio
async def test_use_tool_wrong_arguments() -> None:
    dt = DeferredToolkit(_add)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "add", "arguments": '{"x": 1, "y": 2}'}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("use with wrong args")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "Wrong arguments" in result_text or "error" in result_text.lower()


@pytest.mark.asyncio
async def test_use_memory_toolkit_tool_via_deferred() -> None:
    """End-to-end: use MemoryToolkit tools through DeferredToolkit."""
    memory = MemoryToolkit()
    dt = DeferredToolkit(*memory.tools)
    memory._store.store("sky", "The sky is blue.")

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="use_tool",
                arguments=json.dumps({"name": "recall", "arguments": '{"query": "sky"}'}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[dt])
    await agent.ask("recall sky memory")

    result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_text = result_msg.results[0].result.parts[0].content
    assert "sky" in result_text
    assert "The sky is blue." in result_text
