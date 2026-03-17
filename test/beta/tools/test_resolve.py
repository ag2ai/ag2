# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta.annotations import Variable
from autogen.beta.context import Context
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchTool


def _make_context(**variables: object) -> Context:
    return Context(stream=MagicMock(), variables=variables)


# --- resolve_variable ---


def test_resolve_variable_passthrough() -> None:
    ctx = _make_context()
    assert resolve_variable("hello", ctx) == "hello"
    assert resolve_variable(42, ctx) == 42
    assert resolve_variable(None, ctx) is None


def test_resolve_variable_from_context() -> None:
    loc = UserLocation(country="US")
    ctx = _make_context(user_location=loc)

    result = resolve_variable(Variable("user_location"), ctx)

    assert result is loc


def test_resolve_variable_default() -> None:
    ctx = _make_context()
    fallback = UserLocation(country="DE")

    result = resolve_variable(Variable("user_location", default=fallback), ctx)

    assert result is fallback


def test_resolve_variable_default_factory() -> None:
    ctx = _make_context()

    result = resolve_variable(Variable("counter", default_factory=dict), ctx)

    assert result == {}


def test_resolve_variable_context_takes_precedence_over_default() -> None:
    ctx = _make_context(mode="fast")

    result = resolve_variable(Variable("mode", default="slow"), ctx)

    assert result == "fast"


def test_resolve_variable_missing_raises() -> None:
    ctx = _make_context()

    with pytest.raises(KeyError, match="user_location"):
        resolve_variable(Variable("user_location"), ctx)


# --- WebSearchTool.schemas() with Variable ---


@pytest.mark.asyncio
async def test_web_search_tool_static_values() -> None:
    tool = WebSearchTool(search_context_size="high", max_uses=5)
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert schema.search_context_size == "high"
    assert schema.max_uses == 5
    assert schema.user_location is None


@pytest.mark.asyncio
async def test_web_search_tool_variable_resolved_with_default_name() -> None:
    loc = UserLocation(city="Berlin", country="DE")
    tool = WebSearchTool(
        search_context_size="high",
        user_location=Variable(),
    )
    ctx = _make_context(user_location=loc)

    [schema] = await tool.schemas(ctx)

    assert schema.user_location is loc
    assert schema.search_context_size == "high"


@pytest.mark.asyncio
async def test_web_search_tool_variable_resolved() -> None:
    loc = UserLocation(city="Berlin", country="DE")
    tool = WebSearchTool(
        search_context_size="high",
        user_location=Variable("user_location"),
    )
    ctx = _make_context(user_location=loc)

    [schema] = await tool.schemas(ctx)

    assert schema.user_location is loc
    assert schema.search_context_size == "high"


@pytest.mark.asyncio
async def test_web_search_tool_variable_with_default() -> None:
    fallback = UserLocation(country="US")
    tool = WebSearchTool(user_location=Variable("user_location", default=fallback))
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert schema.user_location is fallback


@pytest.mark.asyncio
async def test_web_search_tool_variable_missing_raises() -> None:
    tool = WebSearchTool(user_location=Variable("user_location"))
    ctx = _make_context()

    with pytest.raises(KeyError, match="user_location"):
        await tool.schemas(ctx)
