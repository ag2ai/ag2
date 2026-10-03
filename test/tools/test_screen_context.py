# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Boundaries the screen-context example promises: local-only history, explicit-request retrieval, untrusted OCR."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from examples.tools.screen_context import SCREEN_PROMPT, ScreenContextStore, build_agent, build_screen_history_tool

HISTORY = "screen_history.jsonl"


@pytest.fixture()
def store(tmp_path: Path) -> ScreenContextStore:
    return ScreenContextStore(tmp_path / HISTORY)


def test_record_appends_a_timestamped_local_line(store: ScreenContextStore, tmp_path: Path) -> None:
    store.record(app="Google Chrome", bundle="com.google.Chrome", text="TypeError: unsupported operand")
    store.record(app="Xcode", bundle="com.apple.dt.Xcode", text="build succeeded")

    lines = (tmp_path / HISTORY).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    first = json.loads(lines[0])
    assert set(first) == {"timestamp", "app", "bundle", "text"}
    assert first["app"] == "Google Chrome"
    assert first["bundle"] == "com.google.Chrome"
    assert first["text"] == "TypeError: unsupported operand"
    assert datetime.fromisoformat(first["timestamp"]).tzinfo is not None


def test_constructing_the_store_creates_nothing(tmp_path: Path) -> None:
    ScreenContextStore(tmp_path / "nested" / HISTORY)

    assert not (tmp_path / "nested").exists()


def test_search_matches_text_case_insensitively_newest_first(store: ScreenContextStore) -> None:
    store.record(app="Google Chrome", bundle="com.google.Chrome", text="TypeError: unsupported operand")
    store.record(app="Terminal", bundle="com.apple.Terminal", text="tyPeError in the build log")

    assert [entry.text for entry in store.search("typeerror")] == [
        "tyPeError in the build log",
        "TypeError: unsupported operand",
    ]

    assert store.search("nothing like this") == []


def test_search_ignores_a_blank_query(store: ScreenContextStore) -> None:
    store.record(app="Google Chrome", bundle="com.google.Chrome", text="TypeError")

    assert store.search("   ") == []


@pytest.mark.asyncio
async def test_result_reaches_the_model_labelled_untrusted(store: ScreenContextStore) -> None:
    store.record(app="Google Chrome", bundle="com.google.Chrome", text="TypeError: unsupported operand type(s) for +=")
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="screen_history", arguments=json.dumps({"query": "TypeError"})),
            "You saw a TypeError in Google Chrome, at the timestamp above.",
        )
    )

    reply = await build_agent(store, config=config).ask("What error message was I looking at a moment ago?")

    assert reply.body == "You saw a TypeError in Google Chrome, at the timestamp above."
    tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    [tool_result] = tool_results_event.results
    seen_by_llm = tool_result.result.parts[0].content
    assert "trust=untrusted" in seen_by_llm
    assert "Google Chrome" in seen_by_llm
    assert "TypeError: unsupported operand type(s) for +=" in seen_by_llm
    assert tool_result.result.metadata == {"source": "observed_screen", "trust": "untrusted"}


@pytest.mark.asyncio
async def test_missing_history_returns_a_clear_miss(store: ScreenContextStore) -> None:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="screen_history", arguments=json.dumps({"query": "deploy token"})),
            "There is nothing in your screen history about that.",
        )
    )

    reply = await build_agent(store, config=config).ask("What was the deploy token I saw earlier?")

    assert reply.body == "There is nothing in your screen history about that."
    tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    [tool_result] = tool_results_event.results
    assert "No entries in the local screen history" in tool_result.result.parts[0].content


@pytest.mark.asyncio
async def test_only_retrieval_is_exposed_with_its_boundaries(store: ScreenContextStore) -> None:
    screen_history = build_screen_history_tool(store)
    agent = build_agent(store, config=TestConfig("no tool call needed"))

    assert [tool.name for tool in agent.tools] == [screen_history.name] == ["screen_history"]
    description = screen_history.schema.function.description
    assert "explicitly asks" in description
    assert "untrusted" in description
    assert "explicitly asks" in SCREEN_PROMPT
