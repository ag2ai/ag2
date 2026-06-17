# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.a2ui import a2ui_action
from autogen.beta.a2ui._runtime import _A2UIRuntime
from autogen.beta.a2ui.action_tool import A2UIActionTool
from autogen.beta.a2ui.actions import A2UIEventAction
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


class TestDecorator:
    def test_bare_decorator_produces_action_tool(self) -> None:
        @a2ui_action
        def schedule_posts(time: str) -> str:
            """Schedule all posts."""
            return f"scheduled {time}"

        assert isinstance(schedule_posts, A2UIActionTool)
        assert schedule_posts.name == "schedule_posts"
        action = schedule_posts.action
        assert isinstance(action, A2UIEventAction)
        assert action.name == "schedule_posts"
        assert action.action_type == "event"
        assert action.tool_name == "schedule_posts"
        assert action.description == "Schedule all posts."

    def test_call_with_overrides(self) -> None:
        @a2ui_action(name="book", description="Book a table")
        def book_restaurant(restaurant_id: str) -> str:
            return restaurant_id

        assert book_restaurant.name == "book"
        assert book_restaurant.action.name == "book"
        assert book_restaurant.action.tool_name == "book"
        assert book_restaurant.action.description == "Book a table"

    def test_example_context_auto_derived_from_schema(self) -> None:
        @a2ui_action
        def mixed(name: str, count: int, ok: bool) -> str:
            return name

        assert mixed.action.example_context == {
            "name": "<string>",
            "count": "<integer>",
            "ok": "<boolean>",
        }

    def test_example_context_handles_optional_and_containers(self) -> None:
        @a2ui_action
        def complex_args(maybe: str | None, tags: list[str], meta: dict[str, str]) -> str:
            return maybe or ""

        # Optional[str] (anyOf with null) resolves to the non-null branch;
        # containers get fresh empty placeholders.
        assert complex_args.action.example_context == {
            "maybe": "<string>",
            "tags": [],
            "meta": {},
        }

    def test_example_context_override_wins(self) -> None:
        @a2ui_action(example_context={"time": "2:00 PM"})
        def schedule(time: str) -> str:
            return time

        assert schedule.action.example_context == {"time": "2:00 PM"}

    def test_no_params_yields_empty_example_context(self) -> None:
        @a2ui_action
        def refresh() -> str:
            return "ok"

        assert refresh.action.example_context == {}


class TestRuntimeActionCollection:
    def test_collects_action_from_decorated_tool(self) -> None:
        @a2ui_action(description="Schedule posts")
        def schedule(time: str) -> str:
            return time

        rt = _A2UIRuntime(Agent(name="ui", tools=[schedule]))

        collected = rt.get_action("schedule")
        assert collected is not None
        assert collected.action_type == "event"
        assert collected.tool_name == "schedule"

        prompt = rt.system_prompt_section
        assert "Server Events" in prompt
        assert "schedule" in prompt

    def test_only_action_tools_are_collected(self) -> None:
        def web_search(query: str) -> str:
            return query

        @a2ui_action
        def schedule(time: str) -> str:
            return time

        # Plain tools (web_search) are not clickable buttons; only @a2ui_action
        # tools contribute an A2UIEventAction.
        rt = _A2UIRuntime(Agent(name="ui", tools=[web_search, schedule]))

        assert {a.name for a in rt.actions} == {"schedule"}
        assert rt.get_action("schedule") is not None
        assert rt.get_action("web_search") is None


@pytest.mark.asyncio
async def test_decorated_tool_is_callable_by_agent() -> None:
    calls: list[str] = []

    @a2ui_action(description="Schedule posts")
    def schedule(time: str) -> str:
        calls.append(time)
        return f"scheduled {time}"

    agent = Agent(
        name="ui",
        tools=[schedule],
        config=TestConfig(
            ToolCallEvent(name="schedule", arguments='{"time": "2pm"}'),
            "done",
        ),
    )

    await agent.ask("schedule for 2pm")

    assert calls == ["2pm"]
