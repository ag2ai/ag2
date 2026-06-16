# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.a2ui import (
    A2UIAgent,
    A2UIEventAction,
    A2UIFunctionCallAction,
    a2ui_action,
)
from autogen.beta.a2ui.action_tool import A2UIActionTool
from autogen.beta.a2ui.actions import A2UIAction
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
        assert isinstance(action, A2UIAction)
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


class TestAgentActionCollection:
    def test_collects_action_from_decorated_tool(self) -> None:
        @a2ui_action(description="Schedule posts")
        def schedule(time: str) -> str:
            return time

        agent = A2UIAgent(name="ui", tools=[schedule])

        collected = agent.get_action("schedule")
        assert collected is not None
        assert collected.action_type == "event"
        assert collected.tool_name == "schedule"

        prompt = "\n".join(agent._system_prompt)
        assert "Server Events" in prompt
        assert "schedule" in prompt

    def test_collects_bare_action(self) -> None:
        action = A2UIEventAction(name="rewrite", description="Regenerate previews")
        agent = A2UIAgent(name="ui", tools=[action])

        assert agent.get_action("rewrite") is action

    def test_collects_function_call_declaration(self) -> None:
        action = A2UIFunctionCallAction(
            name="openUrl",
            description="Open a URL",
            example_args={"url": "https://example.com"},
        )
        agent = A2UIAgent(name="ui", tools=[action])

        prompt = "\n".join(agent._system_prompt)
        assert "Client Functions" in prompt
        assert "openUrl" in prompt

    def test_mixed_tools_and_actions(self) -> None:
        def web_search(query: str) -> str:
            return query

        @a2ui_action
        def schedule(time: str) -> str:
            return time

        bare = A2UIEventAction(name="rewrite", description="Regenerate")

        agent = A2UIAgent(name="ui", tools=[web_search, schedule, bare])

        assert {a.name for a in agent.actions} == {"schedule", "rewrite"}
        assert agent.get_action("schedule") is not None
        assert agent.get_action("rewrite") is not None

    def test_duplicate_action_name_raises(self) -> None:
        @a2ui_action(name="dup")
        def first(x: str) -> str:
            return x

        clash = A2UIEventAction(name="dup", description="clash")

        with pytest.raises(ValueError, match="Duplicate A2UI action name 'dup'"):
            A2UIAgent(name="ui", tools=[first, clash])

    def test_bare_action_is_not_forwarded_as_executable_tool(self) -> None:
        # A bare A2UIAction is not a Tool; if it were forwarded to the base
        # Agent it would fail to register. Successful construction proves it is
        # filtered out of the executable tool list.
        agent = A2UIAgent(name="ui", tools=[A2UIEventAction(name="rewrite")])
        assert agent.get_action("rewrite") is not None


@pytest.mark.asyncio
class TestActionToolExecution:
    async def test_decorated_tool_is_callable_by_agent(self) -> None:
        calls: list[str] = []

        @a2ui_action(description="Schedule posts")
        def schedule(time: str) -> str:
            calls.append(time)
            return f"scheduled {time}"

        agent = A2UIAgent(
            name="ui",
            validate_responses=False,
            tools=[schedule],
            config=TestConfig(
                ToolCallEvent(name="schedule", arguments='{"time": "2pm"}'),
                "done",
            ),
        )

        await agent.ask("schedule for 2pm")

        assert calls == ["2pm"]
