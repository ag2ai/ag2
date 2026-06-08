# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.events import ToolCallEvent, ToolResultsEvent
from autogen.beta.exceptions import ToolNotFoundError
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.skills import SkillPlugin


@pytest.mark.asyncio
class TestSkillPlugin:
    async def test_injects_catalog_into_prompt(self, skill_tree: Path) -> None:
        agent = Agent("a", config=TestConfig("done"), plugins=[SkillPlugin(skill_tree)])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "- react-best-practices: Best practices for React development" in catalog
        assert "- markdown-guide: Guide for writing Markdown" in catalog

    async def test_empty_skills_prompt(self, tmp_path: Path) -> None:
        agent = Agent("a", config=TestConfig("done"), plugins=[SkillPlugin(tmp_path / "empty")])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "no local skills" in catalog.lower()

    async def test_does_not_expose_list_skills_tool(self, skill_tree: Path) -> None:
        # The spec puts the catalog in the prompt, so the plugin registers no
        # list_skills tool — invoking it raises ToolNotFoundError.
        agent = Agent(
            "a",
            config=TestConfig(ToolCallEvent(name="list_skills"), "done"),
            plugins=[SkillPlugin(skill_tree)],
        )

        with pytest.raises(ToolNotFoundError, match="list_skills"):
            await agent.ask("hi")

    async def test_load_skill_tool_runs(self, skill_tree: Path) -> None:
        tracking = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "react-best-practices"})),
                "done",
            )
        )
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill_tree)])

        await agent.ask("hi")

        # Second LLM call receives the tool result; verify SKILL.md was loaded.
        tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
        assert tool_result_msg.results[0].name == "load_skill"
        assert "React Best Practices" in tool_result_msg.results[0].result.parts[0].content

    async def test_run_skill_script_tool_runs(self, skill_tree: Path) -> None:
        tracking = TrackingConfig(
            TestConfig(
                ToolCallEvent(
                    name="run_skill_script",
                    arguments=json.dumps({"name": "react-best-practices", "script": "scaffold.py"}),
                ),
                "done",
            )
        )
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill_tree)])

        await agent.ask("hi")

        # Second LLM call receives the tool result; verify the script ran.
        tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
        assert "scaffold" in tool_result_msg.results[0].result.parts[0].content
