# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.skills import LocalRuntime, SkillPlugin


def _write_skill(base: Path, name: str, *, script: bool = False, resource: bool = False) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill\n---\n# {name}\n")
    if script:
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("echo hi\n")
    if resource:
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "guide.md").write_text("guidance\n")
    return base


@pytest.mark.asyncio
class TestSkillPlugin:
    async def test_injects_catalog_into_prompt(self, skill_tree: Path) -> None:
        agent = Agent("a", config=TestConfig("done"), plugins=[SkillPlugin(skill_tree)])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        # XML catalog format (spec Step 3).
        assert "<available_skills>" in catalog
        assert "<name>react-best-practices</name>" in catalog
        assert "<description>Best practices for React development</description>" in catalog
        assert "<name>markdown-guide</name>" in catalog
        # Activation instruction + per-skill location (spec Step 3).
        assert "call load_skill(name)" in catalog
        location = skill_tree / "react-best-practices" / "SKILL.md"
        assert f"<location>{location}</location>" in catalog

    async def test_empty_skills_contributes_nothing(self, tmp_path: Path) -> None:
        # No skills → no catalog and no dead tools (spec Step 3).
        agent = Agent(
            "a",
            config=TestConfig(ToolCallEvent(name="load_skill"), "done"),
            plugins=[SkillPlugin(tmp_path / "empty")],
        )

        with pytest.raises(ToolNotFoundError, match="load_skill"):
            await agent.ask("hi")

    async def test_empty_skills_prompt_is_empty(self, tmp_path: Path) -> None:
        agent = Agent("a", config=TestConfig("done"), plugins=[SkillPlugin(tmp_path / "empty")])

        reply = await agent.ask("hi")

        assert reply.context.prompt == []

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

    async def test_gates_out_read_resource_when_no_skill_has_resources(self, tmp_path: Path) -> None:
        # A skill with a script but no resources: run_skill_script is registered,
        # read_skill_resource is not (no dead tool).
        _write_skill(tmp_path, "scripts-only", script=True)
        agent = Agent(
            "a",
            config=TestConfig(ToolCallEvent(name="read_skill_resource"), "done"),
            plugins=[SkillPlugin(tmp_path)],
        )

        with pytest.raises(ToolNotFoundError, match="read_skill_resource"):
            await agent.ask("hi")

    async def test_gates_out_run_script_when_no_skill_has_scripts(self, tmp_path: Path) -> None:
        # A skill with a resource but no scripts: read_skill_resource is
        # registered, run_skill_script is not.
        _write_skill(tmp_path, "resources-only", resource=True)
        agent = Agent(
            "a",
            config=TestConfig(ToolCallEvent(name="run_skill_script"), "done"),
            plugins=[SkillPlugin(tmp_path)],
        )

        with pytest.raises(ToolNotFoundError, match="run_skill_script"):
            await agent.ask("hi")

    async def test_last_runtime_shadows_earlier_on_name_clash(self, tmp_path: Path) -> None:
        # The same skill name in two runtimes resolves to the last runtime
        # (project overrides global) in both the catalog and load_skill.
        glob = tmp_path / "global"
        proj = tmp_path / "project"
        (glob / "dup").mkdir(parents=True)
        (glob / "dup" / "SKILL.md").write_text("---\nname: dup\ndescription: from global\n---\n# global body\n")
        (proj / "dup").mkdir(parents=True)
        (proj / "dup" / "SKILL.md").write_text("---\nname: dup\ndescription: from project\n---\n# project body\n")

        tracking = TrackingConfig(
            TestConfig(ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "dup"})), "done")
        )
        agent = Agent(
            "a",
            config=tracking,
            plugins=[SkillPlugin(LocalRuntime(dir=glob), LocalRuntime(dir=proj))],
        )

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "<description>from project</description>" in catalog
        assert "from global" not in catalog
        tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
        assert "project body" in tool_result_msg.results[0].result.parts[0].content
