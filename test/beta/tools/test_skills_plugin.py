# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.testing import TestConfig
from autogen.beta.tools.skills import LocalRuntime, SkillsPlugin


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    (alpha / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: alpha
            description: Alpha skill
            ---
            # Alpha
            Alpha instructions.
        """),
        encoding="utf-8",
    )

    beta = tmp_path / "beta"
    beta.mkdir()
    (beta / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: beta
            description: Beta skill
            ---
            # Beta
            Beta instructions.
        """),
        encoding="utf-8",
    )

    return tmp_path


class TestSkillsPluginUnit:
    def test_prompt_contains_loaded_skills(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin("alpha", runtime=skill_dir)
        assert "## Skill: alpha" in plugin._system_prompt[0]
        assert "Alpha instructions." in plugin._system_prompt[0]

    def test_prompt_contains_multiple_skills(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin("alpha", "beta", runtime=skill_dir)
        assert "## Skill: alpha" in plugin._system_prompt[0]
        assert "## Skill: beta" in plugin._system_prompt[0]

    def test_no_skills_produces_empty_prompt(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin(runtime=skill_dir)
        # No skill names → no system prompt entries, toolkit still added
        assert plugin._system_prompt == [] or plugin._system_prompt == [""]

    def test_tools_include_skills_toolkit(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin("alpha", runtime=skill_dir)
        tool_names = {t.name for t in plugin._tools}
        assert any("list" in n or "skill" in n for n in tool_names)

    def test_accepts_runtime_instance(self, skill_dir: Path) -> None:
        runtime = LocalRuntime(dir=skill_dir)
        plugin = SkillsPlugin("alpha", runtime=runtime)
        assert "Alpha instructions." in plugin._system_prompt[0]

    def test_accepts_path_string(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin("alpha", runtime=str(skill_dir))
        assert "Alpha instructions." in plugin._system_prompt[0]

    def test_registers_system_prompt_on_agent(self, skill_dir: Path) -> None:
        plugin = SkillsPlugin("alpha", runtime=skill_dir)
        agent = Agent("agent", config=TestConfig("done"), plugins=[plugin])
        assert any("alpha" in p.lower() for p in agent._system_prompt)
