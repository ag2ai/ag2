# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.context import ConversationContext
from autogen.beta.tools.local_skills.tool import LocalSkillsTool
from autogen.beta.tools.runtime import LocalRuntime

SKILL_MD = """\
---
name: my-skill
description: A test skill for unit tests.
---
# My Skill
Do something useful.
"""


def _make_skill_dir(base: Path, name: str = "my-skill", has_scripts: bool = False) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD)
    if has_scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "run.sh"
        script.write_text("#!/bin/sh\necho hello\n")
        script.chmod(0o755)
    return skill_dir


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_skills_schema(context: ConversationContext) -> None:
    """list_skills is parameterless — Gemini requires type=object, not null."""
    tool = LocalSkillsTool()

    [schema] = await tool.list_skills.schemas(context)
    [api_tool] = build_tools([schema])

    assert len(api_tool.function_declarations) == 1
    decl = api_tool.function_declarations[0]
    assert decl.name == "list_skills"
    assert decl.parameters.type == types.Type.OBJECT


# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_skill_schema(context: ConversationContext) -> None:
    tool = LocalSkillsTool()

    [schema] = await tool.load_skill.schemas(context)
    [api_tool] = build_tools([schema])

    decl = api_tool.function_declarations[0]
    assert decl.name == "load_skill"
    assert "name" in decl.parameters.properties
    assert "name" in (decl.parameters.required or [])


# ---------------------------------------------------------------------------
# run_skill_script
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_skill_script_schema(context: ConversationContext) -> None:
    tool = LocalSkillsTool()

    [schema] = await tool.run_skill_script.schemas(context)
    [api_tool] = build_tools([schema])

    decl = api_tool.function_declarations[0]
    assert decl.name == "run_skill_script"
    props = decl.parameters.properties
    assert "name" in props
    assert "script" in props
    # args is optional — must be present in properties but not required
    assert "args" in props
    required = decl.parameters.required or []
    assert "name" in required
    assert "script" in required
    assert "args" not in required


# ---------------------------------------------------------------------------
# Full toolset: all three tools build without errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_toolset_schemas_build(tmp_path: Path, context: ConversationContext) -> None:
    _make_skill_dir(tmp_path)
    tool = LocalSkillsTool(runtime=LocalRuntime(dir=tmp_path))

    schemas = []
    for attr in ("list_skills", "load_skill", "run_skill_script"):
        [s] = await getattr(tool, attr).schemas(context)
        schemas.append(s)

    [api_tool] = build_tools(schemas)

    names = {d.name for d in api_tool.function_declarations}
    assert names == {"list_skills", "load_skill", "run_skill_script"}


# ---------------------------------------------------------------------------
# list_skills returns skill data from the filesystem
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_skills_returns_installed_skills(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path)
    tool = LocalSkillsTool(runtime=LocalRuntime(dir=tmp_path))

    result = await tool.list_skills.model.call()

    assert result == [{"name": "my-skill", "description": "A test skill for unit tests."}]


# ---------------------------------------------------------------------------
# load_skill reads SKILL.md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_skill_reads_skill_md(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path)
    tool = LocalSkillsTool(runtime=LocalRuntime(dir=tmp_path))

    content = await tool.load_skill.model.call(name="my-skill")

    assert "My Skill" in content
    assert "Do something useful." in content


# ---------------------------------------------------------------------------
# run_skill_script executes the script
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_skill_script_executes(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, has_scripts=True)
    tool = LocalSkillsTool(runtime=LocalRuntime(dir=tmp_path))

    output = await tool.run_skill_script.model.call(name="my-skill", script="run.sh")

    assert "hello" in output
