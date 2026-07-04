# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.openai.mappers import (
    extract_skills_for_shell,
    merge_skills_into_shell_tools,
)
from ag2.tools.builtin.skills import Skill, SkillsTool


@pytest.mark.asyncio
async def test_extract_skill_references(context: Context) -> None:
    tool = SkillsTool("skill_abc", Skill("skill_def", version="2"), Skill("skill_ghi", version="latest"))

    schemas = await tool.schemas(context)

    assert extract_skills_for_shell(schemas) == [
        {"type": "skill_reference", "skill_id": "skill_abc"},
        {"type": "skill_reference", "skill_id": "skill_def", "version": 2},
        # non-numeric version ("latest") is omitted — OpenAI treats absent as latest
        {"type": "skill_reference", "skill_id": "skill_ghi"},
    ]


def test_merge_into_existing_shell_tool() -> None:
    tools = [{"type": "shell"}]
    skills = [{"type": "skill_reference", "skill_id": "skill_abc"}]

    merged = merge_skills_into_shell_tools(tools, skills)

    assert merged == [
        {"type": "shell", "environment": {"type": "container_auto", "skills": skills}},
    ]


def test_merge_preserves_container_auto_options() -> None:
    tools = [
        {"type": "web_search"},
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
            },
        },
    ]
    skills = [{"type": "skill_reference", "skill_id": "skill_abc"}]

    merged = merge_skills_into_shell_tools(tools, skills)

    assert merged[1]["environment"] == {
        "type": "container_auto",
        "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
        "skills": skills,
    }


def test_merge_appends_shell_when_absent() -> None:
    tools = [{"type": "web_search"}]
    skills = [{"type": "skill_reference", "skill_id": "skill_abc"}]

    merged = merge_skills_into_shell_tools(tools, skills)

    assert merged == [
        {"type": "web_search"},
        {"type": "shell", "environment": {"type": "container_auto", "skills": skills}},
    ]


def test_merge_rejects_container_reference() -> None:
    tools = [{"type": "shell", "environment": {"type": "container_reference", "container_id": "cont_1"}}]
    skills = [{"type": "skill_reference", "skill_id": "skill_abc"}]

    with pytest.raises(ValueError, match="container"):
        merge_skills_into_shell_tools(tools, skills)


def test_merge_noop_without_skills() -> None:
    tools = [{"type": "shell"}]

    assert merge_skills_into_shell_tools(tools, []) == [{"type": "shell"}]
