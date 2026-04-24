# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from autogen.beta.context import ConversationContext
from autogen.beta.tools.builtin.skills import Skill, SkillsTool, SkillsToolSchema


@pytest.mark.asyncio
async def test_strings_become_skill_objects(context: ConversationContext) -> None:
    t = SkillsTool("pptx", "xlsx")

    [schema] = await t.schemas(context)

    assert isinstance(schema, SkillsToolSchema)
    assert schema.type == "skills"
    assert schema.skills == [Skill(id="pptx"), Skill(id="xlsx")]


@pytest.mark.asyncio
async def test_skill_objects_preserved(context: ConversationContext) -> None:
    t = SkillsTool(Skill("openai-spreadsheets"), Skill("skill_abc123", version=2))

    [schema] = await t.schemas(context)

    assert schema.skills == [
        Skill(id="openai-spreadsheets", version=None),
        Skill(id="skill_abc123", version=2),
    ]


@pytest.mark.asyncio
async def test_mixed_strings_and_skill_objects(context: ConversationContext) -> None:
    t = SkillsTool("pptx", Skill("xlsx", version="20251013"))

    [schema] = await t.schemas(context)

    assert schema.skills == [
        Skill(id="pptx", version=None),
        Skill(id="xlsx", version="20251013"),
    ]


@pytest.mark.asyncio
async def test_no_args_produces_empty_skills(context: ConversationContext) -> None:
    t = SkillsTool()

    [schema] = await t.schemas(context)

    assert schema.skills == []


@pytest.mark.asyncio
async def test_register_is_noop(context: ConversationContext) -> None:
    t = SkillsTool("pptx")

    with ExitStack() as stack:
        t.register(stack, context)  # must not raise
