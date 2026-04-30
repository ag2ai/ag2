# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.anthropic.mappers import extract_skills_for_container, tool_to_api
from autogen.beta.context import ConversationContext
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools import Skill, SkillsTool, WebSearchTool


@pytest.mark.asyncio
async def test_extract_skills_strings(context: ConversationContext) -> None:
    t = SkillsTool("pptx", "xlsx")
    [schema] = await t.schemas(context)

    result = extract_skills_for_container([schema])

    assert result == [
        {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
        {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
    ]


@pytest.mark.asyncio
async def test_extract_skills_with_version(context: ConversationContext) -> None:
    t = SkillsTool(Skill("pptx", version="20251013"), Skill("xlsx", version="latest"))
    [schema] = await t.schemas(context)

    result = extract_skills_for_container([schema])

    assert result == [
        {"type": "anthropic", "skill_id": "pptx", "version": "20251013"},
        {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
    ]


@pytest.mark.asyncio
async def test_extract_skills_no_skills_schema(context: ConversationContext) -> None:
    """Filter must return an empty list for non-Skills schemas.

    Guards against the filter accidentally capturing unrelated tools
    (e.g. WebSearch) when new tool types are added to the codebase.
    """
    ws = WebSearchTool()
    [ws_schema] = await ws.schemas(context)

    result = extract_skills_for_container([ws_schema])

    assert result == []


@pytest.mark.asyncio
async def test_tool_to_api_raises_for_skills_schema(context: ConversationContext) -> None:
    """SkillsToolSchema must NOT reach tool_to_api.

    Skills are sent via the `container.skills` request field, not the
    `tools[]` array. The client filters SkillsToolSchema out before calling
    tool_to_api; this test ensures tool_to_api fails loudly if the filter
    ever regresses, instead of silently producing a broken API request.
    """
    t = SkillsTool("pptx")
    [schema] = await t.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)
