# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auto-injected ``knowledge`` tool.

When an Agent is configured with a ``KnowledgeConfig`` carrying a store,
a ``knowledge`` tool with read/write/list/delete actions is auto-attached.
These tests exercise each action of that tool.
"""

import pytest

from autogen.beta import Agent
from autogen.beta.agent import KnowledgeConfig
from autogen.beta.knowledge import MemoryKnowledgeStore


def _knowledge_tool_call(agent: Agent):
    """Extract the underlying async function from the auto-injected tool."""
    return agent._build_knowledge_tool()[0].model.call


@pytest.mark.asyncio
class TestKnowledgeTool:
    async def test_read_returns_content(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello world")
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="read", path="/test.txt")
        assert result == "hello world"

    async def test_read_missing_path_reports_not_found(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="read", path="/missing.txt")
        assert "Not found" in result

    async def test_write_persists_content(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="write", path="/note.txt", content="my note")
        assert "Written" in result
        assert await store.read("/note.txt") == "my note"

    async def test_write_without_content_reports_error(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="write", path="/note.txt")
        assert "Error" in result

    async def test_list_includes_skill_md_and_entries(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/dir/SKILL.md", "This directory stores artifacts.")
        await store.write("/dir/file1.txt", "data")
        await store.write("/dir/file2.txt", "data")
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="list", path="/dir/")
        assert "This directory stores artifacts." in result
        assert "file1.txt" in result
        assert "file2.txt" in result

    async def test_list_empty_directory(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="list", path="/empty/")
        assert "Empty" in result

    async def test_delete_removes_path(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "data")
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="delete", path="/test.txt")
        assert "Deleted" in result
        assert await store.read("/test.txt") is None

    async def test_unknown_action_reports_error(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("test", knowledge=KnowledgeConfig(store=store))

        result = await _knowledge_tool_call(agent)(action="bogus")
        assert "Unknown action" in result
