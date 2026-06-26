# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auto-injected ``knowledge`` tool.

When an Agent is configured with a ``KnowledgeConfig`` carrying a store,
a ``knowledge`` tool with read/write/list/delete actions is auto-attached.
These tests exercise each action of that tool.
"""

import pytest

from ag2 import Agent
from ag2.agent import KnowledgeConfig
from ag2.events import EventLogFailed, ModelMessage, ModelResponse
from ag2.knowledge import LOG_PREFIX, DefaultBootstrap, MemoryKnowledgeStore
from ag2.stream import MemoryStream
from ag2.testing import TestConfig


def _knowledge_tool_call(agent: Agent):
    """Extract the underlying async function from the auto-injected tool.

    Auto-injected tools (the knowledge tool) live in ``_additional_tools``,
    not the public ``agent.tools`` (which holds only user-supplied tools).
    """
    return agent._additional_tools[0].model.call


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


@pytest.mark.asyncio
class TestExposeToolFlag:
    """``expose_tool=False`` registers the store but withholds the LLM tool."""

    async def test_expose_tool_false_skips_auto_tool(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent(
            "policy-only",
            knowledge=KnowledgeConfig(store=store, expose_tool=False),
        )
        assert not agent._additional_tools

    async def test_expose_tool_true_is_default(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent("with-tool", knowledge=KnowledgeConfig(store=store))
        assert len(agent._additional_tools) == 1

    async def test_default_bootstrap_omits_tool_instruction_when_unexposed(self) -> None:
        """The root SKILL.md should not tell the model about a tool that does not exist."""
        store = MemoryKnowledgeStore()
        agent = Agent(
            "policy-only",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(store=store, expose_tool=False),
        )
        await agent.ask("hi")

        root_skill = await store.read("/SKILL.md")
        assert root_skill is not None
        assert "knowledge` tool" not in root_skill

    async def test_default_bootstrap_mentions_tool_when_exposed(self) -> None:
        store = MemoryKnowledgeStore()
        agent = Agent(
            "with-tool",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(store=store),
        )
        await agent.ask("hi")

        root_skill = await store.read("/SKILL.md")
        assert root_skill is not None
        assert "knowledge` tool" in root_skill


@pytest.mark.asyncio
class TestWriteEventLogFlag:
    """``write_event_log=False`` keeps ``/log/{stream}.jsonl`` from being written."""

    async def test_default_writes_event_log(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        agent = Agent(
            "logger",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(store=store),
        )
        await agent.ask("hi", stream=stream)

        path = f"{LOG_PREFIX}{stream.id}.jsonl"
        assert await store.read(path) is not None

    async def test_opt_out_skips_event_log(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        agent = Agent(
            "quiet",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(store=store, write_event_log=False),
        )
        await agent.ask("hi", stream=stream)

        path = f"{LOG_PREFIX}{stream.id}.jsonl"
        assert await store.read(path) is None


class _FailingStore(MemoryKnowledgeStore):
    """Wraps MemoryKnowledgeStore but raises on the final log persist.

    Boots and reads behave normally so the turn completes successfully; the
    failure only happens when ``EventLogWriter.persist`` writes to
    ``/log/{stream_id}.jsonl``.
    """

    def __init__(self) -> None:
        super().__init__()

    async def write(self, path: str, content: str) -> None:
        if path.startswith(LOG_PREFIX) and path.endswith(".jsonl"):
            raise OSError("disk on fire")
        await super().write(path, content)


@pytest.mark.asyncio
class TestEventLogFailedLifecycle:
    async def test_failure_emits_stream_event(self) -> None:
        store = _FailingStore()
        stream = MemoryStream()
        failures: list[EventLogFailed] = []
        stream.where(EventLogFailed).subscribe(lambda e: failures.append(e))

        agent = Agent(
            "log-fails",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(store=store),
        )
        # Turn must still succeed even if log persistence fails.
        await agent.ask("hi", stream=stream)

        assert len(failures) == 1
        assert failures[0].agent == "log-fails"
        assert failures[0].error_type == "OSError"
        assert "disk on fire" in failures[0].error


@pytest.mark.asyncio
class TestDefaultBootstrap:
    """The bootstrapper's SKILL.md content must match what the LLM can actually do."""

    async def test_mention_tool_true_includes_tool_instruction(self) -> None:
        store = MemoryKnowledgeStore()
        await DefaultBootstrap(mention_tool=True).bootstrap(store, "alice")

        root_skill = await store.read("/SKILL.md")
        assert root_skill is not None
        assert "Use the `knowledge` tool" in root_skill

    async def test_mention_tool_false_omits_tool_instruction(self) -> None:
        store = MemoryKnowledgeStore()
        await DefaultBootstrap(mention_tool=False).bootstrap(store, "alice")

        root_skill = await store.read("/SKILL.md")
        assert root_skill is not None
        assert "knowledge` tool" not in root_skill
        assert "no tool exposed" in root_skill
