# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""KnowledgePlugin actions, exposure, bootstrap, and logging through Agent."""

import json

import pytest

from ag2 import Agent
from ag2.events import EventLogFailed, ModelMessage, ModelResponse, ToolCallEvent, ToolResultsEvent
from ag2.knowledge import LOG_PREFIX, DefaultBootstrap, KnowledgePlugin, KnowledgeStore, MemoryKnowledgeStore
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig


async def knowledge_result(store: MemoryKnowledgeStore, **arguments: str) -> str:
    config = TrackingConfig(TestConfig(ToolCallEvent(name="knowledge", arguments=json.dumps(arguments)), "done"))
    reply = await Agent("test", config=config, plugins=[KnowledgePlugin(store)]).ask("Use knowledge")
    results = [e for e in await reply.history.get_events() if isinstance(e, ToolResultsEvent)]
    return results[0].results[0].result.parts[0].content


@pytest.mark.asyncio
class TestKnowledgeTool:
    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ({"action": "read", "path": "/test.txt"}, "hello world"),
            ({"action": "read", "path": "/missing"}, "Not found: /missing"),
            ({"action": "write", "path": "/note", "content": "my note"}, "Written to /note"),
            ({"action": "write", "path": "/note"}, "Error: content is required for write action."),
            ({"action": "list", "path": "/empty/"}, "Empty: /empty/"),
            ({"action": "delete", "path": "/test.txt"}, "Deleted: /test.txt"),
            ({"action": "bogus"}, "Unknown action: bogus. Available: read, write, list, delete."),
        ],
    )
    async def test_actions(self, arguments: dict[str, str], expected: str) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello world")
        assert await knowledge_result(store, **arguments) == expected
        if arguments["action"] == "delete":
            assert await store.read("/test.txt") is None
        if arguments.get("content"):
            assert await store.read("/note") == "my note"

    async def test_list_includes_skill_md_and_entries(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/dir/SKILL.md", "This directory stores artifacts.")
        await store.write("/dir/file1.txt", "data")
        result = await knowledge_result(store, action="list", path="/dir/")
        assert result == "This directory stores artifacts.\n---\nSKILL.md\nfile1.txt"


@pytest.mark.asyncio
class TestExposeToolFlag:
    """``expose_tool=False`` registers the store but withholds the LLM tool."""

    @pytest.mark.parametrize("expose", [True, False])
    async def test_tool_exposure_and_store_dependency(self, expose: bool) -> None:
        store = MemoryKnowledgeStore()
        config = TrackingConfig(TestConfig("ok"))
        agent = Agent("test", config=config, plugins=[KnowledgePlugin(store, expose_tool=expose)])
        await agent.ask("hi")
        assert [s.function.name for s in config.calls[0].tools] == (["knowledge"] if expose else [])
        assert config.calls[0].dependencies[KnowledgeStore] is store

    async def test_default_bootstrap_omits_tool_instruction_when_unexposed(self) -> None:
        """The root SKILL.md should not tell the model about a tool that does not exist."""
        store = MemoryKnowledgeStore()
        agent = Agent(
            "policy-only",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            plugins=[KnowledgePlugin(store=store, expose_tool=False)],
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
            plugins=[KnowledgePlugin(store=store)],
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
            plugins=[KnowledgePlugin(store=store)],
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
            plugins=[KnowledgePlugin(store=store, write_event_log=False)],
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
            plugins=[KnowledgePlugin(store=store)],
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
