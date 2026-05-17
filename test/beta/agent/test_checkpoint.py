# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Checkpoint, MemoryStream
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.fixture()
def simple_config() -> TestConfig:
    return TestConfig("Hello from the model!")


@pytest.fixture()
def tool_config() -> TestConfig:
    return TestConfig(
        ToolCallEvent(name="get_data"),
        "Here is your data summary.",
    )


# ---------------------------------------------------------------------------
# Checkpoint.from_reply basics
# ---------------------------------------------------------------------------


class TestCheckpointFromReply:
    @pytest.mark.asyncio()
    async def test_captures_agent_name(self, simple_config: TestConfig) -> None:
        agent = Agent("my-agent", config=simple_config)
        reply = await agent.ask("hello")
        cp = await Checkpoint.from_reply(reply)
        assert cp.agent_name == "my-agent"

    @pytest.mark.asyncio()
    async def test_captures_stream_id(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        stream = MemoryStream()
        reply = await agent.ask("hello", stream=stream)
        cp = await Checkpoint.from_reply(reply)
        assert cp.stream_id == stream.id

    @pytest.mark.asyncio()
    async def test_captures_events(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hello")
        cp = await Checkpoint.from_reply(reply)
        assert len(cp.events) > 0
        # Every stored event must have a type tag
        for e in cp.events:
            assert "__event__" in e

    @pytest.mark.asyncio()
    async def test_captures_context_variables(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hello", variables={"session_id": "abc123", "user": "alice"})
        cp = await Checkpoint.from_reply(reply)
        assert cp.variables["session_id"] == "abc123"
        assert cp.variables["user"] == "alice"

    @pytest.mark.asyncio()
    async def test_empty_variables_by_default(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hello")
        cp = await Checkpoint.from_reply(reply)
        assert isinstance(cp.variables, dict)

    @pytest.mark.asyncio()
    async def test_has_created_at(self, simple_config: TestConfig) -> None:
        from datetime import datetime

        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hello")
        cp = await Checkpoint.from_reply(reply)
        assert isinstance(cp.created_at, datetime)

    @pytest.mark.asyncio()
    async def test_repr(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hello")
        cp = await Checkpoint.from_reply(reply)
        r = repr(cp)
        assert "agent" in r
        assert "events=" in r


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointSaveLoad:
    @pytest.mark.asyncio()
    async def test_save_load_preserves_agent_name(self, simple_config: TestConfig) -> None:
        agent = Agent("my-agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cp.save(path)
            loaded = Checkpoint.load(path)
            assert loaded.agent_name == cp.agent_name
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_save_load_preserves_stream_id(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cp.save(path)
            loaded = Checkpoint.load(path)
            assert loaded.stream_id == cp.stream_id
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_save_load_preserves_event_count(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cp.save(path)
            loaded = Checkpoint.load(path)
            assert len(loaded.events) == len(cp.events)
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_save_load_preserves_variables(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi", variables={"key": "value", "count": 42})
        cp = await Checkpoint.from_reply(reply)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cp.save(path)
            loaded = Checkpoint.load(path)
            assert loaded.variables == cp.variables
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_save_writes_valid_json(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            cp.save(path)
            # Must parse without errors
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["version"] == "1"
            assert "events" in data
            assert "variables" in data
        finally:
            path.unlink(missing_ok=True)

    def test_load_rejects_unknown_version(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                {
                    "version": "99",
                    "agent_name": "a",
                    "stream_id": str(__import__("uuid").uuid4()),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "variables": {},
                    "events": [],
                },
                f,
            )
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="version"):
                Checkpoint.load(path)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MemoryStream.from_checkpoint
# ---------------------------------------------------------------------------


class TestMemoryStreamFromCheckpoint:
    @pytest.mark.asyncio()
    async def test_stream_has_same_id(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        restored = await MemoryStream.from_checkpoint(cp)
        assert restored.id == cp.stream_id

    @pytest.mark.asyncio()
    async def test_restored_stream_has_events(self, simple_config: TestConfig) -> None:
        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        restored = await MemoryStream.from_checkpoint(cp)
        events = list(await restored.history.get_events())
        assert len(events) == len(cp.events)

    @pytest.mark.asyncio()
    async def test_restored_events_are_base_events(self, simple_config: TestConfig) -> None:
        from autogen.beta.events.base import BaseEvent

        agent = Agent("agent", config=simple_config)
        reply = await agent.ask("hi")
        cp = await Checkpoint.from_reply(reply)

        restored = await MemoryStream.from_checkpoint(cp)
        events = list(await restored.history.get_events())
        assert all(isinstance(e, BaseEvent) for e in events)


# ---------------------------------------------------------------------------
# Resume conversation from checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    @pytest.mark.asyncio()
    async def test_resume_sees_previous_history(self, mock: MagicMock) -> None:
        """After resuming from checkpoint, the model is called with prior context."""

        def capture_tool() -> str:
            return "captured"

        # Two-turn config: first ask triggers tool, second ask is plain reply
        config = TestConfig(
            ToolCallEvent(name="capture_tool"),
            "Turn 1 done.",
            "Turn 2 done.",
        )
        agent = Agent("agent", config=config, tools=[capture_tool])

        # Turn 1
        reply1 = await agent.ask("start")
        cp = await Checkpoint.from_reply(reply1)

        # Turn 2 using checkpoint-restored stream
        stream = await MemoryStream.from_checkpoint(cp)
        reply2 = await agent.ask("continue", stream=stream)

        # The stream after turn 2 has more events than the checkpoint
        events_after = list(await reply2.history.get_events())
        assert len(events_after) > len(cp.events)

    @pytest.mark.asyncio()
    async def test_resume_restores_variables(self, simple_config: TestConfig) -> None:
        """Context variables saved in a checkpoint are passed back on resume."""
        agent = Agent("agent", config=simple_config)

        reply1 = await agent.ask("hi", variables={"session": "xyz"})
        cp = await Checkpoint.from_reply(reply1)

        # Variables survive save/load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cp.save(path)
            loaded = Checkpoint.load(path)
        finally:
            path.unlink(missing_ok=True)

        assert loaded.variables.get("session") == "xyz"

    @pytest.mark.asyncio()
    async def test_resume_does_not_mutate_original_checkpoint(self, simple_config: TestConfig) -> None:
        """Resuming from a checkpoint should not modify the checkpoint's event list."""
        config = TestConfig("First reply.", "Second reply.")
        agent = Agent("agent", config=config)

        reply1 = await agent.ask("first")
        cp = await Checkpoint.from_reply(reply1)
        original_event_count = len(cp.events)

        stream = await MemoryStream.from_checkpoint(cp)
        await agent.ask("second", stream=stream)

        assert len(cp.events) == original_event_count

    @pytest.mark.asyncio()
    async def test_checkpoint_after_tool_call(self, tool_config: TestConfig, mock: MagicMock) -> None:
        """Checkpoint captures tool call events."""

        call_log: list[str] = []

        def get_data() -> str:
            call_log.append("called")
            return "some data"

        agent = Agent("agent", config=tool_config, tools=[get_data])
        reply = await agent.ask("get me data")
        cp = await Checkpoint.from_reply(reply)

        # At least one event should be a ToolCallEvent or ToolResultsEvent
        event_type_names = {e.get("__event__", "").split(".")[-1] for e in cp.events}
        assert any("Tool" in name for name in event_type_names)

    @pytest.mark.asyncio()
    async def test_agent_name_property_on_reply(self, simple_config: TestConfig) -> None:
        agent = Agent("named-agent", config=simple_config)
        reply = await agent.ask("hi")
        assert reply.agent_name == "named-agent"
