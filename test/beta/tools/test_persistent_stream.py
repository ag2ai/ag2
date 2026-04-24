# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.history import MemoryStorage
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.subagents.persistent_stream import persistent_stream


@pytest.fixture()
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture()
def parent_stream(storage: MemoryStorage) -> MemoryStream:
    return MemoryStream(storage=storage)


@pytest.fixture()
def ctx(parent_stream: MemoryStream) -> Context:
    return Context(stream=parent_stream, dependencies={})


def _make_agent(name: str = "helper") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


class TestPersistentStream:
    def test_returns_memory_stream(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        result = factory(agent, ctx)

        assert isinstance(result, MemoryStream)

    def test_reuses_same_stream_id_on_second_call(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        first = factory(agent, ctx)
        second = factory(agent, ctx)

        assert first.id == second.id

    def test_different_agents_get_different_streams(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent_a = _make_agent("alice")
        agent_b = _make_agent("bob")

        stream_a = factory(agent_a, ctx)
        stream_b = factory(agent_b, ctx)

        assert stream_a.id != stream_b.id

    def test_stores_stream_id_in_dependencies(self, ctx: Context) -> None:
        factory = persistent_stream()
        agent = _make_agent("helper")

        stream = factory(agent, ctx)

        assert ctx.dependencies["ag:helper:stream"] == stream.id

    def test_uses_parent_storage_backend(self, ctx: Context, storage: MemoryStorage) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        stream = factory(agent, ctx)

        assert stream.history.storage is storage

    def test_independent_contexts_get_independent_streams(self, storage: MemoryStorage) -> None:
        factory = persistent_stream()
        agent = _make_agent()

        ctx1 = Context(stream=MemoryStream(storage=storage), dependencies={})
        ctx2 = Context(stream=MemoryStream(storage=storage), dependencies={})

        stream1 = factory(agent, ctx1)
        stream2 = factory(agent, ctx2)

        assert stream1.id != stream2.id
