# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — discovery verb unit tests (find_actors, describe_actor)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    HubClient,
    LocalLink,
    SessionType,
)
from autogen.beta.network.client.verbs.discovery import (
    _summarise_identity,
    build_discovery_verbs,
)
from autogen.beta.network.hub import Hub


@pytest_asyncio.fixture
async def discovery_setup():
    """One hub with three identities + an alice-bound ActorClient."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    class _Actor:
        async def ask(self, content: str, **kwargs: Any) -> str:
            return content

    alice_client = await hub_client.register(
        _Actor(),
        identity=ActorIdentity(
            name="alice",
            display="Alice the Researcher",
            summary="Produces cited literature reviews",
            capabilities=["research", "summarization"],
            domains=["biomedicine"],
            strengths="biomed synthesis",
            skill_md="# Alice\nask in consulting",
        ),
    )
    # Two more identities the verb can discover (no client needed —
    # they only need to exist on the hub).
    await hub.register(
        ActorIdentity(
            name="bob",
            summary="Drafts marketing copy",
            capabilities=["writing"],
            domains=["finance"],
        ),
    )
    await hub.register(
        ActorIdentity(
            name="carol",
            summary="Audits ML systems",
            capabilities=["audit"],
            domains=["compliance"],
        ),
    )

    try:
        yield hub, alice_client
    finally:
        await hub_client.close()
        await link.close()


def _make_context() -> ConversationContext:
    return ConversationContext(stream=MagicMock())


async def _call(tool: Any, **arguments: Any) -> Any:
    """Invoke a verb tool directly and return the raw Python result."""

    event = ToolCallEvent(
        name=tool.schema.function.name,
        arguments=json.dumps(arguments),
    )
    result: ToolResultEvent = await tool(event, _make_context())
    return result.result.content


@pytest.mark.asyncio
class TestFindActors:
    async def test_unfiltered_lists_all(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _describe = build_discovery_verbs(alice_client)

        results = await _call(find_actors)
        names = {r["name"] for r in results}
        assert names == {"alice", "bob", "carol"}

    async def test_capability_filter(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        results = await _call(find_actors, capability="research")
        assert [r["name"] for r in results] == ["alice"]

    async def test_query_filter(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        results = await _call(find_actors, query="literature")
        assert [r["name"] for r in results] == ["alice"]

    async def test_query_matches_domain(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        results = await _call(find_actors, query="finance")
        assert [r["name"] for r in results] == ["bob"]

    async def test_query_and_capability_anded(
        self, discovery_setup
    ) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        # capability=research narrows to alice only; query=biomed
        # matches her summary so the AND keeps her in.
        results = await _call(
            find_actors, capability="research", query="biomed"
        )
        assert [r["name"] for r in results] == ["alice"]

    async def test_summary_strips_skill_md(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        results = await _call(find_actors, query="biomed")
        assert "skill_md" not in results[0]
        # But it does include the LLM-friendly fields
        assert "name" in results[0]
        assert "capabilities" in results[0]
        assert "summary" in results[0]

    async def test_no_match_returns_empty_list(self, discovery_setup) -> None:
        _hub, alice_client = discovery_setup
        find_actors, _ = build_discovery_verbs(alice_client)

        results = await _call(find_actors, query="zzz-not-present")
        assert results == []


@pytest.mark.asyncio
class TestDescribeActor:
    async def test_describe_returns_full_identity_with_skill(
        self, discovery_setup
    ) -> None:
        _hub, alice_client = discovery_setup
        _find, describe_actor = build_discovery_verbs(alice_client)

        result = await _call(describe_actor, name="alice")
        assert result["name"] == "alice"
        assert result["display"] == "Alice the Researcher"
        assert result["skill_md"] == "# Alice\nask in consulting"
        assert "auth" in result
        assert result["capabilities"] == ["research", "summarization"]

    async def test_describe_by_actor_id_works(
        self, discovery_setup
    ) -> None:
        _hub, alice_client = discovery_setup
        _find, describe_actor = build_discovery_verbs(alice_client)

        # Discover alice's id via find first
        find_actors, _ = build_discovery_verbs(alice_client)
        [alice_summary] = await _call(find_actors, query="literature")
        result = await _call(describe_actor, name=alice_summary["actor_id"])
        assert result["name"] == "alice"

    async def test_describe_unknown_returns_error(
        self, discovery_setup
    ) -> None:
        _hub, alice_client = discovery_setup
        _find, describe_actor = build_discovery_verbs(alice_client)

        result = await _call(describe_actor, name="ghost")
        assert "error" in result


class TestSummariseIdentity:
    """Pin the LLM-summary shape so future identity additions are deliberate."""

    def test_summary_keys_are_stable(self) -> None:
        identity = ActorIdentity(
            name="x",
            display="X",
            summary="s",
            capabilities=["c"],
            domains=["d"],
        )
        identity.actor_id = "id-x"
        result = _summarise_identity(identity)
        assert set(result.keys()) == {
            "name",
            "actor_id",
            "display",
            "summary",
            "capabilities",
            "domains",
            "owner",
            "version",
            "runtime_kind",
        }
