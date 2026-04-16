# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — Step 0 prerequisite regressions.

Five small additive items the verb surface depends on. Each test
locks the contract from the design doc so a future refactor that
breaks the prework also breaks the relevant verb.

* :class:`TestHubFindQuery` — ``Hub.find(query=...)`` substring match
* :class:`TestHttpFindQueryParam` — ``GET /v1/actors?query=`` forwarding
* :class:`TestActorClientOpenIntent` — ``intent`` propagates to ``labels``
* :class:`TestNewDIKeys` — qualified key strings exist and are exported
* :class:`TestDIKeysPopulated` — ``_ask`` stamps live handles
* :class:`TestPreCannedAliases` — ``SessionInject`` etc. resolve correctly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ACTOR_CLIENT_DEP,
    HUB_DEP,
    SESSION_DEP,
    SESSION_ID_VAR,
    TASK_DEP,
    ActorClient,
    ActorClientInject,
    ActorIdentity,
    HubClient,
    HubInject,
    LocalLink,
    SessionInject,
    SessionType,
    TaskInject,
)
from autogen.beta.network.client.handlers import _ask
from autogen.beta.network.client.session import Session
from autogen.beta.network.client.task import Task
from autogen.beta.network.http import build_app
from autogen.beta.network.hub import Hub


# ---------------------------------------------------------------------------
# Reusable doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeReply:
    body: str


class _RecorderActor:
    """Records every ``ask`` invocation so tests can inspect kwargs."""

    def __init__(self, name: str = "fake", reply: str = "") -> None:
        self.name = name
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def ask(self, content: str, **kwargs: Any) -> FakeReply:
        self.calls.append({"content": content, **kwargs})
        return FakeReply(body=self.reply or f"{self.name}: {content}")


# ---------------------------------------------------------------------------
# Hub.find(query=...)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHubFindQuery:
    async def test_find_query_matches_name_substring(self, hub: Hub) -> None:
        await hub.register(ActorIdentity(name="ag2:researcher:1"))
        await hub.register(ActorIdentity(name="ag2:writer:1"))
        await hub.register(ActorIdentity(name="acme:auditor:1"))

        results = await hub.find(query="research")
        assert {i.name for i in results} == {"ag2:researcher:1"}

    async def test_find_query_matches_summary_substring(self, hub: Hub) -> None:
        await hub.register(
            ActorIdentity(name="alice", summary="Produces cited literature reviews"),
        )
        await hub.register(
            ActorIdentity(name="bob", summary="Drafts marketing copy"),
        )
        results = await hub.find(query="literature")
        assert [i.name for i in results] == ["alice"]

    async def test_find_query_matches_display_and_strengths(
        self, hub: Hub
    ) -> None:
        await hub.register(
            ActorIdentity(
                name="alice",
                display="Alice the Researcher",
                strengths="Strong at biomed synthesis",
            ),
        )
        await hub.register(ActorIdentity(name="bob"))
        # Display match
        assert [i.name for i in await hub.find(query="researcher")] == ["alice"]
        # Strengths match
        assert [i.name for i in await hub.find(query="biomed")] == ["alice"]

    async def test_find_query_matches_domain_entry(self, hub: Hub) -> None:
        await hub.register(
            ActorIdentity(name="alice", domains=["biomedicine", "climate"]),
        )
        await hub.register(
            ActorIdentity(name="bob", domains=["finance"]),
        )
        results = await hub.find(query="climate")
        assert [i.name for i in results] == ["alice"]

    async def test_find_query_is_case_insensitive(self, hub: Hub) -> None:
        await hub.register(
            ActorIdentity(name="alice", summary="Loves CAPS LOCK"),
        )
        results = await hub.find(query="caps")
        assert [i.name for i in results] == ["alice"]

    async def test_find_query_and_capability_are_anded(self, hub: Hub) -> None:
        await hub.register(
            ActorIdentity(
                name="alice",
                capabilities=["research"],
                summary="biomed",
            ),
        )
        await hub.register(
            ActorIdentity(
                name="bob",
                capabilities=["research"],
                summary="finance",
            ),
        )
        results = await hub.find(capability="research", query="biomed")
        assert [i.name for i in results] == ["alice"]

    async def test_find_query_no_match_returns_empty(self, hub: Hub) -> None:
        await hub.register(ActorIdentity(name="alice", summary="bio"))
        results = await hub.find(query="zzz-not-present")
        assert results == []

    async def test_find_query_none_returns_all(self, hub: Hub) -> None:
        await hub.register(ActorIdentity(name="alice"))
        await hub.register(ActorIdentity(name="bob"))
        assert {i.name for i in await hub.find(query=None)} == {"alice", "bob"}


# ---------------------------------------------------------------------------
# HTTP /v1/actors?query=
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHttpFindQueryParam:
    async def test_query_param_filters_results(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        await hub.register(
            ActorIdentity(name="alice", summary="biomed researcher"),
        )
        await hub.register(
            ActorIdentity(name="bob", summary="finance analyst"),
        )

        app = build_app(hub)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            resp = await ac.get("/v1/actors", params={"query": "biomed"})
            assert resp.status_code == 200
            actors = resp.json()["actors"]
            assert [a["name"] for a in actors] == ["alice"]

    async def test_query_and_capability_together(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        await hub.register(
            ActorIdentity(
                name="alice",
                summary="biomed",
                capabilities=["research"],
            ),
        )
        await hub.register(
            ActorIdentity(
                name="bob",
                summary="biomed",
                capabilities=["writing"],
            ),
        )
        app = build_app(hub)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            resp = await ac.get(
                "/v1/actors",
                params={"capability": "research", "query": "biomed"},
            )
            actors = resp.json()["actors"]
            assert [a["name"] for a in actors] == ["alice"]

    async def test_query_omitted_returns_all(self) -> None:
        store = MemoryKnowledgeStore()
        hub = Hub(store)
        await hub.register(ActorIdentity(name="alice"))
        await hub.register(ActorIdentity(name="bob"))
        app = build_app(hub)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            resp = await ac.get("/v1/actors")
            actors = resp.json()["actors"]
            assert {a["name"] for a in actors} == {"alice", "bob"}


# ---------------------------------------------------------------------------
# ActorClient.open(intent=...)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def two_clients():
    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    alice = _RecorderActor("alice")
    bob = _RecorderActor("bob")
    alice_client = await hub_client.register(
        alice, identity=ActorIdentity(name="alice", capabilities=["ask"])
    )
    bob_client = await hub_client.register(
        bob, identity=ActorIdentity(name="bob", capabilities=["answer"])
    )
    try:
        yield hub, alice_client, bob_client
    finally:
        await hub_client.close()
        await link.close()


@pytest.mark.asyncio
class TestActorClientOpenIntent:
    async def test_intent_lands_in_session_labels(self, two_clients) -> None:
        hub, alice_client, _bob_client = two_clients
        session = await alice_client.open(
            SessionType.CONSULTING,
            target="bob",
            intent="literature review",
        )
        meta = hub.peek_session(session.session_id)
        assert meta is not None
        assert meta.labels == {"intent": "literature review"}

    async def test_intent_visible_via_session_metadata(
        self, two_clients
    ) -> None:
        hub, alice_client, _ = two_clients
        session = await alice_client.open(
            SessionType.CONSULTING,
            target="bob",
            intent="climate-change report",
        )
        meta = hub.peek_session(session.session_id)
        assert meta is not None
        assert meta.labels.get("intent") == "climate-change report"

    async def test_explicit_labels_override_intent(self, two_clients) -> None:
        hub, alice_client, _ = two_clients
        session = await alice_client.open(
            SessionType.CONSULTING,
            target="bob",
            intent="initial intent",
            labels={"intent": "explicit override"},
        )
        meta = hub.peek_session(session.session_id)
        assert meta is not None
        assert meta.labels["intent"] == "explicit override"

    async def test_intent_omitted_no_label(self, two_clients) -> None:
        hub, alice_client, _ = two_clients
        session = await alice_client.open(
            SessionType.CONSULTING, target="bob"
        )
        meta = hub.peek_session(session.session_id)
        assert meta is not None
        assert "intent" not in meta.labels


# ---------------------------------------------------------------------------
# DI key constants
# ---------------------------------------------------------------------------


class TestNewDIKeys:
    """Pin the qualified key strings — verbs and user tools depend on them."""

    def test_session_dep_value(self) -> None:
        assert SESSION_DEP == "ag2.network.session"

    def test_actor_client_dep_value(self) -> None:
        assert ACTOR_CLIENT_DEP == "ag2.network.actor_client"

    def test_task_dep_value(self) -> None:
        assert TASK_DEP == "ag2.network.task"

    def test_legacy_keys_unchanged(self) -> None:
        # Phase 1/2 contract — protect from accidental rename
        assert HUB_DEP == "ag2.network.hub"
        assert SESSION_ID_VAR == "ag2.network.session_id"

    def test_keys_are_distinct(self) -> None:
        keys = {SESSION_DEP, ACTOR_CLIENT_DEP, TASK_DEP, HUB_DEP, SESSION_ID_VAR}
        assert len(keys) == 5


# ---------------------------------------------------------------------------
# DI keys populated by _ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDIKeysPopulated:
    async def test_ask_stamps_session_and_actor_client_deps(
        self, two_clients
    ) -> None:
        _hub, alice_client, _ = two_clients
        # Drive a fake session id through _ask. We don't need a real
        # session to live — _ask only reads metadata via peek_session
        # and stamps the result. peek_session returning None still
        # exercises the fallback path; for a populated path we need
        # a live session.
        session = await alice_client.open(
            SessionType.CONSULTING, target="bob"
        )
        actor: _RecorderActor = alice_client.actor  # type: ignore[assignment]

        await _ask(alice_client, "ping", session.session_id)

        # The default consulting handler funnels content through _ask;
        # _ask is the one we want to assert on. Find the call we just made.
        last = actor.calls[-1]
        assert last["variables"][SESSION_ID_VAR] == session.session_id
        deps = last["dependencies"]
        assert deps[HUB_DEP] is alice_client._hub
        assert deps[ACTOR_CLIENT_DEP] is alice_client
        assert isinstance(deps[SESSION_DEP], Session)
        assert deps[SESSION_DEP].session_id == session.session_id
        # No task in this path
        assert TASK_DEP not in deps

    async def test_ask_stamps_task_dep_when_handler_is_task(
        self, two_clients
    ) -> None:
        _hub, alice_client, _ = two_clients
        actor: _RecorderActor = alice_client.actor  # type: ignore[assignment]
        session = await alice_client.open(
            SessionType.CONSULTING, target="bob"
        )

        # Build a fake Task handle — the handler does not care what
        # state the task is in for the DI population test.
        from autogen.beta.network.task import TaskMetadata, TaskSpec, TaskState
        from autogen.beta.network.client.task import Task as TaskHandle

        spec = TaskSpec(title="t", description="d")
        meta = TaskMetadata(
            task_id="task-test",
            session_id=session.session_id,
            owner_id=alice_client.actor_id,
            requester_id=alice_client.actor_id,
            spec=spec,
            state=TaskState.RUNNING,
            created_at="2026-04-14T00:00:00Z",
            expires_at="2026-04-14T01:00:00Z",
        )
        task = TaskHandle(session=session, metadata=meta)

        await _ask(alice_client, "ping", session.session_id, task=task)
        last = actor.calls[-1]
        deps = last["dependencies"]
        assert deps[TASK_DEP] is task

    async def test_ask_passes_network_verbs_as_tools(
        self, two_clients
    ) -> None:
        _hub, alice_client, _ = two_clients
        actor: _RecorderActor = alice_client.actor  # type: ignore[assignment]
        session = await alice_client.open(
            SessionType.CONSULTING, target="bob"
        )

        await _ask(alice_client, "ping", session.session_id)

        last = actor.calls[-1]
        tools = last.get("tools", [])
        # Eight verbs per design §10.5
        assert len(tools) == 8
        names = {t.schema.function.name for t in tools}
        assert names == {
            "find_actors",
            "describe_actor",
            "open_session",
            "say",
            "listen",
            "run_task",
            "track_task",
            "leave",
        }

    async def test_ask_no_session_id_skips_dep_population(
        self, two_clients
    ) -> None:
        _hub, alice_client, _ = two_clients
        actor: _RecorderActor = alice_client.actor  # type: ignore[assignment]
        await _ask(alice_client, "ping", session_id=None)
        # Without a session id we don't stamp anything (matches Phase 1
        # behavior). Verbs are still injected as tools — this is the
        # "actor running outside a session" path which Phase 6 leaves
        # well-alone for now.
        last = actor.calls[-1]
        assert "variables" not in last
        assert "dependencies" not in last


# ---------------------------------------------------------------------------
# Pre-canned Annotated aliases
# ---------------------------------------------------------------------------


class TestPreCannedAliases:
    """Each alias resolves to ``Annotated[T, Inject('ag2.network.X')]``.

    We do not run the resolver here — the integration tests in Step 4
    cover live DI resolution. These tests just pin the alias surface
    so the qualified key strings stay in sync with
    ``policies.session_inbox``.
    """

    def test_session_inject_uses_qualified_key(self) -> None:
        from typing import get_args

        args = get_args(SessionInject)
        assert len(args) == 2
        # First arg is the type (Session forward ref or class)
        # Second is the Inject instance
        inject = args[1]
        assert inject.name == SESSION_DEP

    def test_task_inject_uses_qualified_key(self) -> None:
        from typing import get_args

        args = get_args(TaskInject)
        assert args[1].name == TASK_DEP

    def test_actor_client_inject_uses_qualified_key(self) -> None:
        from typing import get_args

        args = get_args(ActorClientInject)
        assert args[1].name == ACTOR_CLIENT_DEP

    def test_hub_inject_uses_qualified_key(self) -> None:
        from typing import get_args

        args = get_args(HubInject)
        assert args[1].name == HUB_DEP
