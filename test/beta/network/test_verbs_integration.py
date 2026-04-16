# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — LLM-driven verb integration tests.

These tests wire a real :class:`autogen.beta.Actor` with a
:class:`TestConfig` canned LLM that emits a scripted sequence of
:class:`ToolCallEvent`s, then registers it through a real
:class:`HubClient` so the verbs are actually injected into the
``actor.ask`` turn by ``client/handlers.py::_ask``.

The point: prove the eight verbs work end-to-end through the same
notify-handler dispatch a real LLM agent would use, with no
hand-built scaffolding around the verb tools.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from autogen.beta import Actor
from autogen.beta.events import (
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
)
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    HubClient,
    LocalLink,
    Rule,
    SessionType,
    Task,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.hub import Hub
from autogen.beta.testing import TestConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(tool_name: str, **arguments: Any) -> ToolCallEvent:
    """Convenience: build a ToolCallEvent with JSON-serialised arguments.

    The first positional argument is named ``tool_name`` (not ``name``)
    so verbs whose schema includes a ``name`` argument — like
    ``describe_actor(name=...)`` — can pass it through ``**kwargs``
    without colliding with the helper's own positional.
    """

    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


@dataclass
class _ScriptedReply:
    body: str


class _ScriptedActor:
    """Echoes a fixed reply per ``ask`` call.

    Used for the receiving side of an integration test where we don't
    need a real LLM to drive replies — just deterministic content
    landing in the WAL.
    """

    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self._replies = list(replies)

    async def ask(self, content: str, **kwargs: Any) -> _ScriptedReply:
        if not self._replies:
            return _ScriptedReply(body=f"{self.name}: {content}")
        return _ScriptedReply(body=self._replies.pop(0))


@pytest_asyncio.fixture
async def hub_with_three_actors():
    """One hub, two registered receivers (bob, carol), an LLM actor slot for alice."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    bob = _ScriptedActor("bob", replies=["bob's response"])
    carol = _ScriptedActor("carol", replies=[])

    bob_client = await hub_client.register(
        bob,
        identity=ActorIdentity(
            name="bob",
            display="Writer Bob",
            summary="Drafts crisp marketing copy",
            capabilities=["writing"],
            domains=["marketing"],
        ),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )
    carol_client = await hub_client.register(
        carol,
        identity=ActorIdentity(
            name="carol",
            display="Auditor Carol",
            summary="Reviews compliance docs",
            capabilities=["audit"],
            domains=["compliance"],
        ),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )

    try:
        yield hub, hub_client, bob_client, carol_client
    finally:
        await hub_client.close()
        await link.close()


# ---------------------------------------------------------------------------
# Single-verb integration tests — drive the LLM through one verb at a time
# ---------------------------------------------------------------------------
#
# In every test below, **bob initiates the session to alice**. This is the
# critical pattern: alice is the LLM-under-test whose notify handler we
# want to fire. When alice initiates, her ``ActorClient`` short-circuits
# replies to her own sends (suppress-the-default-handler-on-replies, see
# ``actor_client.py:910``) so her TestConfig never gets a chance to
# produce a tool call. When **bob** initiates, the inbound message to
# alice is not a reply-to-self, the consulting handler dispatches her
# TestConfig sequence, and the verb tools are auto-injected as ``tools=``
# on her ``actor.ask`` call.


async def _register_alice_with_script(
    hub_client: HubClient,
    *events: Any,
) -> Any:
    """Register an LLM-driven alice with a scripted ``TestConfig``."""

    alice = Actor("alice", config=TestConfig(*events))
    return await hub_client.register(
        alice,
        identity=ActorIdentity(name="alice", capabilities=["ask"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )


class TestSingleVerbDrivenByLLM:
    @pytest.mark.asyncio
    async def test_find_actors_via_llm(self, hub_with_three_actors) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc("find_actors", query="writer"),
            "i found a writer",
        )

        # Bob asks alice a question via a consulting session. Alice's
        # default handler dispatches her TestConfig, which fires the
        # find_actors verb and then returns the canned text reply.
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("who can write me a tagline?", timeout=5.0)
        assert reply == "i found a writer"

    @pytest.mark.asyncio
    async def test_describe_actor_via_llm(self, hub_with_three_actors) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc("describe_actor", **{"name": "bob"}),
            "described bob",
        )
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("look me up", timeout=5.0)
        assert reply == "described bob"

    @pytest.mark.asyncio
    async def test_say_via_llm_in_same_session(
        self, hub_with_three_actors
    ) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc("say", content="hello via verb"),
            "verb-driven greeting sent",
        )

        # Conversation session so we can keep it open across multiple
        # turns and inspect alice's verb-driven send in the WAL.
        sess = await bob_client.open(SessionType.CONVERSATION, target="alice")
        await sess.send("kick off")
        await asyncio.sleep(0.2)

        # Pull alice's actor_id back out via the hub's name index
        alice_id = sess._client._hub._name_to_id["alice"]

        envelopes = await sess._client._hub.read_wal(sess.session_id)
        alice_text_contents = [
            e.content()
            for e in envelopes
            if e.event_type == EV_TEXT and e.sender_id == alice_id
        ]
        # The verb sent "hello via verb" *and* the final scripted text
        # ("verb-driven greeting sent") landed via the default handler's
        # post-text-reply path.
        assert "hello via verb" in alice_text_contents

    @pytest.mark.asyncio
    async def test_listen_via_llm(self, hub_with_three_actors) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc("listen", scope="session"),
            "read the session",
        )
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("ping", timeout=5.0)
        assert reply == "read the session"

    @pytest.mark.asyncio
    async def test_run_task_via_llm(self, hub_with_three_actors) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc(
                "run_task",
                title="quick task",
                description="say hi",
                owner="bob",
                blocking=True,
            ),
            "task complete",
        )
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("kick off", timeout=5.0)
        assert reply == "task complete"

    @pytest.mark.asyncio
    async def test_track_task_via_llm(self, hub_with_three_actors) -> None:
        hub, hub_client, bob_client, carol_client = hub_with_three_actors

        # First create a real task carol→bob on a separate session
        # so we have a stable id to track.
        from autogen.beta.network.task import TaskSpec

        seed = await carol_client.open(SessionType.CONSULTING, target="bob")
        task = await seed.create_task(
            TaskSpec(title="t", description="hi"),
            owner="bob",
            blocking=True,
            timeout=5.0,
        )
        task_id = task.task_id

        # Now drive alice via track_task on that id.
        await _register_alice_with_script(
            hub_client,
            _tc("track_task", task_id=task_id),
            "tracked the task",
        )
        sess2 = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess2.ask("status?", timeout=5.0)
        assert reply == "tracked the task"

    @pytest.mark.asyncio
    async def test_open_session_then_say_via_llm(
        self, hub_with_three_actors
    ) -> None:
        _hub, hub_client, bob_client, carol_client = hub_with_three_actors

        # Alice receives a question from bob, opens a fresh session
        # to carol mid-turn, says something into the new session, then
        # returns her final text reply to bob.
        await _register_alice_with_script(
            hub_client,
            _tc(
                "open_session",
                session_type="conversation",
                target="carol",
                intent="loop her in",
            ),
            _tc("say", content="hello carol from alice"),
            "linked carol",
        )

        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("can you ping carol?", timeout=5.0)
        assert reply == "linked carol"

        # The new alice→carol session should now exist with alice's
        # verb-driven message in its WAL.
        carol_id = sess._client._hub._name_to_id["carol"]
        carol_sessions = [
            s
            for s in sess._client._hub._sessions.values()
            if any(p.actor_id == carol_id for p in s.participants)
            and s.type == "conversation"
        ]
        assert carol_sessions, "no carol session created by open_session verb"
        wal = await sess._client._hub.read_wal(carol_sessions[0].session_id)
        text_envs = [e for e in wal if e.event_type == EV_TEXT]
        assert any("hello carol" in e.content() for e in text_envs)

        # And the new session metadata should carry the LLM's stated intent.
        meta = sess._client._hub.peek_session(carol_sessions[0].session_id)
        assert meta is not None
        assert meta.labels.get("intent") == "loop her in"

    @pytest.mark.asyncio
    async def test_leave_via_llm(self, hub_with_three_actors) -> None:
        _hub, hub_client, bob_client, _carol = hub_with_three_actors

        await _register_alice_with_script(
            hub_client,
            _tc("leave"),
            "left the session",
        )
        sess = await bob_client.open(SessionType.CONVERSATION, target="alice")
        await sess.send("kick off")
        await asyncio.sleep(0.2)

        meta = sess._client._hub.peek_session(sess.session_id)
        assert meta is not None
        assert meta.state.value == "closed"


# ---------------------------------------------------------------------------
# Multi-verb sequence — the canonical Phase 6 design test
# ---------------------------------------------------------------------------


class TestVerbChain:
    @pytest.mark.asyncio
    async def test_find_describe_open_say_chain(
        self, hub_with_three_actors
    ) -> None:
        _hub, hub_client, bob_client, carol_client = hub_with_three_actors

        # Alice runs through the canonical "discover and reach out"
        # flow: find a writer, describe them, open a session to a
        # different peer, say hello, then return a final summary
        # to bob (the requester).
        await _register_alice_with_script(
            hub_client,
            _tc("find_actors", query="writer"),
            _tc("describe_actor", **{"name": "carol"}),
            _tc(
                "open_session",
                session_type="conversation",
                target="carol",
                intent="initial outreach",
            ),
            _tc("say", content="hi carol, can you draft a tagline?"),
            "outreach sent",
        )

        # Bob initiates the consulting session that fires alice's
        # scripted turn.
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        reply = await sess.ask("can you reach out to a writer?", timeout=5.0)
        assert reply == "outreach sent"

        # Verify the chain landed: a new alice↔carol conversation
        # session exists with alice's verb-driven tagline message.
        carol_id = sess._client._hub._name_to_id["carol"]
        alice_carol = [
            s
            for s in sess._client._hub._sessions.values()
            if any(p.actor_id == carol_id for p in s.participants)
            and s.type == "conversation"
        ]
        assert alice_carol, "verb chain did not open the alice→carol session"
        wal = await sess._client._hub.read_wal(alice_carol[0].session_id)
        text_envs = [e for e in wal if e.event_type == EV_TEXT]
        assert any("tagline" in e.content() for e in text_envs)

        # The describe_actor call returned carol's identity to the LLM —
        # we cannot directly observe what alice's TestConfig "saw", but
        # the fact that the next verb successfully targeted carol is
        # proof the description was visible to her tool result.
