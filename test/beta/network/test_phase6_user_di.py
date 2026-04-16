# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — user-defined tool DI integration.

The pre-canned ``Annotated`` aliases (``SessionInject`` / ``TaskInject``
/ ``ActorClientInject`` / ``HubInject``) live in
``autogen.beta.network.client.inject``. They wrap
``Inject("ag2.network.<key>")`` so a user ``@tool`` can pull the live
network handles out of ``context.dependencies`` without having to
remember the qualified key strings.

These tests prove the aliases resolve to the *same* objects the
framework's auto-injected verbs see, that the keys are absent outside
a notify-handler turn (standalone ``Actor.ask``), and that user tools
can compose with verbs in the same turn.
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
    ActorClient,
    ActorClientInject,
    ActorIdentity,
    HubClient,
    HubInject,
    LocalLink,
    Rule,
    SessionInject,
    SessionType,
)
from autogen.beta.network.client.session import Session
from autogen.beta.network.hub import Hub
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool


def _tc(tool_name: str, **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


@dataclass
class _Reply:
    body: str


class _ScriptedActor:
    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self._replies = list(replies)

    async def ask(self, content: str, **kwargs: Any) -> _Reply:
        if not self._replies:
            return _Reply(body=f"{self.name}: {content}")
        return _Reply(body=self._replies.pop(0))


@pytest_asyncio.fixture
async def hub_with_alice_bob():
    """One hub with bob as a scripted receiver; alice slot is for the LLM-under-test."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    bob_client = await hub_client.register(
        _ScriptedActor("bob", replies=["bob's answer"]),
        identity=ActorIdentity(name="bob", capabilities=["answer"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )
    try:
        yield hub, hub_client, bob_client
    finally:
        await hub_client.close()
        await link.close()


# ---------------------------------------------------------------------------
# Live DI inside a notify-handler turn
# ---------------------------------------------------------------------------


class TestSessionInjectLive:
    @pytest.mark.asyncio
    async def test_session_inject_resolves_to_live_session(
        self, hub_with_alice_bob
    ) -> None:
        _hub, hub_client, bob_client = hub_with_alice_bob

        captured: dict[str, Any] = {}

        @tool
        async def my_tool(question: str, session: SessionInject) -> str:
            """User tool that needs the current session handle."""
            captured["session_id"] = session.session_id
            captured["session_type"] = type(session).__name__
            return f"saw session {session.session_id}"

        alice_cfg = TestConfig(
            _tc("my_tool", question="anything"),
            "done",
        )
        alice = Actor("alice", config=alice_cfg, tools=[my_tool])
        await hub_client.register(
            alice,
            identity=ActorIdentity(name="alice"),
            rule=Rule(access=AccessBlock(outbound_to=["*"])),
        )

        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        await sess.ask("kick off", timeout=5.0)

        assert "session_id" in captured
        assert captured["session_id"] == sess.session_id
        assert captured["session_type"] == "Session"

    @pytest.mark.asyncio
    async def test_actor_client_inject_resolves_to_owning_client(
        self, hub_with_alice_bob
    ) -> None:
        _hub, hub_client, bob_client = hub_with_alice_bob

        captured: dict[str, Any] = {}

        @tool
        async def my_tool(client: ActorClientInject) -> str:
            captured["actor_id"] = client.actor_id
            captured["identity_name"] = client.identity.name
            return "ok"

        alice_cfg = TestConfig(
            _tc("my_tool"),
            "done",
        )
        alice = Actor("alice", config=alice_cfg, tools=[my_tool])
        alice_client = await hub_client.register(
            alice,
            identity=ActorIdentity(name="alice"),
            rule=Rule(access=AccessBlock(outbound_to=["*"])),
        )

        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        await sess.ask("ping", timeout=5.0)

        assert captured["identity_name"] == "alice"
        assert captured["actor_id"] == alice_client.actor_id

    @pytest.mark.asyncio
    async def test_hub_inject_resolves_to_hub(
        self, hub_with_alice_bob
    ) -> None:
        hub, hub_client, bob_client = hub_with_alice_bob

        captured: dict[str, Any] = {}

        @tool
        async def my_tool(network_hub: HubInject) -> str:
            captured["is_same_hub"] = network_hub is hub
            return "ok"

        alice_cfg = TestConfig(_tc("my_tool"), "done")
        alice = Actor("alice", config=alice_cfg, tools=[my_tool])
        await hub_client.register(
            alice,
            identity=ActorIdentity(name="alice"),
            rule=Rule(access=AccessBlock(outbound_to=["*"])),
        )
        sess = await bob_client.open(SessionType.CONSULTING, target="alice")
        await sess.ask("ping", timeout=5.0)
        assert captured["is_same_hub"] is True

    @pytest.mark.asyncio
    async def test_user_tool_can_compose_with_verbs_via_session_inject(
        self, hub_with_alice_bob
    ) -> None:
        """A user tool gets ``SessionInject`` and uses ``session.send`` directly.

        This is the path that proves user tools and verbs share the
        same live handle — not just two distinct copies of the same
        metadata.
        """

        _hub, hub_client, bob_client = hub_with_alice_bob

        @tool
        async def echo_tool(content: str, session: SessionInject) -> str:
            await session.send(f"user-tool-echo: {content}")
            return "posted"

        alice_cfg = TestConfig(
            _tc("echo_tool", content="ping"),
            "responded via user tool",
        )
        alice = Actor("alice", config=alice_cfg, tools=[echo_tool])
        await hub_client.register(
            alice,
            identity=ActorIdentity(name="alice"),
            rule=Rule(access=AccessBlock(outbound_to=["*"])),
        )

        sess = await bob_client.open(SessionType.CONVERSATION, target="alice")
        await sess.send("trigger")
        await asyncio.sleep(0.2)

        wal = await sess._client._hub.read_wal(sess.session_id)
        text_envs = [
            e for e in wal if e.event_type == "ag2.msg.text"
        ]
        contents = [e.content() for e in text_envs]
        assert "user-tool-echo: ping" in contents


# ---------------------------------------------------------------------------
# Standalone (no hub) — DI keys must be absent
# ---------------------------------------------------------------------------


class TestInjectMissingOutsideNetworkTurn:
    @pytest.mark.asyncio
    async def test_session_inject_raises_outside_network(self) -> None:
        """A standalone ``Actor.ask`` cannot resolve ``SessionInject``.

        Without a hub-bound ``ActorClient`` driving the turn, the
        framework never stamps ``SESSION_DEP`` into
        ``context.dependencies``. The pydantic-driven argument
        resolver for the user tool then raises a missing-field
        validation error. The framework's tool runner catches that
        as a ``ToolErrorEvent``; the test client re-raises on the
        next call when it sees the error in the conversation, so the
        whole ``actor.ask`` propagates the failure to the caller.

        This is the desired behaviour: the user tool never receives
        a partial / fake session — failure surfaces clearly.
        """

        from pydantic import ValidationError

        @tool
        async def my_tool(session: SessionInject) -> str:  # noqa: ARG001
            return "should not run"

        alice = Actor(
            "alice",
            config=TestConfig(_tc("my_tool"), "done"),
            tools=[my_tool],
        )
        with pytest.raises(ValidationError):
            await alice.ask("ping")

    @pytest.mark.asyncio
    async def test_session_inject_with_default_works_outside_network(
        self,
    ) -> None:
        """A user tool that wants standalone-friendly behaviour can
        give the inject site an explicit ``default``.

        Phase 6 documents that the bare ``SessionInject`` alias has
        no default — to opt into "session is optional" the user
        must use ``Annotated[Session | None, Inject(SESSION_DEP, default=None)]``
        in their own annotation. This test pins that escape hatch.
        """

        from typing import Annotated

        from autogen.beta.annotations import Inject

        from autogen.beta.network import SESSION_DEP

        OptSession = Annotated[
            "Session | None",
            Inject(SESSION_DEP, default=None),
        ]

        captured: dict[str, Any] = {}

        @tool
        async def my_tool(session: OptSession = None) -> str:
            captured["session"] = session
            return "ok"

        alice = Actor(
            "alice",
            config=TestConfig(_tc("my_tool"), "done"),
            tools=[my_tool],
        )
        # Should NOT raise — the default kicks in.
        await alice.ask("ping")
        assert captured["session"] is None


# ---------------------------------------------------------------------------
# DI handle is the SAME object as what verbs use
# ---------------------------------------------------------------------------


class TestSessionInjectSharedWithVerbs:
    @pytest.mark.asyncio
    async def test_user_tool_session_is_same_as_verb_session(
        self, hub_with_alice_bob
    ) -> None:
        """A user tool sees the same Session id the ``say`` verb would target.

        The verbs default-target the session in
        ``context.dependencies[SESSION_DEP]``. If the user tool
        receives ``SessionInject`` and posts a message into it via
        ``session.send`` directly, the resulting envelope must land
        in the same WAL the verb-driven envelopes do.
        """

        _hub, hub_client, bob_client = hub_with_alice_bob

        captured: dict[str, str] = {}

        @tool
        async def first_tool(session: SessionInject) -> str:
            captured["user_tool_session_id"] = session.session_id
            await session.send("from-user-tool")
            return "ok"

        # The LLM calls the user tool, then the say verb, then closes
        # with a final text reply.
        alice_cfg = TestConfig(
            _tc("first_tool"),
            _tc("say", content="from-verb"),
            "done",
        )
        alice = Actor("alice", config=alice_cfg, tools=[first_tool])
        await hub_client.register(
            alice,
            identity=ActorIdentity(name="alice"),
            rule=Rule(access=AccessBlock(outbound_to=["*"])),
        )

        sess = await bob_client.open(SessionType.CONVERSATION, target="alice")
        await sess.send("kick off")
        await asyncio.sleep(0.2)

        # Both messages land in the same session
        assert captured["user_tool_session_id"] == sess.session_id

        wal = await sess._client._hub.read_wal(sess.session_id)
        text_contents = [
            e.content() for e in wal if e.event_type == "ag2.msg.text"
        ]
        assert "from-user-tool" in text_contents
        assert "from-verb" in text_contents
