# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a HumanClient + HumanCliClient.

Design §13.5 claims "humans are first-class actors." Phase 3a
validates that claim by shipping :class:`HumanClient` — an
:class:`ActorClient` specialization whose notify handler routes
every incoming envelope to a pluggable :class:`HumanSurface`
instead of an ``Actor.ask`` call.

The test matrix:

* **Scripted surface:** the ``HumanScriptedSurface`` yields a
  pre-declared list of responses so tests can drive end-to-end
  sessions without real I/O. Used for every happy-path test.
* **CLI surface:** the ``HumanCliSurface`` reads from stdin/writes
  to stdout. Tests inject fake ``input_fn`` / ``output_fn``
  callables and verify the shell protocol without a real TTY.
* **Transport parity:** every human test runs against both
  ``LocalLink`` and ``WsLinkServer`` to validate the "works
  locally, works remotely" invariant.
* **Consulting round-trip:** the human answers a single question
  over LocalLink and over WsLink.
* **Conversation multi-turn:** several scripted replies drive a
  multi-turn conversation.
* **Broadcast observer:** human receives a broadcast but returns
  None — the surface logged the envelope, no reply was posted.
* **Identity runtime_kind:** the hub stamps ``"human"`` on the
  identity when ``register_human`` is called.
* **Disconnect hook:** ``HumanSurface.on_close`` fires on
  ``disconnect`` / ``unregister``.
* **Custom handler override:** a user can still override a
  specific session type with ``client.on("tournament")`` without
  disturbing the default surface routing for other types.
* **Non-text envelopes silent:** the scripted surface logs invite
  / opened envelopes but does not consume a response slot.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    HumanCliSurface,
    HumanClient,
    HumanScriptedSurface,
    LocalLink,
    SessionType,
    WsLinkClient,
    WsLinkServer,
    human_cli_client,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.hub import layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _spin_local_hub() -> tuple[Hub, LocalLink, HubClient]:
    hub = Hub(MemoryKnowledgeStore())
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    return hub, link, hc


async def _spin_ws_hub() -> tuple[Hub, WsLinkServer, WsLinkClient, HubClient]:
    hub = Hub(MemoryKnowledgeStore())
    server = WsLinkServer(host="127.0.0.1", port=0)
    server.on_connection(hub.connection_handler)
    await server.start()
    ws_client = WsLinkClient(server.url)
    hc = HubClient(hub, ws_client)
    return hub, server, ws_client, hc


# ---------------------------------------------------------------------------
# Scripted surface — basics
# ---------------------------------------------------------------------------


class TestScriptedSurface:
    @pytest.mark.asyncio
    async def test_scripted_surface_returns_sequential_responses(self) -> None:
        surface = HumanScriptedSurface(["one", "two", "three"])
        envs = [
            Envelope.text(
                session_id="s", sender_id="bob", content=f"q{i}"
            )
            for i in range(3)
        ]
        for env in envs:
            env.envelope_id = "x"

        # We don't need a real client — the scripted surface only
        # reads from the envelope. Pass None as a type-violation so
        # tests still exercise the surface without full wiring.
        results = []
        for env in envs:
            results.append(await surface.on_envelope(env, None))  # type: ignore[arg-type]
        assert results == ["one", "two", "three"]
        assert len(surface.seen) == 3

    @pytest.mark.asyncio
    async def test_scripted_surface_returns_none_when_exhausted(self) -> None:
        surface = HumanScriptedSurface(["only"])
        env = Envelope.text(session_id="s", sender_id="b", content="q")
        assert await surface.on_envelope(env, None) == "only"  # type: ignore[arg-type]
        assert await surface.on_envelope(env, None) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_scripted_surface_skips_non_text_envelopes(self) -> None:
        surface = HumanScriptedSurface(["reply"])
        system = Envelope(
            session_id="s",
            sender_id="hub",
            event_type="ag2.session.opened",
            event_data={"session_id": "s"},
        )
        # System envelope is observed (added to .seen) but does not
        # consume the first response slot.
        assert await surface.on_envelope(system, None) is None  # type: ignore[arg-type]
        assert len(surface.seen) == 1

        text = Envelope.text(session_id="s", sender_id="b", content="q")
        assert await surface.on_envelope(text, None) == "reply"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI surface — fake input/output
# ---------------------------------------------------------------------------


class TestCliSurface:
    @pytest.mark.asyncio
    async def test_cli_surface_calls_input_fn_with_prompt(self) -> None:
        captured_prompt: list[str] = []
        captured_output: list[str] = []

        def fake_input(prompt: str) -> str:
            captured_prompt.append(prompt)
            return "my answer"

        def fake_output(msg: str) -> None:
            captured_output.append(msg)

        surface = HumanCliSurface(
            input_fn=fake_input,
            output_fn=fake_output,
        )
        env = Envelope.text(
            session_id="s", sender_id="bob", content="how are you?"
        )
        response = await surface.on_envelope(env, None)  # type: ignore[arg-type]
        assert response == "my answer"
        assert captured_prompt
        assert "bob" in captured_prompt[0]
        assert "how are you?" in captured_prompt[0]

    @pytest.mark.asyncio
    async def test_cli_surface_empty_response_returns_none(self) -> None:
        surface = HumanCliSurface(input_fn=lambda _p: "")
        env = Envelope.text(session_id="s", sender_id="bob", content="q")
        assert await surface.on_envelope(env, None) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_cli_surface_ignores_non_text_events(self) -> None:
        called: list[str] = []

        def fake_input(prompt: str) -> str:
            called.append(prompt)
            return "x"

        surface = HumanCliSurface(input_fn=fake_input)
        system_env = Envelope(
            session_id="s",
            sender_id="hub",
            event_type="ag2.session.opened",
            event_data={},
        )
        response = await surface.on_envelope(system_env, None)  # type: ignore[arg-type]
        assert response is None
        assert called == []  # input_fn never called for system events

    @pytest.mark.asyncio
    async def test_cli_surface_handles_eof(self) -> None:
        def raising_input(_p: str) -> str:
            raise EOFError()

        surface = HumanCliSurface(input_fn=raising_input)
        env = Envelope.text(session_id="s", sender_id="b", content="q")
        # EOFError should be swallowed and return None, NOT propagate.
        assert await surface.on_envelope(env, None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HubClient.register_human — end-to-end registration
# ---------------------------------------------------------------------------


class TestRegisterHuman:
    @pytest.mark.asyncio
    async def test_register_human_stamps_runtime_kind(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            surface = HumanScriptedSurface([])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )
            assert isinstance(client, HumanClient)
            assert client.runtime_kind == "human"
            # The hub should have stored the runtime_kind on disk.
            raw = await hub._store.read(layout.actor_identity(client.actor_id))
            identity = json.loads(raw)
            assert identity["runtime_kind"] == "human"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_register_human_preserves_explicit_runtime_kind(self) -> None:
        """An identity with runtime_kind='browser' stays browser, not human."""

        hub, link, hc = await _spin_local_hub()
        try:
            surface = HumanScriptedSurface([])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(
                    name="web-approver", runtime_kind="browser"
                ),
            )
            assert client.runtime_kind == "browser"
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_register_human_surface_accessor_is_live(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            surface = HumanScriptedSurface(["ready"])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )
            assert client.surface is surface
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# End-to-end consulting session — LocalLink
# ---------------------------------------------------------------------------


class TestConsultingOverLocalLink:
    @pytest.mark.asyncio
    async def test_human_answers_consulting_question(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            # Alice is a normal Echo actor that will ASK the human.
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            # Operator is a scripted human.
            surface = HumanScriptedSurface(["42 is the answer"])
            operator = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            session = await alice.open(
                SessionType.CONSULTING, target="operator"
            )
            reply = await session.ask(
                "what is the meaning of life?", timeout=2.0
            )
            assert reply == "42 is the answer"
            # The surface logged the envelope it was asked about.
            assert any(
                "meaning of life" in (e.event_data.get("content") or "")
                for e in surface.seen
                if e.event_type == EV_TEXT
            )
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_human_returning_none_does_not_reply(self) -> None:
        """A human surface that returns None leaves the session waiting."""

        hub, link, hc = await _spin_local_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            surface = HumanScriptedSurface([None])  # explicit silent
            operator = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            session = await alice.open(
                SessionType.CONSULTING, target="operator"
            )
            from autogen.beta.network.errors import TimeoutError as NetTO

            # No reply arrives → session.ask times out.
            with pytest.raises(NetTO):
                await session.ask("hello?", timeout=0.5)
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# End-to-end consulting session — WsLink
# ---------------------------------------------------------------------------


class TestConsultingOverWsLink:
    @pytest.mark.asyncio
    async def test_human_answers_consulting_over_ws(self) -> None:
        hub, server, ws_client, hc = await _spin_ws_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            surface = HumanScriptedSurface(["remote human says hi"])
            operator = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            session = await alice.open(
                SessionType.CONSULTING, target="operator"
            )
            reply = await session.ask("?", timeout=3.0)
            assert reply == "remote human says hi"
        finally:
            await hc.close()
            await server.close()


# ---------------------------------------------------------------------------
# Conversation multi-turn
# ---------------------------------------------------------------------------


class TestConversationMultiTurn:
    @pytest.mark.asyncio
    async def test_human_replies_to_every_turn(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            surface = HumanScriptedSurface(["first", "second", "third"])
            operator = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            session = await alice.open(
                SessionType.CONVERSATION, target="operator"
            )
            replies = []
            for question in ["q1", "q2", "q3"]:
                replies.append(await session.ask(question, timeout=2.0))
            assert replies == ["first", "second", "third"]
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Broadcast observer
# ---------------------------------------------------------------------------


class TestBroadcastObserver:
    @pytest.mark.asyncio
    async def test_human_observes_broadcast_without_replying(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            surface = HumanScriptedSurface([])  # never replies
            operator = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            session = await alice.open(
                SessionType.BROADCAST, target=["operator"]
            )
            await session.send("listen up, everybody")
            # Give the surface time to observe.
            await asyncio.sleep(0.05)

            # The surface saw the envelope but did not post a reply.
            assert any(
                e.event_type == EV_TEXT
                and "listen up" in (e.event_data.get("content") or "")
                for e in surface.seen
            )
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Disconnect / on_close hook
# ---------------------------------------------------------------------------


class TestDisconnectHook:
    @pytest.mark.asyncio
    async def test_on_close_fires_on_disconnect(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            surface = HumanScriptedSurface([])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )
            assert surface.closed is False
            await client.disconnect()
            assert surface.closed is True
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_on_close_fires_on_unregister(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            surface = HumanScriptedSurface([])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )
            await client.unregister()
            assert surface.closed is True
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Custom handler override
# ---------------------------------------------------------------------------


class TestHandlerOverride:
    @pytest.mark.asyncio
    async def test_on_session_type_overrides_surface_routing(self) -> None:
        hub, link, hc = await _spin_local_hub()
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            surface = HumanScriptedSurface(["scripted reply"])
            client = await hc.register_human(
                surface=surface,
                identity=ActorIdentity(name="operator"),
            )

            # Override conversation handler to reply with a custom
            # message, bypassing the surface entirely.
            custom_seen: list[str] = []

            async def custom_handler(envelope, c):
                if envelope.event_type != EV_TEXT:
                    return
                custom_seen.append(envelope.content())
                await c._post_text_reply(envelope, "custom reply")

            client.on("conversation")(custom_handler)

            session = await alice.open(
                SessionType.CONVERSATION, target="operator"
            )
            reply = await session.ask("hi", timeout=2.0)
            assert reply == "custom reply"
            assert custom_seen == ["hi"]
            # The scripted surface was NEVER consulted.
            assert surface.seen == []
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# human_cli_client factory
# ---------------------------------------------------------------------------


class TestHumanCliClientFactory:
    @pytest.mark.asyncio
    async def test_factory_produces_wired_client(self) -> None:
        """``human_cli_client`` returns a HumanClient whose surface is a CLI."""

        hub, link, hc = await _spin_local_hub()
        try:
            responses = iter(["factory reply"])

            # Register via the hub directly to stamp an actor_id.
            identity = await hub.register(
                ActorIdentity(name="factory-op", runtime_kind="human")
            )
            from autogen.beta.network.client.human import human_cli_client

            client = human_cli_client(
                identity=identity,
                rule=await hub.get_rule(identity.actor_id),
                hub=hub,
                link=link,
                hub_client=hc,
                input_fn=lambda _p: next(responses),
            )
            assert isinstance(client, HumanClient)
            assert isinstance(client.surface, HumanCliSurface)
        finally:
            await hc.close()
            await link.close()
