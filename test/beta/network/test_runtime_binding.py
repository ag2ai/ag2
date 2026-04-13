# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a runtime.json binding update on hello.

Design §3.4 specifies ``hub/actors/{actor_id}/runtime.json`` carrying
``binding`` / ``target`` / ``ws_url`` / ``http_url`` / ``reachable`` /
``last_heartbeat``. Phase 1 only wrote a placeholder
``{"binding": "local", "reachable": false}`` at registration time and
never updated it — which meant discovery responses reported stale or
outright wrong state the moment the actor connected.

Phase 3a wires ``Hub._write_runtime`` into both entry points:

* ``_handle_hello`` stamps the actor's runtime with the live
  endpoint's binding shape (``LocalLink`` → ``"local"``, ``WsLink``
  will carry ``"ws"`` + ``ws_url`` when it lands).
* ``connection_handler``'s finally block flips ``reachable=false``
  while preserving the last-known binding fields so discovery can
  still describe where the actor was.
* A reconnect that replaces the current endpoint re-stamps runtime
  via ``_handle_hello`` on the new connection; the old connection's
  cleanup must not race-overwrite that with ``reachable=false``.

This test module exercises every transition.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Hub,
    HubClient,
    LocalLink,
    SessionType,
)
from autogen.beta.network.hub import layout


class _Echo:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


async def _read_runtime(hub: Hub, actor_id: str) -> dict:
    raw = await hub._store.read(layout.actor_runtime(actor_id))
    assert raw is not None, f"runtime.json missing for {actor_id}"
    return json.loads(raw)


async def _spin() -> tuple[Hub, HubClient, LocalLink, "ActorClient"]:
    hub = Hub(MemoryKnowledgeStore())
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = await hc.register(_Echo("alice"), identity=ActorIdentity(name="alice"))
    return hub, hc, link, alice


# ---------------------------------------------------------------------------
# Phase 1 baseline — runtime.json is written at registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.asyncio
    async def test_runtime_exists_after_register_but_stale(self) -> None:
        """Before hello lands, runtime.json has a placeholder shape."""

        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)

        # Directly call hub.register so we don't go through the link
        # handshake — the runtime file should exist as a stub at this
        # point regardless of whether a client ever connects.
        identity = await hub.register(ActorIdentity(name="solo"))
        try:
            runtime = await _read_runtime(hub, identity.actor_id)
            assert runtime["actor_id"] == identity.actor_id
            assert runtime["binding"] == "local"
            # Phase 3a: a registration without a live connection is
            # NOT reachable yet — hello flips this.
            assert runtime["reachable"] is False
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Hello path — runtime is updated with live endpoint shape
# ---------------------------------------------------------------------------


class TestHelloStampsRuntime:
    @pytest.mark.asyncio
    async def test_hello_marks_runtime_reachable_with_binding(self) -> None:
        hub, hc, link, alice = await _spin()
        try:
            runtime = await _read_runtime(hub, alice.actor_id)
            assert runtime["actor_id"] == alice.actor_id
            assert runtime["binding"] == "local"
            assert runtime["reachable"] is True
            assert runtime["target"]  # non-empty endpoint id
            assert runtime["ws_url"] is None  # LocalLink has no ws_url
            assert runtime["http_url"] is None
            assert "last_heartbeat" in runtime
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_hello_stamps_endpoint_id_as_target(self) -> None:
        hub, hc, link, alice = await _spin()
        try:
            runtime = await _read_runtime(hub, alice.actor_id)
            # The endpoint id should be the UUID7 the LocalLink stamped.
            endpoint = hub._endpoints[alice.actor_id]
            assert runtime["target"] == endpoint.endpoint_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_hello_runtime_persisted_to_disk(self, tmp_path) -> None:
        """Disk store: runtime.json appears on the filesystem."""

        root = tmp_path / "hub"
        hub = Hub(DiskKnowledgeStore(str(root)))
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        alice = await hc.register(
            _Echo("alice"), identity=ActorIdentity(name="alice")
        )
        try:
            runtime = await _read_runtime(hub, alice.actor_id)
            assert runtime["reachable"] is True
            assert runtime["binding"] == "local"
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Disconnect path — runtime flips to reachable=false
# ---------------------------------------------------------------------------


class TestDisconnectMarksUnreachable:
    @pytest.mark.asyncio
    async def test_disconnect_flips_reachable_to_false(self) -> None:
        hub, hc, link, alice = await _spin()
        alice_id = alice.actor_id

        # Snapshot the "live" runtime then disconnect.
        before = await _read_runtime(hub, alice_id)
        assert before["reachable"] is True

        await alice.disconnect()
        # Give the hub's connection_handler finally block a tick.
        await asyncio.sleep(0.02)

        after = await _read_runtime(hub, alice_id)
        assert after["reachable"] is False
        # Binding fields preserved so discovery can describe the
        # last-known address.
        assert after["binding"] == before["binding"]
        assert after["target"] == before["target"]

        await hc.close()
        await link.close()

    @pytest.mark.asyncio
    async def test_reconnect_repaints_reachable(self) -> None:
        """A reconnect hello must leave runtime at reachable=true.

        The critical ordering: the new hello runs on a fresh endpoint
        and stamps runtime with the new target+reachable=true; then
        the OLD connection's finally block runs. The finally block
        must NOT overwrite the new runtime with reachable=false,
        because the actor is live on the new endpoint.
        """

        hub, hc, link, alice = await _spin()
        try:
            before_target = (await _read_runtime(hub, alice.actor_id))["target"]

            await alice.reconnect()
            # Give the old handler's finally block time to run after
            # the new hello has already stamped runtime.
            await asyncio.sleep(0.05)

            runtime = await _read_runtime(hub, alice.actor_id)
            assert runtime["reachable"] is True
            # The new endpoint has a fresh id, so target should have
            # rotated to the new one.
            assert runtime["target"] != before_target
            new_target = hub._endpoints[alice.actor_id].endpoint_id
            assert runtime["target"] == new_target
        finally:
            await hc.close()
            await link.close()


# ---------------------------------------------------------------------------
# Multi-actor independence
# ---------------------------------------------------------------------------


class TestMultiActorRuntime:
    @pytest.mark.asyncio
    async def test_two_actors_have_independent_runtime_entries(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        alice = await hc.register(
            _Echo("alice"), identity=ActorIdentity(name="alice")
        )
        bob = await hc.register(
            _Echo("bob"), identity=ActorIdentity(name="bob")
        )
        try:
            alice_runtime = await _read_runtime(hub, alice.actor_id)
            bob_runtime = await _read_runtime(hub, bob.actor_id)
            assert alice_runtime["actor_id"] == alice.actor_id
            assert bob_runtime["actor_id"] == bob.actor_id
            assert alice_runtime["target"] != bob_runtime["target"]
            assert alice_runtime["reachable"] is True
            assert bob_runtime["reachable"] is True
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_disconnect_only_one_actor_leaves_other_alive(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        alice = await hc.register(
            _Echo("alice"), identity=ActorIdentity(name="alice")
        )
        bob = await hc.register(
            _Echo("bob"), identity=ActorIdentity(name="bob")
        )
        alice_id = alice.actor_id
        bob_id = bob.actor_id

        await alice.disconnect()
        await asyncio.sleep(0.02)

        alice_runtime = await _read_runtime(hub, alice_id)
        bob_runtime = await _read_runtime(hub, bob_id)
        assert alice_runtime["reachable"] is False
        assert bob_runtime["reachable"] is True

        await hc.close()
        await link.close()


# ---------------------------------------------------------------------------
# End-to-end — runtime staying fresh across a real session
# ---------------------------------------------------------------------------


class TestRuntimeFreshnessAcrossSession:
    @pytest.mark.asyncio
    async def test_runtime_last_heartbeat_updates_on_reconnect(self) -> None:
        """Each hello bumps ``last_heartbeat`` to the current clock."""

        ticks = iter(
            [
                "2026-04-13T12:00:00Z",
                "2026-04-13T12:00:01Z",
                "2026-04-13T12:00:02Z",
                "2026-04-13T12:00:03Z",
                "2026-04-13T12:00:04Z",
                "2026-04-13T12:00:05Z",
                "2026-04-13T12:00:06Z",
            ]
        )

        def clock() -> str:
            return next(ticks)

        hub = Hub(MemoryKnowledgeStore(), clock=clock)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        alice = await hc.register(
            _Echo("alice"), identity=ActorIdentity(name="alice")
        )
        try:
            first = (await _read_runtime(hub, alice.actor_id))["last_heartbeat"]
            await alice.reconnect()
            await asyncio.sleep(0.02)
            second = (await _read_runtime(hub, alice.actor_id))["last_heartbeat"]
            assert second > first
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_session_open_does_not_clobber_runtime(self) -> None:
        """Opening a session should not modify the runtime record.

        Runtime only changes on hello / disconnect — session lifecycle
        has no bearing on an actor's transport binding.
        """

        hub, hc, link, alice = await _spin()
        bob = await hc.register(
            _Echo("bob"), identity=ActorIdentity(name="bob")
        )
        try:
            alice_runtime_before = await _read_runtime(hub, alice.actor_id)

            async def quiet(*_args, **_kwargs):
                return None

            bob.on("conversation")(quiet)
            alice.on("conversation")(quiet)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("hello")

            alice_runtime_after = await _read_runtime(hub, alice.actor_id)
            # target + binding stayed the same — only last_heartbeat
            # would drift if the implementation were wrong.
            assert alice_runtime_after["target"] == alice_runtime_before["target"]
            assert alice_runtime_after["binding"] == alice_runtime_before["binding"]
            assert alice_runtime_after["reachable"] is True
        finally:
            await hc.close()
            await link.close()
