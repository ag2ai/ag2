# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3b ``on_change``-driven identity/rule cache invalidation.

When an operator or a sidecar process rewrites an actor's rule or
identity JSON via the shared store (bypassing ``hub.set_rule`` /
``hub.register``), a long-running hub must notice and drop its stale
in-memory cache entry so subsequent reads see the new data.

Scoped intentionally small: only identity + rule are invalidated.
Cross-process session subscription fan-out is explicitly deferred to a
later phase — §14 Phase 3b "Deferred".
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Hub,
    HubClient,
    InboxBlock,
    KnowledgeAccess,
    LimitsBlock,
    LocalLink,
    Rule,
    TransformSpec,
)
from autogen.beta.network.hub import layout
from autogen.beta.network.transport.frames import RuleChangedFrame


# ---------------------------------------------------------------------------
# Memory store — active subscribers fire synchronously
# ---------------------------------------------------------------------------


class TestMemoryStoreInvalidation:
    @pytest.mark.asyncio
    async def test_rule_rewrite_invalidates_cache(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            old = await hub.get_rule(stamped.actor_id)
            assert old.limits.max_concurrent_sessions == 32  # default

            # External mutation — bypass hub.set_rule entirely.
            new_rule = Rule(limits=LimitsBlock(max_concurrent_sessions=7))
            await hub._store.write(
                layout.actor_rule(stamped.actor_id),
                new_rule.to_json(),
            )
            # Give the callback time to fire.
            await asyncio.sleep(0.02)

            refreshed = await hub.get_rule(stamped.actor_id)
            assert refreshed.limits.max_concurrent_sessions == 7
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_identity_rewrite_invalidates_cache(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))

            # External mutation — change capabilities.
            updated = ActorIdentity(
                name="alice",
                actor_id=stamped.actor_id,
                capabilities=["research", "new-skill"],
            )
            await hub._store.write(
                layout.actor_identity(stamped.actor_id),
                updated.to_json(),
            )
            await asyncio.sleep(0.02)

            refreshed = await hub.describe(stamped.actor_id)
            assert "new-skill" in refreshed.capabilities
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_rule_delete_drops_from_cache(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            assert stamped.actor_id in hub._rules

            await hub._store.delete(layout.actor_rule(stamped.actor_id))
            await asyncio.sleep(0.02)

            assert stamped.actor_id not in hub._rules
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_identity_delete_drops_from_name_index(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            assert "alice" in hub._name_to_id

            await hub._store.delete(layout.actor_identity(stamped.actor_id))
            await asyncio.sleep(0.02)

            assert "alice" not in hub._name_to_id
            assert stamped.actor_id not in hub._identities
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_name_change_updates_index(self) -> None:
        """A rename via external edit updates the name → id index."""

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            renamed = ActorIdentity(
                name="alice_v2",
                actor_id=stamped.actor_id,
            )
            await hub._store.write(
                layout.actor_identity(stamped.actor_id),
                renamed.to_json(),
            )
            await asyncio.sleep(0.02)

            assert "alice" not in hub._name_to_id
            assert hub._name_to_id.get("alice_v2") == stamped.actor_id
            assert hub._identities[stamped.actor_id].name == "alice_v2"
        finally:
            await hub.close()


# ---------------------------------------------------------------------------
# set_rule path is observed too (PUT /v1/actors/{id}/rule uses it)
# ---------------------------------------------------------------------------


class TestSetRuleThroughCacheLayer:
    @pytest.mark.asyncio
    async def test_set_rule_is_observed(self) -> None:
        """``hub.set_rule`` writes through the store; on_change fires."""

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            new_rule = Rule(
                access=AccessBlock(
                    knowledge=KnowledgeAccess(
                        expose=["/public/**"],
                        readers=["ag2:*:*"],
                    )
                )
            )
            await hub.set_rule(stamped.actor_id, new_rule)
            # The cache is updated by set_rule directly, so there is no
            # stale window — but the on_change callback also fires and
            # we must not accidentally revert the cache to a parse of
            # the same file. Verify the post-state is stable.
            await asyncio.sleep(0.02)
            refreshed = await hub.get_rule(stamped.actor_id)
            assert refreshed.access.knowledge.expose == ["/public/**"]
            assert refreshed.access.knowledge.readers == ["ag2:*:*"]
        finally:
            await hub.close()


# ---------------------------------------------------------------------------
# Lifecycle — subscription is cleaned up on close
# ---------------------------------------------------------------------------


class TestCacheInvalidationLifecycle:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        await hub.close()
        await hub.close()  # no second-call failure
        assert hub._cache_sub is None

    @pytest.mark.asyncio
    async def test_after_close_no_further_invalidation(self) -> None:
        """After close, store rewrites do not attempt to touch caches."""

        store = MemoryKnowledgeStore()
        hub = await Hub.open(store)
        stamped = await hub.register(ActorIdentity(name="alice"))
        await hub.close()

        # Rewrite through the store — must not raise or re-fire.
        new_rule = Rule(limits=LimitsBlock(max_concurrent_sessions=99))
        await store.write(layout.actor_rule(stamped.actor_id), new_rule.to_json())
        await asyncio.sleep(0.02)
        # No assertions on the cache state — hub is closed; the
        # important thing is no exception escapes here.

    @pytest.mark.asyncio
    async def test_sync_constructor_has_no_subscription(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        assert hub._cache_sub is None


# ---------------------------------------------------------------------------
# Disk store — watchdog-driven
# ---------------------------------------------------------------------------


class TestDiskStoreInvalidation:
    @pytest.mark.asyncio
    async def test_disk_rule_rewrite_invalidates(self, tmp_path) -> None:
        """watchdog fires on a real filesystem rewrite."""

        pytest.importorskip("watchdog")

        store = DiskKnowledgeStore(str(tmp_path / "hub"))
        hub = await Hub.open(store)
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            assert hub._rules[stamped.actor_id].limits.max_concurrent_sessions == 32

            new_rule = Rule(limits=LimitsBlock(max_concurrent_sessions=3))
            await store.write(
                layout.actor_rule(stamped.actor_id),
                new_rule.to_json(),
            )

            # watchdog fires asynchronously; poll the cache for up to
            # 2 seconds with a generous interval since the FSEvents /
            # inotify latency can be 100-300ms.
            for _ in range(40):
                cached = hub._rules.get(stamped.actor_id)
                if cached is not None and cached.limits.max_concurrent_sessions == 3:
                    break
                await asyncio.sleep(0.05)

            assert hub._rules[stamped.actor_id].limits.max_concurrent_sessions == 3
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_disk_close_releases_watchdog(self, tmp_path) -> None:
        pytest.importorskip("watchdog")

        store = DiskKnowledgeStore(str(tmp_path / "hub"))
        hub = await Hub.open(store)
        await hub.close()
        assert hub._cache_sub is None


# ---------------------------------------------------------------------------
# RuleChangedFrame emission — regression tests for the Phase 5a seam
# ---------------------------------------------------------------------------


class _EchoActor:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ask(self, content: str, **_kwargs):
        class _R:
            def __init__(self, body: str) -> None:
                self.body = body

        return _R(f"echo:{content}")


def _capture_rule_changed(endpoint) -> list[RuleChangedFrame]:
    """Wrap ``endpoint.send_frame`` to collect every outbound RuleChangedFrame.

    The tests need to assert the frame was both (a) constructed without
    raising and (b) carries the expected ``version`` / ``transforms``
    payload, so we intercept at the endpoint boundary and still forward
    the frame to the real client so the inbox loop stays healthy.
    """

    captured: list[RuleChangedFrame] = []
    original = endpoint.send_frame

    async def _wrapper(frame):
        if isinstance(frame, RuleChangedFrame):
            captured.append(frame)
        await original(frame)

    endpoint.send_frame = _wrapper  # type: ignore[method-assign]
    return captured


class TestRuleChangedEmission:
    @pytest.mark.asyncio
    async def test_set_rule_emits_frame_with_version_to_live_endpoint(self) -> None:
        """``set_rule`` against a live endpoint constructs a valid frame.

        Regression for the latent bug where ``Hub.set_rule`` built
        ``RuleChangedFrame(actor_id=..., transforms=...)`` without the
        required ``version`` field. The buggy line was behind an
        ``if endpoint is not None and not endpoint.closed`` guard, so
        every prior test registered through ``hub.register(...)``
        (no endpoint) and never tripped the ``TypeError``.
        """

        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            client = await hc.register(
                _EchoActor("alice"), identity=ActorIdentity(name="alice")
            )
            endpoint = hub._endpoints[client.actor_id]
            captured = _capture_rule_changed(endpoint)

            new_rule = Rule(
                version=7,
                limits=LimitsBlock(max_concurrent_sessions=3),
                transforms=[
                    TransformSpec(stage="pre_receive", apply="redact_pii"),
                ],
            )
            await hub.set_rule(client.actor_id, new_rule)

            assert len(captured) == 1
            frame = captured[0]
            assert frame.actor_id == client.actor_id
            assert frame.version == 7
            assert frame.transforms == [
                {"stage": "pre_receive", "apply": "redact_pii", "when": {}}
            ]
        finally:
            await hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_out_of_band_rule_write_emits_frame(self) -> None:
        """The FS-watcher path must also push ``RuleChangedFrame``.

        Before the fix, ``_reload_actor_rule`` only refreshed the
        in-memory cache on an external store write, and the
        ``ActorClient`` never saw the change — which broke §4.3 for
        operator-edited rules.
        """

        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            client = await hc.register(
                _EchoActor("alice"), identity=ActorIdentity(name="alice")
            )
            endpoint = hub._endpoints[client.actor_id]
            captured = _capture_rule_changed(endpoint)

            # External mutation — bypass hub.set_rule entirely.
            new_rule = Rule(
                version=4,
                transforms=[
                    TransformSpec(
                        stage="pre_send",
                        apply={
                            "python": {
                                "module": "myorg.guards",
                                "class": "PromptGuard",
                                "config": {"max_tokens": 8000},
                            }
                        },
                        when={"session_type": "consulting"},
                    ),
                ],
            )
            await hub._store.write(
                layout.actor_rule(client.actor_id),
                new_rule.to_json(),
            )

            # Poll: the memory-store on_change callback is synchronous
            # but the reload schedules `_emit_rule_changed` behind the
            # cache-invalidation lock.
            for _ in range(40):
                if captured:
                    break
                await asyncio.sleep(0.02)

            assert len(captured) >= 1
            frame = captured[-1]
            assert frame.actor_id == client.actor_id
            assert frame.version == 4
            assert frame.transforms[0]["stage"] == "pre_send"
            assert frame.transforms[0]["apply"] == {
                "python": {
                    "module": "myorg.guards",
                    "class": "PromptGuard",
                    "config": {"max_tokens": 8000},
                }
            }
            assert frame.transforms[0]["when"] == {"session_type": "consulting"}
        finally:
            await hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_set_rule_without_live_endpoint_is_silent(self) -> None:
        """``set_rule`` against an offline actor must not raise."""

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="alice"))
            # No HubClient, no link — actor has no live endpoint.
            new_rule = Rule(version=2, limits=LimitsBlock(max_concurrent_sessions=9))
            await hub.set_rule(stamped.actor_id, new_rule)

            refreshed = await hub.get_rule(stamped.actor_id)
            assert refreshed.version == 2
            assert refreshed.limits.max_concurrent_sessions == 9
        finally:
            await hub.close()
