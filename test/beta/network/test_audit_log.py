# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 3b ``hub/admin/audit.jsonl`` writer.

Every registry / session / rule / subscription / admin mutation writes a
JSON line with the shape described in §13.6:

```
{
  "ts":            ISO-8601 timestamp
  "actor_id":      actor responsible for the mutation (may be null for hub events)
  "action":        short string identifier
  "resource_type": "actor" | "session" | "rule" | "envelope"
  "resource_id":   id of the mutated resource
  "decision":      "allow" | "deny" | "timeout" | "rejected" | "drop" | "evict"
  "reason":        free-form string
  "trace_id":      envelope trace_id (may be null)
}
```

Writes are strictly non-fatal: tests that never call ``Hub.open`` never
start the writer, and the ``_audit`` call site short-circuits cleanly.
Tests that *do* want to assert on audit output use ``Hub.open`` (which
starts the writer) + ``await hub.close()`` (which drains it).
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
    InboxBlock,
    LimitsBlock,
    LocalLink,
    Rule,
    SessionType,
)
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


async def _read_audit_entries(hub: Hub) -> list[dict]:
    """Return all audit log entries as parsed dicts (empty list if none)."""

    raw = await hub._store.read(layout.admin_audit_log())
    if raw is None:
        return []
    return [json.loads(line) for line in raw.strip().split("\n") if line]


async def _wait_for_audit_entries(
    hub: Hub, *, minimum: int, timeout: float = 1.0
) -> list[dict]:
    """Poll the audit log until at least ``minimum`` entries are visible."""

    deadline = asyncio.get_event_loop().time() + timeout
    entries: list[dict] = []
    while asyncio.get_event_loop().time() < deadline:
        entries = await _read_audit_entries(hub)
        if len(entries) >= minimum:
            return entries
        await asyncio.sleep(0.01)
    return entries


# ---------------------------------------------------------------------------
# Sync constructor — writer is not started, _audit is a no-op
# ---------------------------------------------------------------------------


class TestAuditWriterDisabled:
    @pytest.mark.asyncio
    async def test_sync_constructor_does_not_start_writer(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        assert hub._audit_queue is None
        assert hub._audit_task is None

        # Audit calls are silent no-ops.
        hub._audit(
            actor_id="a",
            action="noop",
            resource_type="actor",
            resource_id="a",
            decision="allow",
        )

        entries = await _read_audit_entries(hub)
        assert entries == []

    @pytest.mark.asyncio
    async def test_close_is_noop_without_writer(self) -> None:
        hub = Hub(MemoryKnowledgeStore())
        await hub.close()  # must not raise


# ---------------------------------------------------------------------------
# Hub.open starts the writer; mutations are logged
# ---------------------------------------------------------------------------


class TestRegistrationAudit:
    @pytest.mark.asyncio
    async def test_register_writes_audit_line(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            identity = ActorIdentity(name="alice")
            stamped = await hub.register(identity)

            entries = await _wait_for_audit_entries(hub, minimum=1)
            assert len(entries) >= 1

            register_entries = [
                e for e in entries if e["action"] == "register_actor"
            ]
            assert len(register_entries) == 1
            entry = register_entries[0]
            assert entry["actor_id"] == stamped.actor_id
            assert entry["resource_type"] == "actor"
            assert entry["resource_id"] == stamped.actor_id
            assert entry["decision"] == "allow"
            assert entry["reason"] == "alice"
            assert "ts" in entry
            assert entry["trace_id"] is None
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_unregister_writes_audit_line(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="bob"))
            await hub.unregister(stamped.actor_id)

            entries = await _wait_for_audit_entries(hub, minimum=2)
            actions = [e["action"] for e in entries]
            assert "register_actor" in actions
            assert "unregister_actor" in actions

            unreg = next(e for e in entries if e["action"] == "unregister_actor")
            assert unreg["actor_id"] == stamped.actor_id
            assert unreg["reason"] == "bob"
        finally:
            await hub.close()


class TestSessionAudit:
    @pytest.mark.asyncio
    async def test_create_session_writes_audit(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )

            await alice.open(SessionType.CONSULTING, target="bob")

            entries = await _wait_for_audit_entries(hub, minimum=3)
            creates = [e for e in entries if e["action"] == "create_session"]
            assert len(creates) == 1
            entry = creates[0]
            assert entry["actor_id"] == alice.actor_id
            assert entry["decision"] == "allow"
            assert entry["reason"] == "consulting"
            assert entry["resource_type"] == "session"
        finally:
            await hc.close()
            await link.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_close_session_writes_audit(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"), identity=ActorIdentity(name="bob")
            )

            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await hub.close_session(
                session.session_id,
                reason="test_explicit",
                requested_by=alice.actor_id,
            )

            entries = await _wait_for_audit_entries(hub, minimum=3)
            closes = [e for e in entries if e["action"] == "close_session"]
            assert len(closes) == 1
            entry = closes[0]
            assert entry["actor_id"] == alice.actor_id
            assert entry["resource_id"] == session.session_id
            assert entry["decision"] == "allow"
            assert entry["reason"] == "test_explicit"
        finally:
            await hc.close()
            await link.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_ttl_sweep_writes_expire_audit(self) -> None:
        """An expired session writes an ``expire_session`` audit line."""

        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            short_ttl_rule = Rule(limits=LimitsBlock(session_ttl_default="1s"))
            alice = await hc.register(
                _Echo("alice"),
                identity=ActorIdentity(name="alice"),
                rule=short_ttl_rule,
            )
            bob = await hc.register(
                _Echo("bob"),
                identity=ActorIdentity(name="bob"),
                rule=short_ttl_rule,
            )
            session = await alice.open(SessionType.CONVERSATION, target="bob")

            # Force-expire by rewriting expires_at in place, then sweep.
            meta = hub._sessions[session.session_id]
            meta.expires_at = "1970-01-01T00:00:00+00:00"
            expired = await hub.sweep_expired_sessions()
            assert session.session_id in expired

            entries = await _wait_for_audit_entries(hub, minimum=4)
            expires = [e for e in entries if e["action"] == "expire_session"]
            assert len(expires) == 1
            entry = expires[0]
            assert entry["resource_id"] == session.session_id
            assert entry["reason"] == "ttl_expired"
            assert entry["actor_id"] is None  # hub-emitted
        finally:
            await hc.close()
            await link.close()
            await hub.close()


class TestRuleAudit:
    @pytest.mark.asyncio
    async def test_set_rule_writes_audit(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="bob"))
            new_rule = Rule(
                limits=LimitsBlock(
                    inbox=InboxBlock(max_pending=10, overflow="spool")
                )
            )
            await hub.set_rule(stamped.actor_id, new_rule)

            entries = await _wait_for_audit_entries(hub, minimum=2)
            updates = [e for e in entries if e["action"] == "update_rule"]
            assert len(updates) == 1
            entry = updates[0]
            assert entry["actor_id"] == stamped.actor_id
            assert entry["resource_type"] == "rule"
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_set_rule_updates_in_memory_and_store(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            stamped = await hub.register(ActorIdentity(name="bob"))
            old = await hub.get_rule(stamped.actor_id)
            assert old.limits.inbox.max_pending == 0

            new_rule = Rule(
                limits=LimitsBlock(
                    inbox=InboxBlock(max_pending=5, overflow="reject")
                )
            )
            await hub.set_rule(stamped.actor_id, new_rule)

            refreshed = await hub.get_rule(stamped.actor_id)
            assert refreshed.limits.inbox.max_pending == 5

            # And on disk.
            raw = await hub._store.read(layout.actor_rule(stamped.actor_id))
            restored = Rule.from_json(raw)
            assert restored.limits.inbox.max_pending == 5
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_set_rule_unknown_actor_raises(self) -> None:
        from autogen.beta.network import UnknownActorError

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            with pytest.raises(UnknownActorError):
                await hub.set_rule("ghost-id", Rule())
        finally:
            await hub.close()


# ---------------------------------------------------------------------------
# Inbox drop actions (task #3) produce audit entries too
# ---------------------------------------------------------------------------


class TestInboxDropAudit:
    @pytest.mark.asyncio
    async def test_drop_newest_writes_audit(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            bob_rule = Rule(
                limits=LimitsBlock(
                    inbox=InboxBlock(max_pending=1, overflow="drop_newest")
                )
            )
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"),
                identity=ActorIdentity(name="bob"),
                rule=bob_rule,
            )

            async def block(_envelope, _client):
                await asyncio.Event().wait()

            bob.on("conversation")(block)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.02)
            await session.send("two-dropped")
            await asyncio.sleep(0.05)

            entries = await _wait_for_audit_entries(hub, minimum=4)
            drops = [e for e in entries if e["action"] == "inbox_drop_newest"]
            assert len(drops) == 1
            drop = drops[0]
            assert drop["actor_id"] == bob.actor_id
            assert drop["decision"] == "drop"
            assert drop["resource_type"] == "envelope"
        finally:
            await hc.close()
            await link.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_drop_oldest_writes_audit(self) -> None:
        hub = await Hub.open(MemoryKnowledgeStore())
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            bob_rule = Rule(
                limits=LimitsBlock(
                    inbox=InboxBlock(max_pending=1, overflow="drop_oldest")
                )
            )
            alice = await hc.register(
                _Echo("alice"), identity=ActorIdentity(name="alice")
            )
            bob = await hc.register(
                _Echo("bob"),
                identity=ActorIdentity(name="bob"),
                rule=bob_rule,
            )

            async def block(_envelope, _client):
                await asyncio.Event().wait()

            bob.on("conversation")(block)
            session = await alice.open(SessionType.CONVERSATION, target="bob")
            await session.send("one")
            await asyncio.sleep(0.02)
            await session.send("two-evicts-one")
            await asyncio.sleep(0.05)

            entries = await _wait_for_audit_entries(hub, minimum=4)
            evicts = [e for e in entries if e["action"] == "inbox_drop_oldest"]
            assert len(evicts) == 1
            evict = evicts[0]
            assert evict["actor_id"] == bob.actor_id
            assert evict["decision"] == "evict"
            assert evict["resource_type"] == "envelope"
        finally:
            await hc.close()
            await link.close()
            await hub.close()


# ---------------------------------------------------------------------------
# Writer mechanics
# ---------------------------------------------------------------------------


class TestAuditWriterMechanics:
    @pytest.mark.asyncio
    async def test_audit_entry_schema(self) -> None:
        """Every writer entry carries the full documented schema."""

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            await hub.register(ActorIdentity(name="alice"))
            entries = await _wait_for_audit_entries(hub, minimum=1)
            entry = next(e for e in entries if e["action"] == "register_actor")
            expected_keys = {
                "ts",
                "actor_id",
                "action",
                "resource_type",
                "resource_id",
                "decision",
                "reason",
                "trace_id",
            }
            assert set(entry.keys()) == expected_keys
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_disk_store_round_trip(self, tmp_path) -> None:
        """Audit log survives on disk and is readable as jsonl."""

        store = DiskKnowledgeStore(str(tmp_path / "hub"))
        hub = await Hub.open(store)
        try:
            await hub.register(ActorIdentity(name="alice"))
            await hub.register(ActorIdentity(name="bob"))
            entries = await _wait_for_audit_entries(hub, minimum=2)
            assert len(entries) >= 2
        finally:
            await hub.close()

        # Reopen from disk and read the file directly.
        store2 = DiskKnowledgeStore(str(tmp_path / "hub"))
        raw = await store2.read(layout.admin_audit_log())
        assert raw is not None
        lines = [line for line in raw.strip().split("\n") if line]
        assert len(lines) >= 2
        for line in lines:
            parsed = json.loads(line)
            assert "action" in parsed

    @pytest.mark.asyncio
    async def test_close_drains_queue(self) -> None:
        """After close the queue is drained and the task is gone."""

        hub = await Hub.open(MemoryKnowledgeStore())
        await hub.register(ActorIdentity(name="alice"))
        await hub.close()
        assert hub._audit_task is None
        assert hub._audit_queue is None

    @pytest.mark.asyncio
    async def test_sorted_keys_are_stable(self) -> None:
        """Writer uses ``sort_keys`` so test assertions on output are stable."""

        hub = await Hub.open(MemoryKnowledgeStore())
        try:
            await hub.register(ActorIdentity(name="alice"))
            await _wait_for_audit_entries(hub, minimum=1)
            raw = await hub._store.read(layout.admin_audit_log())
            assert raw is not None
            # Each line must be valid JSON with keys in a deterministic order.
            first_line = raw.strip().split("\n")[0]
            parsed = json.loads(first_line)
            re_serialized = json.dumps(parsed, sort_keys=True)
            assert first_line == re_serialized
        finally:
            await hub.close()
