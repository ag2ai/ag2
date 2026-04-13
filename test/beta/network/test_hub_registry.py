# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub registry tests: register / unregister / find / describe / rule storage."""

from __future__ import annotations

import json

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    AuthBlock,
    DuplicateRegistrationError,
    Hub,
    Rule,
    UnknownActorError,
)
from autogen.beta.network.hub import layout


@pytest.mark.asyncio
async def test_register_stamps_actor_id_and_persists_identity(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    ident = ActorIdentity(name="alice", capabilities=["research"], summary="…")
    stamped = await hub.register(ident)

    assert stamped.actor_id is not None
    assert stamped.name == "alice"

    # Identity written to the store.
    raw = await mem_store.read(layout.actor_identity(stamped.actor_id))
    assert raw is not None
    restored = ActorIdentity.from_json(raw)
    assert restored.actor_id == stamped.actor_id
    assert restored.capabilities == ["research"]


@pytest.mark.asyncio
async def test_register_persists_default_rule(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    stamped = await hub.register(ActorIdentity(name="alice"))
    raw = await mem_store.read(layout.actor_rule(stamped.actor_id))
    assert raw is not None
    Rule.from_json(raw)  # round-trip ok


@pytest.mark.asyncio
async def test_register_persists_explicit_rule(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    custom = Rule(access=AccessBlock(inbound_from=["ag2:*:*"]))
    stamped = await hub.register(ActorIdentity(name="alice"), rule=custom)
    raw = await mem_store.read(layout.actor_rule(stamped.actor_id))
    restored = Rule.from_json(raw or "{}")
    assert restored.access.inbound_from == ["ag2:*:*"]


@pytest.mark.asyncio
async def test_register_writes_skill_md_when_present(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    stamped = await hub.register(
        ActorIdentity(name="alice", skill_md="# Alice\nconsulting preferred"),
    )
    raw = await mem_store.read(layout.actor_skill(stamped.actor_id))
    assert raw is not None
    assert "consulting preferred" in raw


@pytest.mark.asyncio
async def test_register_writes_runtime_binding(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    stamped = await hub.register(ActorIdentity(name="alice"))
    raw = await mem_store.read(layout.actor_runtime(stamped.actor_id))
    data = json.loads(raw or "{}")
    assert data["actor_id"] == stamped.actor_id
    assert data["binding"] == "local"
    assert "last_heartbeat" in data


@pytest.mark.asyncio
async def test_duplicate_registration_raises(hub: Hub) -> None:
    await hub.register(ActorIdentity(name="alice"))
    with pytest.raises(DuplicateRegistrationError):
        await hub.register(ActorIdentity(name="alice"))


@pytest.mark.asyncio
async def test_two_identities_same_name_get_distinct_actor_ids_across_hubs() -> None:
    hub_a = Hub(MemoryKnowledgeStore())
    hub_b = Hub(MemoryKnowledgeStore())
    stamped_a = await hub_a.register(ActorIdentity(name="alice"))
    stamped_b = await hub_b.register(ActorIdentity(name="alice"))
    assert stamped_a.actor_id != stamped_b.actor_id


@pytest.mark.asyncio
async def test_find_lists_all_registered_identities(hub: Hub) -> None:
    await hub.register(ActorIdentity(name="alice", capabilities=["research"]))
    await hub.register(ActorIdentity(name="bob", capabilities=["writing"]))
    results = await hub.find()
    names = {i.name for i in results}
    assert names == {"alice", "bob"}


@pytest.mark.asyncio
async def test_find_by_capability_filters(hub: Hub) -> None:
    await hub.register(ActorIdentity(name="alice", capabilities=["research"]))
    await hub.register(ActorIdentity(name="bob", capabilities=["writing"]))
    assert [i.name for i in await hub.find(capability="research")] == ["alice"]


@pytest.mark.asyncio
async def test_describe_accepts_name_or_actor_id(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    by_name = await hub.describe("alice")
    by_id = await hub.describe(alice.actor_id or "")
    assert by_name.actor_id == by_id.actor_id == alice.actor_id


@pytest.mark.asyncio
async def test_describe_unknown_raises(hub: Hub) -> None:
    with pytest.raises(UnknownActorError):
        await hub.describe("ghost")


@pytest.mark.asyncio
async def test_get_rule_returns_applied_rule(hub: Hub) -> None:
    stamped = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(access=AccessBlock(inbound_from=["ag2:*:*"])),
    )
    rule = await hub.get_rule(stamped.actor_id or "")
    assert rule.access.inbound_from == ["ag2:*:*"]


@pytest.mark.asyncio
async def test_unregister_removes_actor_state(hub: Hub, mem_store: MemoryKnowledgeStore) -> None:
    stamped = await hub.register(ActorIdentity(name="alice"))
    actor_id = stamped.actor_id or ""
    await hub.unregister(actor_id)
    with pytest.raises(UnknownActorError):
        await hub.describe("alice")
    assert not await mem_store.exists(layout.actor_identity(actor_id))


@pytest.mark.asyncio
async def test_unregister_unknown_raises(hub: Hub) -> None:
    with pytest.raises(UnknownActorError):
        await hub.unregister("does-not-exist")


@pytest.mark.asyncio
async def test_register_with_wrong_auth_scheme_refuses() -> None:
    from autogen.beta.network import AuthError

    hub = Hub(MemoryKnowledgeStore())
    ident = ActorIdentity(name="alice", auth=AuthBlock(scheme="jwt"))
    with pytest.raises(AuthError):
        await hub.register(ident)
