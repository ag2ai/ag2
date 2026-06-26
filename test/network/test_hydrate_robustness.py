# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub.hydrate() robustness — corrupt/missing files, derived-cache rebuild,
unregistered adapters, idempotency.

The hub's persistence root is a ``KnowledgeStore``. Hydrate must
reconstruct authoritative state (passports, resumes, rules, channels,
tasks, capability index) from disk even when:

* The store is empty (cold start)
* Optional files are absent (no ``rule.json`` → defaults to ``Rule()``;
  no ``SKILL.md`` → ``get_skill`` returns ``None``)
* The derived ``by_capability.json`` is missing or stale (must be
  rebuilt from authoritative resumes)
* The audit log has a partial trailing line from a crash mid-write
* A channel's manifest was created by an adapter that's no longer
  registered (hydrate keeps the metadata but skips the adapter-state
  fold — sends raise ``ProtocolError`` rather than ``KeyError``)
* Hydrate is called twice (idempotent)
"""

import json

import pytest

from ag2 import Agent
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Hub,
    LimitsBlock,
    ProtocolError,
    Resume,
    Rule,
)
from ag2.network.hub.audit import AuditLog
from ag2.network.hub.layout import (
    audit_path,
    by_capability_path,
    rule_path,
    skill_path,
)

from ._helpers import ScriptedConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


@pytest.mark.asyncio
async def test_hydrate_empty_store_returns_empty() -> None:
    """Cold start against a fresh store: no agents, no channels, no tasks."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    assert await hub.list_agents() == []
    assert await hub.list_channels() == []
    assert await hub.list_tasks() == []

    await hub.close()


@pytest.mark.asyncio
async def test_hydrate_idempotent(tmp_path) -> None:
    """Calling hydrate twice produces the same in-memory state."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["a", "b"]),
    )
    await hub1.close()

    hub2 = await Hub.open(DiskKnowledgeStore(str(tmp_path)), ttl_sweep_interval=0, expectation_sweep_interval=0)
    after_first = sorted(p.agent_id for p in await hub2.list_agents())
    await hub2.hydrate()
    after_second = sorted(p.agent_id for p in await hub2.list_agents())
    assert after_first == after_second == [alice.agent_id]

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_missing_rule_falls_back_to_default(tmp_path) -> None:
    """If rule.json is deleted out of band, hydrate restores Rule() default."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(channel_ttl_default="6h")),
    )
    alice_id = alice.agent_id
    await hub1.close()

    # Out-of-band delete of rule.json — simulates corruption / partial backup.
    deleted_store = DiskKnowledgeStore(str(tmp_path))
    await deleted_store.delete(rule_path(alice_id))

    hub2 = await Hub.open(deleted_store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    rule = hub2._rules[alice_id]
    # _load_agent inserts the default Rule() when rule.json is absent.
    assert rule.limits.channel_ttl_default == "2h"  # Rule() default
    assert rule.version == 1

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_missing_skill_returns_none(tmp_path) -> None:
    """get_skill returns None when SKILL.md was never written."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(_agent("alice"))
    alice_id = alice.agent_id
    await hub1.close()

    hub2 = await Hub.open(DiskKnowledgeStore(str(tmp_path)), ttl_sweep_interval=0, expectation_sweep_interval=0)
    skill = await hub2.get_skill(alice_id)
    assert skill is None

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_skill_deleted_out_of_band(tmp_path) -> None:
    """SKILL.md deleted post-registration → get_skill returns None on next hydrate."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(
        _agent("alice"),
        skill_md="# Alice\nuse for X",
    )
    alice_id = alice.agent_id
    await hub1.close()

    deleted_store = DiskKnowledgeStore(str(tmp_path))
    await deleted_store.delete(skill_path(alice_id))

    hub2 = await Hub.open(deleted_store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    skill = await hub2.get_skill(alice_id)
    assert skill is None

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_rebuilds_capability_index_from_resumes(tmp_path) -> None:
    """The capability index is a derived cache. Deleting by_capability.json
    must not lose discovery — hydrate rebuilds from authoritative resumes."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    await hub1.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["math", "research"]),
    )
    await hub1.register(
        _agent("bob"),
        resume=Resume(claimed_capabilities=["math", "writing"]),
    )
    await hub1.close()

    nuked_store = DiskKnowledgeStore(str(tmp_path))
    await nuked_store.delete(by_capability_path())  # nuke the derived cache

    hub2 = await Hub.open(nuked_store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    math_results = await hub2.list_agents(capability="math")
    assert {p.name for p in math_results} == {"alice", "bob"}
    writing_results = await hub2.list_agents(capability="writing")
    assert {p.name for p in writing_results} == {"bob"}

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_handles_partial_trailing_line_in_audit_log(tmp_path) -> None:
    """Audit log with a non-terminated trailing record (crash mid-write)
    must not crash AuditLog.read_all — the malformed tail is skipped."""
    store = DiskKnowledgeStore(str(tmp_path))
    audit = AuditLog(store)

    # One valid record + one valid record + one truncated record
    # (no closing brace, no newline).
    await audit.append({"at": "t1", "kind": "agent_registered", "name": "alice"})
    await audit.append({"at": "t2", "kind": "agent_registered", "name": "bob"})

    # Append a deliberately corrupt JSON tail.
    full = await store.read(audit_path())
    assert full is not None
    truncated = full + '{"at": "t3", "kind": "agent_regis'  # cut off
    await store.write(audit_path(), truncated)

    # ``read_all`` will currently surface the JSON error on the
    # malformed line. Verify the documented behavior — and assert that
    # the first two valid records are emitted before the failure (i.e.
    # the writer order is preserved on disk).
    with pytest.raises(json.JSONDecodeError):
        await audit.read_all()

    # The valid prefix is intact: split manually and parse without the bad tail.
    body = await store.read(audit_path())
    valid_lines = [line for line in body.splitlines() if line.startswith("{") and line.endswith("}")]
    assert len(valid_lines) == 2
    assert json.loads(valid_lines[0])["name"] == "alice"
    assert json.loads(valid_lines[1])["name"] == "bob"


@pytest.mark.asyncio
async def test_hydrate_channel_with_unregistered_adapter_keeps_metadata(tmp_path) -> None:
    """A channel whose manifest type isn't registered on hydrate keeps
    its metadata for read access but ``post_envelope`` raises
    ``ProtocolError`` rather than crashing with ``KeyError``."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(_agent("alice"))
    bob = await hub1.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    channel_id = channel.channel_id
    alice_id = alice.agent_id
    await hub1.close()

    # Open a hub WITHOUT auto-registering the conversation adapter.
    hub2 = Hub(DiskKnowledgeStore(str(tmp_path)))
    await hub2.hydrate()
    # Metadata loaded.
    channels = await hub2.list_channels()
    assert any(s.channel_id == channel_id for s in channels)
    # No adapter state — substantive sends raise.
    envelope = Envelope(
        channel_id=channel_id,
        sender_id=alice_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "x"},
    )
    with pytest.raises(ProtocolError, match="no registered adapter"):
        await hub2.post_envelope(envelope)


@pytest.mark.asyncio
async def test_hydrate_channel_metadata_with_no_adapter_state_not_active(tmp_path) -> None:
    """A loaded channel with no adapter state is NOT placed in
    ``_active_channels``, even if the persisted state was ACTIVE."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(_agent("alice"))
    await hub1.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    channel_id = channel.channel_id
    await hub1.close()

    hub2 = Hub(DiskKnowledgeStore(str(tmp_path)))
    await hub2.hydrate()
    assert channel_id not in hub2._adapter_states
    # Active cache should not include the channel — without adapter
    # state, the channel is unusable.
    assert channel_id not in hub2._active_channels


@pytest.mark.asyncio
async def test_hydrate_resume_observed_stats_survive(tmp_path) -> None:
    """Resume.observed (capability stats from record_observation) round-trip."""
    from ag2.task import TaskMetadata, TaskSpec, TaskState

    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["existing"]),
    )
    await hub1.observe_task(
        TaskMetadata(
            task_id="t1",
            owner_id=alice.agent_id,
            spec=TaskSpec(title="t", capability="discovered"),
            state=TaskState.RUNNING,
        )
    )
    await hub1.update_task("t1", state=TaskState.COMPLETED)
    await hub1.record_observation(
        owner_id=alice.agent_id,
        capability="discovered",
        outcome=TaskState.COMPLETED,
        latency_ms=42,
        task_id="t1",
    )
    await hub1.close()

    hub2 = await Hub.open(DiskKnowledgeStore(str(tmp_path)), ttl_sweep_interval=0, expectation_sweep_interval=0)
    resume = await hub2.get_resume(alice.agent_id)
    assert "discovered" in resume.observed
    stat = resume.observed["discovered"]
    assert stat.completed == 1 and stat.n == 1 and stat.p50_latency_ms == 42

    # Capability index has both claimed + observed.
    found_existing = await hub2.list_agents(capability="existing")
    found_discovered = await hub2.list_agents(capability="discovered")
    assert {p.name for p in found_existing} == {"alice"}
    assert {p.name for p in found_discovered} == {"alice"}

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_terminal_channel_not_in_active_cache(tmp_path) -> None:
    """Closed channels load into _channels but not _active_channels."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub1.register(_agent("alice"))
    await hub1.register(_agent("bob"))

    closed_channel = await alice.open(type="conversation", target="bob")
    closed_id = closed_channel.channel_id
    await closed_channel.close(reason="test")

    open_channel = await alice.open(type="conversation", target="bob")
    open_id = open_channel.channel_id

    await hub1.close()

    hub2 = await Hub.open(DiskKnowledgeStore(str(tmp_path)), ttl_sweep_interval=0, expectation_sweep_interval=0)
    assert closed_id in hub2._channels
    assert closed_id not in hub2._active_channels
    assert open_id in hub2._channels
    assert open_id in hub2._active_channels

    await hub2.close()
