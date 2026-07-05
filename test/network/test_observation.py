# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Resume observation, capability index, skill render, mutation tests.

Three layers covered:

* **Skill render** (unit) — ``parse_skill_frontmatter`` + ``render_fallback_skill``.
* **Capability index + record_observation** (integration) — register
  populates the index; ``record_observation`` updates ``Resume.observed``
  and adds the agent to the capability bucket; ``unregister`` removes
  the agent and prunes empty buckets; the index round-trips through
  ``Hub.hydrate()``; the on-disk JSON cache reflects the in-memory state.
* **AgentClient mutation** — ``set_resume`` / ``add_example`` go through
  the hub and refresh the local cache.
* **TaskMirror end-to-end** — ``Agent.task(..., capability="X")`` inside
  a notify-handler turn drives ``Hub.record_observation`` automatically.
"""

import json

import pytest

from ag2 import Agent
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    Hub,
    HubClient,
    LocalLink,
    Passport,
    ResumeExample,
)
from ag2.network.client.skill_render import (
    parse_skill_frontmatter,
    render_fallback_skill,
)
from ag2.network.hub.layout import by_capability_path
from ag2.network.identity import (
    ObservedStat,
    Resume,
)
from ag2.task import TaskState
from ag2.testing import TestConfig

from ._helpers import ScriptedConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


# ── Skill render unit tests ─────────────────────────────────────────────────


class TestParseSkillFrontmatter:
    def test_parses_basic_frontmatter_and_body(self) -> None:
        md = "---\nname: alice\ndescription: Senior policy analyst.\n---\n\n## What I do\n\nCost-benefit framing.\n"
        parsed = parse_skill_frontmatter(md)
        assert parsed.frontmatter == {
            "name": "alice",
            "description": "Senior policy analyst.",
        }
        assert "## What I do" in parsed.body
        assert parsed.body.startswith("\n## What I do") or parsed.body.startswith("## What I do")

    def test_no_frontmatter_returns_full_body(self) -> None:
        md = "# alice\n\njust a body\n"
        parsed = parse_skill_frontmatter(md)
        assert parsed.frontmatter == {}
        assert parsed.body == md

    def test_unterminated_frontmatter_returns_full_body(self) -> None:
        md = "---\nname: alice\nbut no closing fence"
        parsed = parse_skill_frontmatter(md)
        assert parsed.frontmatter == {}
        assert parsed.body == md

    def test_skips_empty_and_comment_lines(self) -> None:
        md = "---\nname: alice\n\n# a comment\nrole: analyst\n---\nbody"
        parsed = parse_skill_frontmatter(md)
        assert parsed.frontmatter == {"name": "alice", "role": "analyst"}


class TestRenderFallbackSkill:
    def test_includes_capabilities_domains_and_summary(self) -> None:
        passport = Passport(name="alice")
        resume = Resume(
            claimed_capabilities=["debate", "analysis"],
            domains=["policy", "economics"],
            summary="Senior policy analyst.",
        )
        rendered = render_fallback_skill(passport, resume)
        assert rendered.startswith("---\n")
        assert "name: alice" in rendered
        assert "description: Senior policy analyst." in rendered
        assert "## Capabilities" in rendered
        assert "- debate" in rendered
        assert "## Domains" in rendered

    def test_includes_observed_track_record(self) -> None:
        passport = Passport(name="bob")
        resume = Resume(
            claimed_capabilities=["analysis"],
            observed={
                "analysis": ObservedStat(n=3, completed=2, failed=1),
            },
        )
        rendered = render_fallback_skill(passport, resume)
        assert "## Track record" in rendered
        assert "analysis" in rendered
        assert "n=3" in rendered

    def test_minimal_resume_renders_default_description(self) -> None:
        passport = Passport(name="solo")
        resume = Resume()
        rendered = render_fallback_skill(passport, resume)
        assert "name: solo" in rendered
        assert "Network-registered agent." in rendered


# ── Capability index integration tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_register_populates_capability_index() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["debate", "analysis"]),
    )

    assert hub.agents_with_capability("debate") == [alice.agent_id]
    assert hub.agents_with_capability("analysis") == [alice.agent_id]
    assert hub.agents_with_capability("missing") == []

    await hub.close()


@pytest.mark.asyncio
async def test_unregister_removes_from_capability_index() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"), resume=Resume(claimed_capabilities=["debate"]))
    bob = await hub.register(_agent("bob"), resume=Resume(claimed_capabilities=["debate"]))

    assert set(hub.agents_with_capability("debate")) == {alice.agent_id, bob.agent_id}

    await hub.unregister(alice.agent_id)
    assert hub.agents_with_capability("debate") == [bob.agent_id]

    await hub.unregister(bob.agent_id)
    assert hub.agents_with_capability("debate") == []
    # Empty bucket pruned from the index entirely.
    assert "debate" not in hub._capability_index

    await hub.close()


@pytest.mark.asyncio
async def test_capability_index_persisted_to_disk(tmp_path) -> None:
    store = DiskKnowledgeStore(str(tmp_path))
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["debate"]),
    )

    raw = await store.read(by_capability_path())
    assert raw is not None
    snapshot = json.loads(raw)
    assert snapshot == {"debate": [alice.agent_id]}

    await hub.close()


@pytest.mark.asyncio
async def test_capability_index_rebuilt_on_hydrate(tmp_path) -> None:
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub1.register(
        _agent("alice"),
        resume=Resume(
            claimed_capabilities=["debate"],
            observed={"reviews": ObservedStat(n=5, completed=4, failed=1)},
        ),
    )

    await hub1.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)

    # Both claimed and observed capabilities show up in the rebuilt index.
    assert hub2.agents_with_capability("debate") == [alice.agent_id]
    assert hub2.agents_with_capability("reviews") == [alice.agent_id]

    await hub2.close()


# ── record_observation tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_observation_updates_resume_observed_counters() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        resume=Resume(claimed_capabilities=["analysis"]),
    )

    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="analysis",
        outcome=TaskState.COMPLETED,
        latency_ms=420,
    )
    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="analysis",
        outcome=TaskState.FAILED,
    )
    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="analysis",
        outcome=TaskState.EXPIRED,
    )

    resume = await hub.get_resume(alice.agent_id)
    stat = resume.observed["analysis"]
    assert stat.n == 3
    assert stat.completed == 1
    assert stat.failed == 1
    assert stat.expired == 1
    assert stat.p50_latency_ms == 420  # last observed value (V1 placeholder)

    await hub.close()


@pytest.mark.asyncio
async def test_record_observation_adds_unclaimed_capability_to_index() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))  # no claims

    assert hub.agents_with_capability("emergent") == []

    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="emergent",
        outcome=TaskState.COMPLETED,
    )

    assert hub.agents_with_capability("emergent") == [alice.agent_id]

    await hub.close()


@pytest.mark.asyncio
async def test_record_observation_ignores_non_terminal_state() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="x",
        outcome=TaskState.RUNNING,
    )

    resume = await hub.get_resume(alice.agent_id)
    assert "x" not in resume.observed

    await hub.close()


# ── AgentClient mutation tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_client_set_resume_refreshes_local_cache() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"), resume=Resume(summary="initial"))

    await alice.set_resume(Resume(summary="updated", claimed_capabilities=["x"]))

    assert alice.resume.summary == "updated"
    assert "x" in alice.resume.claimed_capabilities
    # ``set_resume`` re-indexes claimed_capabilities so newly-claimed
    # caps surface under ``peers(action="find", capability=...)``.
    assert hub.agents_with_capability("x") == [alice.agent_id]

    # Removing a claim drops the agent from that bucket so the index
    # stays consistent with the current resume.
    await alice.set_resume(Resume(summary="updated2", claimed_capabilities=[]))
    assert hub.agents_with_capability("x") == []

    await hub.close()


@pytest.mark.asyncio
async def test_agent_client_add_example_appends() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    assert alice.resume.examples == []

    await alice.add_example(ResumeExample(title="reviewed PR #42", outcome="completed"))
    await alice.add_example(ResumeExample(title="triaged incident #7", outcome="completed"))

    fresh = await hub.get_resume(alice.agent_id)
    titles = [e.title for e in fresh.examples]
    assert titles == ["reviewed PR #42", "triaged incident #7"]

    await hub.close()


# ── TaskMirror end-to-end via Task lifecycle ────────────────────────────────


@pytest.mark.asyncio
async def test_task_mirror_records_observation_on_capability_tagged_task() -> None:
    """Running ``agent.task(capability=X)`` to completion through a mirror
    auto-calls ``Hub.record_observation`` and updates ``Resume.observed[X]``.

    Exercises the mirror plumbing directly (without the LLM tool
    surface) so this contract is verified independently of the
    end-to-end notify-handler integration covered in test_tools.py.
    """
    from ag2 import Context
    from ag2.network.task_mirror import TaskMirror
    from ag2.stream import MemoryStream

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    bob_agent = Agent(name="bob", config=ScriptedConfig("ack"))
    bob = await hub.register(
        bob_agent,
        resume=Resume(claimed_capabilities=["analysis"]),
    )

    stream = MemoryStream()
    mirror = TaskMirror(hub=hub, owner_id=bob.agent_id)
    sub_ids = mirror.attach(stream)
    try:
        async with bob_agent.task(
            "analysing alice's question",
            capability="analysis",
            context=Context(stream=stream),
        ) as task:
            await task.complete(result="done")
    finally:
        mirror.detach(stream, sub_ids)

    fresh = await hub.get_resume(bob.agent_id)
    assert "analysis" in fresh.observed
    stat = fresh.observed["analysis"]
    assert stat.n == 1
    assert stat.completed == 1
    assert bob.agent_id in hub.agents_with_capability("analysis")

    await hub.close()


@pytest.mark.asyncio
async def test_record_observation_writes_audit_with_observed_source() -> None:
    """Hub-side observation mutations are auditable as ``resume_set``
    records with ``source="observed"``, distinct from tenant-driven
    ``set_resume`` calls (``source="tenant"``)."""
    from ag2.network.hub.audit import (
        AUDIT_KIND_RESUME_SET,
        RESUME_SOURCE_OBSERVED,
        RESUME_SOURCE_TENANT,
    )

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    bob_hc = HubClient(link, hub=hub)
    bob = await bob_hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(claimed_capabilities=["analysis"]),
    )

    pre_audit = len(await hub._audit_log.read_all())

    # Tenant-driven update.
    await bob_hc._hub.set_resume(bob.agent_id, Resume(summary="updated by tenant"))
    # Hub-driven observation.
    await hub.record_observation(
        owner_id=bob.agent_id,
        capability="analysis",
        outcome=TaskState.COMPLETED,
        latency_ms=42,
    )

    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    resume_records = [r for r in new_records if r["kind"] == AUDIT_KIND_RESUME_SET]

    sources = [r.get("source") for r in resume_records]
    assert RESUME_SOURCE_TENANT in sources
    assert RESUME_SOURCE_OBSERVED in sources

    observed = next(r for r in resume_records if r.get("source") == RESUME_SOURCE_OBSERVED)
    assert observed["agent_id"] == bob.agent_id
    assert observed["capability"] == "analysis"
    assert observed["outcome"] == TaskState.COMPLETED.value

    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_task_capability_survives_hub_hydrate() -> None:
    """``TaskSpec.capability`` round-trips through hub persistence so an
    observation can fire on a terminal event after a hub restart.

    Pre-fix: ``_task_metadata_to_dict`` dropped ``capability``; on
    hydrate the spec came back with ``capability=None`` and
    ``record_observation`` wouldn't fire even after the task terminated.
    """
    import tempfile
    from pathlib import Path

    from ag2 import Context
    from ag2.network.task_mirror import TaskMirror
    from ag2.stream import MemoryStream

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskKnowledgeStore(Path(tmpdir))
        hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        bob_agent = Agent(name="bob", config=ScriptedConfig("ack"))
        bob = await hub.register(
            bob_agent,
            resume=Resume(claimed_capabilities=["analysis"]),
        )

        # Start a capability-tagged task; mirror persists metadata.
        stream = MemoryStream()
        mirror = TaskMirror(hub=hub, owner_id=bob.agent_id)
        sub_ids = mirror.attach(stream)
        async with bob_agent.task(
            "analysing",
            capability="analysis",
            context=Context(stream=stream),
        ) as task:
            task_id = task.task_id
            await task.complete(result="done")
        mirror.detach(stream, sub_ids)

        await hub.close()

        # Restart: new Hub, same store. ``_load_task`` rehydrates
        # TaskMetadata; the spec must preserve ``capability``.
        hub2 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
        rehydrated = hub2._tasks[task_id]
        assert rehydrated.spec.capability == "analysis"
        await hub2.close()


@pytest.mark.asyncio
async def test_task_mirror_no_observation_when_capability_absent() -> None:
    """Untagged tasks emit lifecycle events but don't touch ``observed``."""
    from ag2 import Context
    from ag2.network.task_mirror import TaskMirror
    from ag2.stream import MemoryStream

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    bob_agent = Agent(name="bob", config=ScriptedConfig("ack"))
    bob = await hub.register(bob_agent)

    stream = MemoryStream()
    mirror = TaskMirror(hub=hub, owner_id=bob.agent_id)
    sub_ids = mirror.attach(stream)
    try:
        async with bob_agent.task(
            "untagged work",
            context=Context(stream=stream),
        ) as task:
            await task.complete(result="ok")
    finally:
        mirror.detach(stream, sub_ids)

    fresh = await hub.get_resume(bob.agent_id)
    assert fresh.observed == {}

    await hub.close()
