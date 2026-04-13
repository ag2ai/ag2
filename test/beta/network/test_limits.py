# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2 extended limits.

Phase 1 enforced ``max_concurrent_sessions`` and ``session_ttl_default``
only. Phase 2 adds:

* **rate** — token-bucket throttle per sender
  (``rule.limits.rate.per_minute`` / ``burst``).
* **delegation_depth** — monotonically-increasing hop counter on
  envelopes, enforced at ``post_envelope`` time.

Tokens-per-hour and cost-per-day are *stored* on the rule but Phase 2
does not enforce them — the design (§14) defers enforcement to Phase 4
(tasks), which is where LLM usage attribution becomes first-class.
Those two fields round-trip here for completeness so Phase 4 lands
additive instead of re-shaping the rule.
"""

from __future__ import annotations

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessBlock,
    ActorIdentity,
    Envelope,
    Hub,
    LimitsBlock,
    RateBlock,
    Rule,
    SessionType,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.errors import LimitExceededError
from autogen.beta.network.hub._limits import _TokenBucket, RateLimiter

from test.beta.network._harness import (
    FakeClient,
    attach_hub_to_link,
    auto_ack_only,
)


# ---------------------------------------------------------------------------
# Unit tests — token bucket math
# ---------------------------------------------------------------------------


def test_token_bucket_allows_up_to_burst_then_refuses() -> None:
    bucket = _TokenBucket.from_rate(RateBlock(per_minute=60, burst=3), clock=0.0)
    assert bucket.consume(clock=0.0) is True
    assert bucket.consume(clock=0.0) is True
    assert bucket.consume(clock=0.0) is True
    assert bucket.consume(clock=0.0) is False


def test_token_bucket_refills_over_time() -> None:
    bucket = _TokenBucket.from_rate(RateBlock(per_minute=60, burst=1), clock=0.0)
    assert bucket.consume(clock=0.0) is True
    assert bucket.consume(clock=0.0) is False
    # 60/min = 1 per second. Advance clock 1 second → bucket refilled.
    assert bucket.consume(clock=1.0) is True


def test_rate_limiter_rebuilds_bucket_when_rate_changes() -> None:
    limiter = RateLimiter()
    # Very tight: 2 bursts, then throttle.
    low = RateBlock(per_minute=60, burst=2)
    assert limiter.check_and_consume("a", low, clock=0.0) is True
    assert limiter.check_and_consume("a", low, clock=0.0) is True
    assert limiter.check_and_consume("a", low, clock=0.0) is False
    # Rule change lifts the ceiling — bucket is rebuilt fresh.
    wide = RateBlock(per_minute=6000, burst=10)
    assert limiter.check_and_consume("a", wide, clock=0.0) is True
    assert limiter.check_and_consume("a", wide, clock=0.0) is True


def test_rate_limiter_disabled_when_per_minute_zero() -> None:
    limiter = RateLimiter()
    rate = RateBlock(per_minute=0, burst=0)
    # Many calls all pass because rate limiting is disabled.
    for _ in range(100):
        assert limiter.check_and_consume("a", rate, clock=0.0) is True


# ---------------------------------------------------------------------------
# Rule serialization
# ---------------------------------------------------------------------------


def test_rate_block_round_trips_through_rule_dict() -> None:
    rule = Rule(
        limits=LimitsBlock(
            rate=RateBlock(per_minute=120, burst=5),
            tokens_per_hour=500_000,
            cost_per_day_usd=7.5,
        )
    )
    data = rule.to_dict()
    assert data["limits"]["rate"] == {"per_minute": 120, "burst": 5}
    assert data["limits"]["tokens_per_hour"] == 500_000
    assert data["limits"]["cost_per_day_usd"] == 7.5
    restored = Rule.from_dict(data)
    assert restored.limits.rate.per_minute == 120
    assert restored.limits.rate.burst == 5
    assert restored.limits.tokens_per_hour == 500_000
    assert restored.limits.cost_per_day_usd == 7.5


def test_envelope_depth_round_trips() -> None:
    env = Envelope.text(session_id="s", sender_id="a", content="x")
    env.depth = 4
    restored = Envelope.from_dict(env.to_dict())
    assert restored.depth == 4


# ---------------------------------------------------------------------------
# Hub integration — rate limit rejection
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


@pytest.mark.asyncio
async def test_hub_rate_limit_rejects_excess_posts(hub: Hub) -> None:
    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(rate=RateBlock(per_minute=60, burst=2))),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        # Two posts drain the bucket (capacity=2).
        for i in range(2):
            env = Envelope.text(
                session_id=meta.session_id,
                sender_id=alice.actor_id or "",
                content=f"m{i}",
                recipient_id=bob.actor_id,
            )
            await hub.post_envelope(env)

        # Third post within the same window is throttled.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="third",
            recipient_id=bob.actor_id,
        )
        with pytest.raises(LimitExceededError, match="rate limit"):
            await hub.post_envelope(env)
    finally:
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_hub_rate_limit_does_not_affect_other_senders(hub: Hub) -> None:
    """Rate state is per-actor, not global."""

    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(rate=RateBlock(per_minute=60, burst=1))),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        # Alice burns her one-token bucket.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="alice",
            recipient_id=bob.actor_id,
        )
        await hub.post_envelope(env)
        # Bob has no rate limit and can post freely.
        for i in range(5):
            env = Envelope.text(
                session_id=meta.session_id,
                sender_id=bob.actor_id or "",
                content=f"bob{i}",
                recipient_id=alice.actor_id,
            )
            await hub.post_envelope(env)
    finally:
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# Delegation depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegation_depth_enforcement(hub: Hub) -> None:
    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(delegation_depth=3)),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        # depth ≤ limit passes.
        for depth in (0, 1, 2, 3):
            env = Envelope.text(
                session_id=meta.session_id,
                sender_id=alice.actor_id or "",
                content=f"d{depth}",
                recipient_id=bob.actor_id,
            )
            env.depth = depth
            await hub.post_envelope(env)

        # depth > limit rejects.
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="too deep",
            recipient_id=bob.actor_id,
        )
        env.depth = 4
        with pytest.raises(LimitExceededError, match="delegation_depth"):
            await hub.post_envelope(env)
    finally:
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_delegation_depth_zero_disables_enforcement(hub: Hub) -> None:
    """A delegation_depth of 0 lets callers send at any depth."""

    alice = await hub.register(
        ActorIdentity(name="alice"),
        rule=Rule(limits=LimitsBlock(delegation_depth=0)),
    )
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONVERSATION,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        env = Envelope.text(
            session_id=meta.session_id,
            sender_id=alice.actor_id or "",
            content="deep",
            recipient_id=bob.actor_id,
        )
        env.depth = 100
        await hub.post_envelope(env)  # does not raise
    finally:
        await b.stop()
        await link.close()
