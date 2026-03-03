from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autogen.agentchat.eligibility_policy import (
    AgentEligibilityPolicy,
    DescriptionMutationMixin,
    SelectionContext,
)


class _AlwaysEligible:
    def is_eligible(self, agent, ctx: SelectionContext) -> bool:
        return True


class _NeverEligible:
    def is_eligible(self, agent, ctx: SelectionContext) -> bool:
        return False


def test_selection_context_fields():
    ctx = SelectionContext(round=1, last_speaker="alice", participants=("alice", "bob"))
    assert ctx.round == 1
    assert ctx.last_speaker == "alice"
    assert ctx.participants == ("alice", "bob")


def test_selection_context_no_last_speaker():
    ctx = SelectionContext(round=0, last_speaker=None, participants=("alice",))
    assert ctx.last_speaker is None


def test_selection_context_frozen():
    ctx = SelectionContext(round=1, last_speaker=None, participants=("alice",))
    with pytest.raises((AttributeError, TypeError)):
        ctx.round = 2  # type: ignore[misc]


def test_always_eligible_satisfies_protocol():
    policy: AgentEligibilityPolicy = _AlwaysEligible()
    ctx = SelectionContext(round=1, last_speaker=None, participants=("alice",))
    assert policy.is_eligible(object(), ctx) is True


def test_never_eligible_satisfies_protocol():
    policy: AgentEligibilityPolicy = _NeverEligible()
    ctx = SelectionContext(round=1, last_speaker=None, participants=("alice",))
    assert policy.is_eligible(object(), ctx) is False


def test_runtime_checkable_isinstance():
    assert isinstance(_AlwaysEligible(), AgentEligibilityPolicy)


def test_description_mutation_on_unavailable():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = DescriptionMutationMixin(agent)
    mixin.mark_unavailable()
    assert agent.description.startswith("[UNAVAILABLE]")
    assert "A helpful planner" in agent.description


def test_description_restore_on_available():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = DescriptionMutationMixin(agent)
    mixin.mark_unavailable()
    mixin.mark_available()
    assert agent.description == "A helpful planner"


def test_double_mark_unavailable_idempotent():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = DescriptionMutationMixin(agent)
    mixin.mark_unavailable()
    mixin.mark_unavailable()
    assert agent.description.count("[UNAVAILABLE]") == 1


def test_mark_available_noop_when_not_marked():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = DescriptionMutationMixin(agent)
    mixin.mark_available()
    assert agent.description == "A helpful planner"


class TestAdversarialEligibilityPolicy:
    """Adversarial tests — attacker mindset."""

    def test_policy_raises_exception_propagates(self):
        """Policy that raises should propagate, not silently pass."""

        class _RaisingPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                raise RuntimeError("policy failure")

        policy = _RaisingPolicy()
        ctx = SelectionContext(round=0, last_speaker=None, participants=("a",))
        with pytest.raises(RuntimeError, match="policy failure"):
            policy.is_eligible(object(), ctx)

    def test_description_mutation_none_description(self):
        """Agent with description=None must not crash mark_unavailable."""
        agent = MagicMock()
        agent.description = None
        mixin = DescriptionMutationMixin(agent)
        mixin.mark_unavailable()
        assert "[UNAVAILABLE]" in agent.description

    def test_description_mutation_empty_string(self):
        """Agent with description='' must get [UNAVAILABLE] prefix."""
        agent = MagicMock()
        agent.description = ""
        mixin = DescriptionMutationMixin(agent)
        mixin.mark_unavailable()
        assert agent.description.startswith("[UNAVAILABLE]")

    def test_mark_available_after_none_description(self):
        """Restoring after None description: original was treated as '' so restores to ''."""
        agent = MagicMock()
        agent.description = None
        mixin = DescriptionMutationMixin(agent)
        mixin.mark_unavailable()
        # mark_unavailable stores "" (via `description or ""`), so restore is ""
        mixin.mark_available()
        assert agent.description == ""

    def test_selection_context_participants_empty_tuple(self):
        """SelectionContext with empty participants tuple is valid."""
        ctx = SelectionContext(round=0, last_speaker=None, participants=())
        assert ctx.participants == ()

    def test_selection_context_negative_round(self):
        """Negative round index is technically allowed (no validation in dataclass)."""
        ctx = SelectionContext(round=-1, last_speaker=None, participants=("a",))
        assert ctx.round == -1

    def test_concurrent_is_eligible_calls(self):
        """Concurrent calls to is_eligible must not corrupt state (thread safety)."""
        import threading

        call_count = 0
        errors = []

        class _CountingPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                nonlocal call_count
                call_count += 1
                return True

        policy = _CountingPolicy()
        ctx = SelectionContext(round=1, last_speaker=None, participants=("a",))

        def call_policy():
            try:
                policy.is_eligible(object(), ctx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_policy) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent calls raised: {errors}"
        assert call_count == 50


def test_description_mutation_thread_safety():
    """Concurrent mark_unavailable/mark_available must not corrupt description."""
    import threading

    agent = MagicMock()
    agent.description = "original"
    mixin = DescriptionMutationMixin(agent)
    errors = []

    def toggle():
        try:
            for _ in range(100):
                mixin.mark_unavailable()
                mixin.mark_available()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=toggle) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Final state: either original or unavailable — not corrupted
    assert agent.description in ("original", "[UNAVAILABLE] original")
