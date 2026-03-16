# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autogen import AgentDescriptionGuard, AgentEligibilityPolicy, SelectionContext


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
    mixin = AgentDescriptionGuard(agent)
    mixin.mark_unavailable()
    assert agent.description.startswith("[UNAVAILABLE]")
    assert "A helpful planner" in agent.description


def test_description_restore_on_available():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = AgentDescriptionGuard(agent)
    mixin.mark_unavailable()
    mixin.mark_available()
    assert agent.description == "A helpful planner"


def test_double_mark_unavailable_idempotent():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = AgentDescriptionGuard(agent)
    mixin.mark_unavailable()
    mixin.mark_unavailable()
    assert agent.description.count("[UNAVAILABLE]") == 1


def test_mark_available_noop_when_not_marked():
    agent = MagicMock()
    agent.description = "A helpful planner"
    mixin = AgentDescriptionGuard(agent)
    mixin.mark_available()
    assert agent.description == "A helpful planner"


class TestAdversarialEligibilityPolicy:
    """Adversarial tests -- attacker mindset."""

    def test_selection_context_last_speaker_not_in_participants(self):
        """last_speaker name absent from participants is a valid but inconsistent ctx state.
        The dataclass must accept it without validation (callers are responsible)."""
        ctx = SelectionContext(round=2, last_speaker="ghost", participants=("alice", "bob"))
        assert ctx.last_speaker == "ghost"
        assert "ghost" not in ctx.participants

    def test_guard_thundering_herd_mark_unavailable(self):
        """100 threads all calling mark_unavailable simultaneously.
        Exactly one [UNAVAILABLE] prefix must appear -- not zero, not two."""
        import threading

        agent = MagicMock()
        agent.description = "original"
        guard = AgentDescriptionGuard(agent)
        errors: list[Exception] = []

        def call():
            try:
                guard.mark_unavailable()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thundering herd raised: {errors}"
        assert agent.description.count("[UNAVAILABLE]") == 1
        assert "original" in agent.description

    def test_description_mutation_none_description(self):
        """Agent with description=None must not crash mark_unavailable."""
        agent = MagicMock()
        agent.description = None
        mixin = AgentDescriptionGuard(agent)
        mixin.mark_unavailable()
        assert "[UNAVAILABLE]" in agent.description

    def test_description_mutation_empty_string(self):
        """Agent with description='' must get [UNAVAILABLE] prefix."""
        agent = MagicMock()
        agent.description = ""
        mixin = AgentDescriptionGuard(agent)
        mixin.mark_unavailable()
        assert agent.description.startswith("[UNAVAILABLE]")

    def test_mark_available_after_none_description(self):
        """Restoring after None description: original None is preserved on restore."""
        agent = MagicMock()
        agent.description = None
        mixin = AgentDescriptionGuard(agent)
        mixin.mark_unavailable()
        # mark_unavailable stores the original None; mark_available restores it.
        mixin.mark_available()
        assert agent.description is None

    def test_mark_available_noop_on_none_description(self):
        """mark_available on agent with description=None that was never marked unavailable."""
        agent = MagicMock()
        agent.description = None
        guard = AgentDescriptionGuard(agent)
        guard.mark_available()  # no-op, must not crash
        assert agent.description is None or agent.description == ""

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

        lock = threading.Lock()
        call_count = 0
        errors = []

        class _CountingPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                nonlocal call_count
                with lock:
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


def test_description_containing_unavailable_substring_not_stripped():
    """R13 String parse: description containing '[UNAVAILABLE]' as substring (not prefix)
    must NOT be stripped by mark_available."""
    agent = MagicMock()
    agent.description = "Agent handles [UNAVAILABLE] status checks"
    guard = AgentDescriptionGuard(agent)

    # mark_available on unmarked agent with substring -- must not strip mid-string match
    guard.mark_available()
    assert agent.description == "Agent handles [UNAVAILABLE] status checks"

    # mark_unavailable adds prefix, mark_available removes only the prefix
    guard.mark_unavailable()
    assert agent.description == "[UNAVAILABLE] Agent handles [UNAVAILABLE] status checks"
    guard.mark_available()
    assert agent.description == "Agent handles [UNAVAILABLE] status checks"


def test_description_external_modification_preserved():
    """External code modifying description between mark_unavailable/mark_available
    must not lose the external change."""
    agent = MagicMock()
    agent.description = "original"
    guard = AgentDescriptionGuard(agent)
    guard.mark_unavailable()
    assert agent.description == "[UNAVAILABLE] original"

    # External code appends to the description while guard is active
    agent.description = "[UNAVAILABLE] original (updated by external code)"

    guard.mark_available()
    # The external modification after the prefix must be preserved
    assert agent.description == "original (updated by external code)"


def test_description_mutation_thread_safety():
    """Concurrent mark_unavailable/mark_available must not corrupt description."""
    import threading

    agent = MagicMock()
    agent.description = "original"
    mixin = AgentDescriptionGuard(agent)
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
    # Final state: either original or unavailable -- not corrupted
    assert agent.description in ("original", "[UNAVAILABLE] original")
    # No double-prefix from concurrent toggle
    assert agent.description.count("[UNAVAILABLE]") <= 1


def test_selection_context_rejects_str_participants():
    """participants='alice' (bare str) must raise TypeError, not iterate chars."""
    with pytest.raises(TypeError, match="not a str"):
        SelectionContext(round=1, last_speaker=None, participants="alice")


def test_mark_available_strips_external_prefix():
    """When an external source adds the [UNAVAILABLE] prefix (not this guard),
    mark_available still strips it -- structural check, not state-based."""
    agent = MagicMock()
    agent.description = "[UNAVAILABLE] injected by external code"
    guard = AgentDescriptionGuard(agent)
    # guard._original_description is _UNSET -- guard never called mark_unavailable
    guard.mark_available()
    assert agent.description == "injected by external code"
