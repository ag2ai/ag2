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
    ctx = SelectionContext(round=1, last_speaker="alice", participants=["alice", "bob"])
    assert ctx.round == 1
    assert ctx.last_speaker == "alice"
    assert ctx.participants == ["alice", "bob"]


def test_selection_context_no_last_speaker():
    ctx = SelectionContext(round=0, last_speaker=None, participants=["alice"])
    assert ctx.last_speaker is None


def test_selection_context_frozen():
    ctx = SelectionContext(round=1, last_speaker=None, participants=["alice"])
    with pytest.raises((AttributeError, TypeError)):
        ctx.round = 2  # type: ignore[misc]


def test_always_eligible_satisfies_protocol():
    policy: AgentEligibilityPolicy = _AlwaysEligible()
    ctx = SelectionContext(round=1, last_speaker=None, participants=["alice"])
    assert policy.is_eligible(object(), ctx) is True


def test_never_eligible_satisfies_protocol():
    policy: AgentEligibilityPolicy = _NeverEligible()
    ctx = SelectionContext(round=1, last_speaker=None, participants=["alice"])
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
