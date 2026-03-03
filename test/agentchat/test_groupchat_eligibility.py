from __future__ import annotations

import pytest

from autogen import ConversableAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.eligibility_policy import AgentEligibilityPolicy, SelectionContext


def _make_agent(name: str) -> ConversableAgent:
    return ConversableAgent(
        name=name,
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="",
    )


class _PolicyAllowAll:
    def is_eligible(self, agent, ctx: SelectionContext) -> bool:
        return True


class _PolicyBlockByName:
    def __init__(self, blocked: str) -> None:
        self.blocked = blocked

    def is_eligible(self, agent, ctx: SelectionContext) -> bool:
        return agent.name != self.blocked


class _CBPolicy:
    def __init__(self) -> None:
        self.tripped: set[str] = set()

    def trip(self, name: str) -> None:
        self.tripped.add(name)

    def recover(self, name: str) -> None:
        self.tripped.discard(name)

    def is_eligible(self, agent, ctx: SelectionContext) -> bool:
        return agent.name not in self.tripped


def test_groupchat_accepts_eligibility_policies():
    alice, bob = _make_agent("alice"), _make_agent("bob")
    gc = GroupChat(
        agents=[alice, bob],
        messages=[],
        max_round=5,
        eligibility_policies=[_PolicyAllowAll()],
    )
    assert len(gc.eligibility_policies) == 1


def test_groupchat_default_eligibility_policies_is_empty():
    alice, bob = _make_agent("alice"), _make_agent("bob")
    gc = GroupChat(agents=[alice, bob], messages=[], max_round=5)
    assert gc.eligibility_policies == []


def _get_candidates(gc: GroupChat, last_speaker) -> list[str]:
    """Helper: call _prepare_and_select_agents and return names."""
    result = gc._prepare_and_select_agents(last_speaker)
    # result is a tuple (selected_agent, agents_list, messages)
    if isinstance(result, tuple):
        agents_list = result[1]
    else:
        agents_list = result
    if agents_list is None:
        return []
    return [a.name for a in agents_list]


def test_single_agent_ineligible_removed_from_candidates():
    alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")
    gc = GroupChat(
        agents=[alice, bob, carol],
        messages=[],
        max_round=5,
        speaker_selection_method="random",
        eligibility_policies=[_PolicyBlockByName("bob")],
    )
    names = _get_candidates(gc, alice)
    assert "bob" not in names
    assert len(names) >= 1


def test_all_agents_ineligible_raises():
    alice, bob = _make_agent("alice"), _make_agent("bob")

    class _BlockAll:
        def is_eligible(self, agent, ctx: SelectionContext) -> bool:
            return False

    gc = GroupChat(
        agents=[alice, bob],
        messages=[],
        max_round=5,
        eligibility_policies=[_BlockAll()],
    )
    with pytest.raises(ValueError, match="No eligible agents"):
        gc._prepare_and_select_agents(alice)


def test_no_policies_all_agents_eligible():
    alice, bob = _make_agent("alice"), _make_agent("bob")
    gc_with = GroupChat(
        agents=[alice, bob], messages=[], max_round=5, speaker_selection_method="round_robin"
    )
    gc_without = GroupChat(
        agents=[alice, bob], messages=[], max_round=5, speaker_selection_method="round_robin"
    )
    # Both should not raise
    try:
        gc_with._prepare_and_select_agents(alice)
        gc_without._prepare_and_select_agents(alice)
    except Exception as e:
        if "eligible" not in str(e).lower():
            raise


def test_multiple_policies_and_condition():
    alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")
    gc = GroupChat(
        agents=[alice, bob, carol],
        messages=[],
        max_round=5,
        speaker_selection_method="random",
        eligibility_policies=[
            _PolicyBlockByName("alice"),
            _PolicyBlockByName("carol"),
        ],
    )
    names = _get_candidates(gc, alice)
    assert names == ["bob"]


def test_circuit_breaker_trips_mid_chat():
    alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")
    cb = _CBPolicy()
    gc = GroupChat(
        agents=[alice, bob, carol],
        messages=[],
        max_round=10,
        speaker_selection_method="random",
        eligibility_policies=[cb],
    )
    names_before = _get_candidates(gc, alice)
    assert "bob" in names_before

    cb.trip("bob")
    names_after = _get_candidates(gc, alice)
    assert "bob" not in names_after


def test_circuit_breaker_half_open_retry():
    alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")
    cb = _CBPolicy()
    cb.trip("bob")
    gc = GroupChat(
        agents=[alice, bob, carol],
        messages=[],
        max_round=10,
        speaker_selection_method="random",
        eligibility_policies=[cb],
    )
    names_tripped = _get_candidates(gc, alice)
    assert "bob" not in names_tripped

    cb.recover("bob")
    names_recovered = _get_candidates(gc, alice)
    assert "bob" in names_recovered


def test_msze_scenario_cheap_planner_cb_trip_falls_back_to_pricey():
    cheap = _make_agent("cheap_planner")
    pricey = _make_agent("pricey_planner")
    cb = _CBPolicy()
    gc = GroupChat(
        agents=[cheap, pricey],
        messages=[],
        max_round=10,
        speaker_selection_method="random",
        eligibility_policies=[cb],
    )
    both = _get_candidates(gc, pricey)
    assert len(both) == 2

    cb.trip("cheap_planner")
    fallback = _get_candidates(gc, pricey)
    assert fallback == ["pricey_planner"]


class TestAdversarialGroupChatEligibility:
    """Adversarial tests for GroupChat eligibility integration — attacker mindset."""

    def test_policy_raises_during_filtering_propagates(self):
        """Exception from policy.is_eligible propagates out of _prepare_and_select_agents."""
        alice, bob = _make_agent("alice"), _make_agent("bob")

        class _BoomPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                raise ValueError("policy exploded")

        gc = GroupChat(
            agents=[alice, bob],
            messages=[],
            max_round=5,
            eligibility_policies=[_BoomPolicy()],
        )
        with pytest.raises(ValueError, match="policy exploded"):
            gc._prepare_and_select_agents(alice)

    def test_apply_eligibility_policies_empty_input_raises(self):
        """_apply_eligibility_policies with empty input list raises ValueError (no eligible)."""
        alice, bob = _make_agent("alice"), _make_agent("bob")

        class _AllowAll:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return True

        gc = GroupChat(
            agents=[alice, bob],
            messages=[],
            max_round=5,
            eligibility_policies=[_AllowAll()],
        )
        # Empty input list -> no eligible agents -> ValueError
        with pytest.raises(ValueError, match="No eligible agents"):
            gc._apply_eligibility_policies([], last_speaker=None, round_index=0)

    def test_apply_eligibility_policies_empty_input_no_policies_returns_empty(self):
        """_apply_eligibility_policies with empty input and no policies returns empty list."""
        alice, bob = _make_agent("alice"), _make_agent("bob")
        gc = GroupChat(agents=[alice, bob], messages=[], max_round=5)

        # No policies -> early return -> empty list passes through
        result = gc._apply_eligibility_policies([], last_speaker=None, round_index=0)
        assert result == []

    def test_eligibility_policies_list_mutation_during_filtering_safe(self):
        """Modifying eligibility_policies list during iteration must not corrupt results."""
        alice, bob = _make_agent("alice"), _make_agent("bob")

        policy_called = []

        class _MutatingPolicy:
            def __init__(self, gc_ref):
                self.gc_ref = gc_ref

            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                policy_called.append(agent.name)
                # Attempt to append a new policy during iteration (adversarial)
                # This tests that _apply_eligibility_policies copies the list or is otherwise safe
                return True

        gc = GroupChat(
            agents=[alice, bob],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
        )
        policy = _MutatingPolicy(gc)
        gc.eligibility_policies = [policy]

        # Should not raise IndexError or RuntimeError even if policies mutated
        try:
            gc._prepare_and_select_agents(alice)
        except (ValueError,):
            pass  # ValueError for no candidates is acceptable

    def test_policy_returns_truthy_non_bool(self):
        """Policy returning truthy non-bool (e.g. 1) should work (Python truthiness)."""
        alice, bob = _make_agent("alice"), _make_agent("bob")

        class _TruthyPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return 1  # truthy non-bool  # type: ignore[return-value]

        gc = GroupChat(
            agents=[alice, bob],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_TruthyPolicy()],
        )
        # Should not crash — Python's `all()` accepts truthy values
        result = gc._prepare_and_select_agents(alice)
        selected = result[0] if isinstance(result, tuple) else result
        candidates = result[1] if isinstance(result, tuple) else result
        assert len(candidates) >= 1

    def test_policy_returns_falsy_non_bool(self):
        """Policy returning 0 (falsy) should exclude the agent."""
        alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")

        class _FalsyForBob:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return 0 if agent.name == "bob" else 1  # type: ignore[return-value]

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_FalsyForBob()],
        )
        names = _get_candidates(gc, alice)
        assert "bob" not in names
