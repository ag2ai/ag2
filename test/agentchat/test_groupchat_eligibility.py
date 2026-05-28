# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from autogen import ConversableAgent, GroupChat, NoEligibleSpeakerError, SelectionContext


def _make_agent(name: str) -> ConversableAgent:
    return ConversableAgent(
        name=name,
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="",
    )


def _make_func_agent(name: str, function_map: dict) -> ConversableAgent:
    return ConversableAgent(
        name=name,
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="",
        function_map=function_map,
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
    agents_list = result[1] if isinstance(result, tuple) else result
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
    with pytest.raises(NoEligibleSpeakerError, match="No eligible agents"):
        gc._prepare_and_select_agents(alice)


def test_no_policies_all_agents_eligible():
    alice, bob = _make_agent("alice"), _make_agent("bob")
    gc = GroupChat(agents=[alice, bob], messages=[], max_round=5, speaker_selection_method="round_robin")
    selected, candidates, _ = gc._prepare_and_select_agents(alice)
    assert len(candidates) >= 1
    assert selected is not None


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
    """Adversarial tests for GroupChat eligibility integration -- attacker mindset."""

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
        # Empty input list -> no eligible agents -> NoEligibleSpeakerError (Bug 1 fix)
        with pytest.raises(NoEligibleSpeakerError, match="No eligible agents"):
            gc._apply_eligibility_policies([], last_speaker=None, round_index=0)

    def test_apply_eligibility_policies_empty_input_no_policies_returns_empty(self):
        """_apply_eligibility_policies with empty input and no policies returns empty list."""
        alice, bob = _make_agent("alice"), _make_agent("bob")
        gc = GroupChat(agents=[alice, bob], messages=[], max_round=5)

        # No policies -> early return -> empty list passes through
        result = gc._apply_eligibility_policies([], last_speaker=None, round_index=0)
        assert result == []

    def test_eligibility_policies_list_mutation_between_rounds_safe(self):
        """Adding a policy to eligibility_policies between rounds does not corrupt state."""
        alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")

        call_log: list[str] = []

        class _LoggingPolicy:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                call_log.append(agent.name)
                return True

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_LoggingPolicy()],
        )

        # Round 1: one policy
        gc._prepare_and_select_agents(alice)
        after_round1 = len(call_log)
        assert after_round1 > 0

        # Add another policy between rounds (runtime mutation)
        gc.eligibility_policies.append(_LoggingPolicy())

        # Round 2: two policies -- should not crash, should call more times
        gc._prepare_and_select_agents(alice)
        assert len(call_log) > after_round1

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
        # Should not crash -- Python's `all()` accepts truthy values
        result = gc._prepare_and_select_agents(alice)
        candidates = result[1] if isinstance(result, tuple) else result
        assert len(candidates) == 2  # both alice and bob pass the truthy policy

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

    def test_transitions_plus_policy_not_bypassed(self):
        """When transition rules constrain to 1 candidate, eligibility policy must still run.

        Bug 2 regression test: single-agent early return was bypassing policies.
        Transition: alice -> bob only. Policy: bob is blocked.
        Expected: NoEligibleSpeakerError (not bob silently selected).
        """
        alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")

        class _BlockBob:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return agent.name != "bob"

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[],
            max_round=5,
            speaker_selection_method="round_robin",
            allowed_or_disallowed_speaker_transitions={alice: [bob]},
            speaker_transitions_type="allowed",
            eligibility_policies=[_BlockBob()],
        )
        with pytest.raises(NoEligibleSpeakerError, match="No eligible agents"):
            gc._prepare_and_select_agents(alice)


def test_callable_speaker_selection_bypasses_policies():
    """When speaker_selection_method is a Callable returning an Agent, eligibility_policies
    are NOT applied -- the caller has explicit control over the selection."""
    alice, bob = _make_agent("alice"), _make_agent("bob")

    class _BlockAll:
        def is_eligible(self, agent, ctx: SelectionContext) -> bool:
            return False

    def _always_alice(last_speaker, gc):  # type: ignore[return-value]
        return alice

    gc = GroupChat(
        agents=[alice, bob],
        messages=[],
        max_round=5,
        speaker_selection_method=_always_alice,
        eligibility_policies=[_BlockAll()],
    )
    # Should NOT raise NoEligibleSpeakerError -- Callable path returns before policy application
    selected, candidates, _ = gc._prepare_and_select_agents(bob)
    assert selected.name == "alice"


class TestAdversarialGroupChatEligibilityDeep:
    """Second-wave adversarial tests -- deeper attacker scenarios."""

    def test_policy_returns_none_excluded(self):
        """Policy returning None (falsy non-bool) must exclude the agent."""
        alice, bob = _make_agent("alice"), _make_agent("bob")

        class _NoneForAlice:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return None if agent.name == "alice" else True  # type: ignore[return-value]

        gc = GroupChat(
            agents=[alice, bob],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_NoneForAlice()],
        )
        names = _get_candidates(gc, bob)
        assert "alice" not in names
        assert "bob" in names

    def test_policy_raises_on_second_agent_propagates(self):
        """Partial failure: first agent passes, second raises -- exception propagates."""
        alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")

        class _RaisesOnBob:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                if agent.name == "bob":
                    raise RuntimeError("bob exploded")
                return True

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_RaisesOnBob()],
        )
        with pytest.raises(RuntimeError, match="bob exploded"):
            gc._prepare_and_select_agents(carol)

    def test_concurrent_prepare_and_select_agents(self):
        """20 threads calling _prepare_and_select_agents simultaneously must not crash."""
        import threading

        agents = [_make_agent(f"agent{i}") for i in range(5)]
        gc = GroupChat(
            agents=agents,
            messages=[],
            max_round=100,
            speaker_selection_method="random",
            eligibility_policies=[_PolicyAllowAll()],
        )
        errors: list[Exception] = []
        results: list[object] = []

        def call() -> None:
            try:
                r = gc._prepare_and_select_agents(agents[0])
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent prepare_and_select raised: {errors}"
        assert len(results) == 20

    def test_single_agent_groupchat_underpopulated_guard_fires_first(self):
        """GroupChat with 1 agent raises ValueError('underpopulated') before policy is applied.
        The built-in guard fires first -- policies are not evaluated."""
        alice = _make_agent("alice")
        gc = GroupChat(
            agents=[alice],
            messages=[],
            max_round=5,
            eligibility_policies=[_PolicyBlockByName("alice")],
        )
        with pytest.raises(ValueError, match="underpopulated"):
            gc._prepare_and_select_agents(alice)

    def test_func_call_filter_singleton_not_early_returned(self):
        """func_call_filter path: policy leaves 2 function executors.
        allow_repeat_speaker=False removes the last_speaker from candidates.
        The remaining agent must be selected -- not the last_speaker."""
        alice = _make_func_agent("alice", {"do_stuff": lambda: None})
        bob = _make_func_agent("bob", {"do_stuff": lambda: None})
        carol = _make_agent("carol")

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            allow_repeat_speaker=False,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyAllowAll()],
        )
        # alice is last_speaker. Both alice and bob can execute do_stuff.
        # allow_repeat_speaker=False removes alice. bob must be selected.
        selected, candidates, _ = gc._prepare_and_select_agents(alice)
        assert selected.name == "bob"

    def test_func_call_filter_policy_blocks_function_agent(self):
        """func_call_filter path: the only function executor is blocked by policy.
        Should raise NoEligibleSpeakerError."""
        alice = _make_func_agent("alice", {"do_stuff": lambda: None})
        bob = _make_agent("bob")

        gc = GroupChat(
            agents=[alice, bob],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyBlockByName("alice")],
        )
        with pytest.raises(NoEligibleSpeakerError, match="No eligible agents"):
            gc._prepare_and_select_agents(bob)

    def test_func_call_filter_fallback_applies_policies(self):
        """func_call_filter fallback path: no agent can execute the specific function,
        so GroupChat falls back to all agents with function_map. Policies must
        still apply on this fallback set."""
        # alice has function_map with "other_func" (not "do_stuff")
        alice = _make_func_agent("alice", {"other_func": lambda: None})
        # bob has function_map with "another_func" (not "do_stuff" either)
        bob = _make_func_agent("bob", {"another_func": lambda: None})
        carol = _make_agent("carol")

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyBlockByName("alice")],
        )
        # No agent can execute "do_stuff" -> fallback to all with function_map (alice, bob).
        # Policy blocks alice -> only bob remains -> bob selected.
        selected, candidates, _ = gc._prepare_and_select_agents(carol)
        assert selected.name == "bob"
        assert len(candidates) == 1

    def test_func_call_filter_fallback_all_blocked_raises(self):
        """Adversarial: func_call_filter fallback path where policy blocks ALL
        agents with function_map. Must raise NoEligibleSpeakerError, not
        silently fall through."""
        alice = _make_func_agent("alice", {"other_func": lambda: None})
        bob = _make_func_agent("bob", {"another_func": lambda: None})
        carol = _make_agent("carol")

        class _BlockBoth:
            def is_eligible(self, agent, ctx):
                return agent.name not in ("alice", "bob")

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            speaker_selection_method="round_robin",
            eligibility_policies=[_BlockBoth()],
        )
        # No agent can execute "do_stuff" -> fallback to function_map holders (alice, bob).
        # Policy blocks both -> NoEligibleSpeakerError.
        with pytest.raises(NoEligibleSpeakerError, match="No eligible agents"):
            gc._prepare_and_select_agents(carol)

    def test_sync_manual_fallback_uses_filtered_agents(self):
        """select_speaker manual fallback must use policy-filtered agent list.

        When manual_select_speaker returns None (user presses 'q' or empty),
        select_speaker falls back to next_agent with the filtered list.
        """
        alice = _make_agent("alice")
        bob = _make_agent("bob")
        carol = _make_agent("carol")

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[{"role": "user", "content": "hello"}],
            max_round=5,
            speaker_selection_method="manual",
            eligibility_policies=[_PolicyBlockByName("bob")],
        )

        selector = _make_agent("selector")
        with patch.object(gc, "manual_select_speaker", return_value=None):
            selected = gc.select_speaker(alice, selector)
        # alice is last_speaker, bob is blocked by policy, so next should be carol
        assert selected.name == "carol"

    def test_async_manual_fallback_uses_filtered_agents(self):
        """a_select_speaker manual fallback must use policy-filtered agent list.

        When manual_select_speaker returns None (user presses 'q' or empty),
        a_select_speaker falls back to next_agent with the filtered list.
        """
        alice = _make_agent("alice")
        bob = _make_agent("bob")
        carol = _make_agent("carol")

        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[{"role": "user", "content": "hello"}],
            max_round=5,
            speaker_selection_method="manual",
            eligibility_policies=[_PolicyBlockByName("bob")],
        )

        # Mock manual_select_speaker to return None (simulates user pressing 'q')
        # This makes _prepare_and_select_agents return (None, agents, None)
        # and a_select_speaker falls through to the manual fallback at L678.
        selector = _make_agent("selector")
        with patch.object(gc, "manual_select_speaker", return_value=None):
            selected = asyncio.run(gc.a_select_speaker(alice, selector))
        # alice is last_speaker, bob is blocked by policy, so next should be carol
        assert selected.name == "carol"

    def test_invalid_policy_rejected_at_construction(self):
        """GroupChat must reject objects that don't implement AgentEligibilityPolicy."""
        alice, bob = _make_agent("alice"), _make_agent("bob")
        with pytest.raises(ValueError, match="eligibility_policies\\[0\\]"):
            GroupChat(
                agents=[alice, bob],
                messages=[],
                max_round=5,
                eligibility_policies=["not_a_policy"],  # type: ignore[list-item]
            )

    def test_policy_filtered_set_is_subset_of_input(self):
        """R2 Invariant: AND semantics must never add agents not in the input set."""
        agents = [_make_agent(f"agent{i}") for i in range(5)]

        class _BlockOdds:
            def is_eligible(self, agent, ctx: SelectionContext) -> bool:
                return not agent.name.endswith(("1", "3"))

        gc = GroupChat(
            agents=agents,
            messages=[],
            max_round=5,
            speaker_selection_method="random",
            eligibility_policies=[_BlockOdds()],
        )
        candidates = _get_candidates(gc, agents[0])
        # Invariant: filtered set is a subset of the full roster
        assert set(candidates).issubset({a.name for a in agents})
        # Invariant: filtered set never contains blocked agents
        assert "agent1" not in candidates
        assert "agent3" not in candidates

    def test_policy_plus_allow_repeat_speaker_false_normal_path(self):
        """R6 Compound: policy + allow_repeat_speaker=False on the normal (non-func_call_filter) path."""
        alice, bob, carol = _make_agent("alice"), _make_agent("bob"), _make_agent("carol")
        gc = GroupChat(
            agents=[alice, bob, carol],
            messages=[],
            max_round=5,
            allow_repeat_speaker=False,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyBlockByName("bob")],
        )
        # alice is last_speaker -> removed by allow_repeat_speaker=False
        # bob is blocked by policy
        # Only carol remains
        selected, candidates, _ = gc._prepare_and_select_agents(alice)
        assert selected.name == "carol"
        assert len(candidates) == 1

    @pytest.mark.parametrize(
        "n_agents,expect_error",
        [
            (1, ValueError),  # below minimum (underpopulated guard)
            (2, None),  # at minimum
            (3, None),  # above minimum
        ],
    )
    def test_agent_count_boundary_triple(self, n_agents, expect_error):
        """R11 Boundary triple: agent count at 1 (below), 2 (at), 3 (above) minimum."""
        agents = [_make_agent(f"a{i}") for i in range(n_agents)]
        gc = GroupChat(
            agents=agents,
            messages=[],
            max_round=5,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyAllowAll()],
        )
        if expect_error:
            with pytest.raises(expect_error, match="underpopulated"):
                gc._prepare_and_select_agents(agents[0])
        else:
            result = gc._prepare_and_select_agents(agents[0])
            assert result[0] is not None or result[1] is not None

    def test_h10_func_call_filter_policies_applied_then_allow_repeat_speaker_empties_candidates(self):
        """H10 regression: func_call_filter path sets _policies_applied=True, then
        allow_repeat_speaker=False removes the only remaining candidate.

        Before the fix: the empty-candidates guard at line 628 was gated on
        `not _policies_applied`, so it was skipped.  The empty list fell through
        to selection methods and produced IndexError or wrong behavior.

        After the fix: an explicit guard after allow_repeat_speaker filtering
        raises NoEligibleSpeakerError when _policies_applied=True and candidates
        is empty.
        """
        # alice is the only agent that can execute do_stuff, and she is the last_speaker.
        # allow_repeat_speaker=False will remove alice from candidates after policies apply.
        # Result: zero candidates -> NoEligibleSpeakerError (not IndexError).
        alice = _make_func_agent("alice", {"do_stuff": lambda: None})
        bob = _make_agent("bob")  # cannot execute do_stuff, no function_map

        gc = GroupChat(
            agents=[alice, bob],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            allow_repeat_speaker=False,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyAllowAll()],
        )
        # alice is last_speaker, is the only do_stuff executor, but allow_repeat_speaker=False
        # removes her.  No candidates remain after the combined filters.
        with pytest.raises(NoEligibleSpeakerError):
            gc._prepare_and_select_agents(alice)

    def test_h10_guard_does_not_fire_when_last_speaker_outside_group(self):
        """H10 guard must NOT raise NoEligibleSpeakerError when last_speaker is outside the
        group.  In that case graph_eligible_agents=[] is the legacy sentinel meaning
        'no transition constraints -- all agents eligible', not 'zero candidates'.

        Regression for the original H10 fix which used to incorrectly fire in this scenario.
        """
        alice = _make_func_agent("alice", {"do_stuff": lambda: None})
        bob = _make_func_agent("bob", {"do_stuff": lambda: None})
        # external_agent is NOT in the GroupChat agents list
        external_agent = _make_agent("external")

        gc = GroupChat(
            agents=[alice, bob],
            messages=[{"role": "assistant", "content": "", "function_call": {"name": "do_stuff"}}],
            max_round=5,
            allow_repeat_speaker=False,
            speaker_selection_method="round_robin",
            eligibility_policies=[_PolicyAllowAll()],
        )
        # external_agent is last_speaker but not in the group -> all agents eligible
        # Both alice and bob can execute do_stuff and neither is the last_speaker
        # -> must not raise
        selected, candidates, _ = gc._prepare_and_select_agents(external_agent)
        assert len(candidates) >= 1
