# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.extensions.compaction_verifier import (
    Action,
    arm_burden,
    burden,
    call_signature,
    history_signatures,
    score_boundary,
)


def act(name: str, args: str = "{}", *, blocked: bool = False, call_id: str = "") -> Action:
    return Action(
        call_id=call_id or f"{name}{args}",
        name=name,
        arguments=args,
        signature=call_signature(name, args),
        blocked=blocked,
    )


class TestBurden:
    def test_refetch_of_a_call_executed_before_the_boundary(self) -> None:
        history = history_signatures([act("login", '{"u": "a"}')])

        b = burden([act("login", '{"u": "a"}'), act("fetch", '{"i": 1}')], history, 5)

        assert (b.blocked, b.refetch, b.wasted, b.actions) == (0, 1, 1, 2)

    def test_a_loop_inside_the_rollout_is_a_refetch(self) -> None:
        b = burden([act("f", '{"i": 1}'), act("f", '{"i": 1}'), act("f", '{"i": 1}')], frozenset(), 5)

        assert b.refetch == 2

    def test_a_step_that_is_blocked_and_a_refetch_is_wasted_once(self) -> None:
        history = history_signatures([act("f")])

        b = burden([act("f", blocked=True)], history, 5)

        assert (b.blocked, b.refetch, b.wasted) == (1, 1, 1)

    def test_blocked_calls_enter_the_seen_set(self) -> None:
        b = burden([act("f", blocked=True), act("f")], frozenset(), 5)

        assert (b.blocked, b.refetch, b.wasted) == (1, 1, 2)

    def test_only_the_first_horizon_actions_count(self) -> None:
        rollout = [act("f", '{"i": 1}'), act("g", blocked=True), act("g", blocked=True)]

        assert burden(rollout, frozenset(), 1).wasted == 0
        assert burden(rollout, frozenset(), 2).wasted == 1
        assert burden(rollout, frozenset(), 5).actions == 3

    def test_a_rollout_shorter_than_the_horizon_stopped(self) -> None:
        assert burden([act("f")], frozenset(), 3).stopped
        assert not burden([act("f")], frozenset(), 1).stopped
        assert burden([], frozenset(), 1).stopped

    def test_tool_key_ignores_arguments(self) -> None:
        history = history_signatures([act("page", '{"n": 1}')], "tool")

        assert burden([act("page", '{"n": 2}')], history, 5, key="tool").refetch == 1
        assert burden([act("page", '{"n": 2}')], history_signatures([act("page", '{"n": 1}')]), 5).refetch == 0


class TestScoreBoundary:
    def test_delta_is_post_minus_pre_mean(self) -> None:
        history = history_signatures([act("login")])
        pre = [[act("fetch", '{"i": 1}')], [act("fetch", '{"i": 1}')]]
        post = [[act("login")], [act("fetch", '{"i": 1}', blocked=True)]]

        (d,) = score_boundary(pre, post, history, [1])

        assert d.pre.wasted == 0.0
        assert d.post.wasted == 1.0
        assert (d.blocked, d.refetch, d.wasted) == (0.5, 0.5, 1.0)
        assert d.harm == 1.0

    def test_harm_clips_each_channel_so_one_cannot_cancel_the_other(self) -> None:
        history = history_signatures([act("login")])
        pre = [[act("fetch", blocked=True)]]
        post = [[act("login")]]

        (d,) = score_boundary(pre, post, history, [1])

        assert (d.blocked, d.refetch, d.wasted) == (-1.0, 1.0, 0.0)
        assert d.harm == 1.0

    def test_stopping_early_is_its_own_channel(self) -> None:
        # the POST agent quits: it wastes nothing, and that is what stopped reports
        pre = [[act("f", '{"i": 1}'), act("f", '{"i": 2}')]] * 2
        post = [[act("f", '{"i": 1}')], []]

        (d,) = score_boundary(pre, post, frozenset(), [2])

        assert d.wasted == 0.0
        assert d.stopped == 1.0
        assert d.post.actions == 0.5

    def test_one_delta_per_horizon(self) -> None:
        deltas = score_boundary([[act("a")]], [[act("a")]], frozenset(), range(1, 4))

        assert [d.horizon for d in deltas] == [1, 2, 3]

    def test_an_empty_arm_cannot_be_scored(self) -> None:
        with pytest.raises(ValueError, match="at least one rollout"):
            arm_burden([], frozenset(), 1)
