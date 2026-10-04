# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from ag2 import Agent
from ag2.compact import TailWindowCompact
from ag2.eval import Trace
from ag2.extensions.compaction_verifier import (
    ActionBudget,
    CompactionVerifier,
    IdentityCompact,
    Recording,
    ReplayMismatchError,
    make_boundary,
    resumable_cuts,
    run_rollout,
)
from ag2.testing import TestConfig

from .conftest import ContextAwareConfig, record_trajectory, world_environment


class TestActionBudget:
    def test_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ActionBudget(0)


class TestRunRollout:
    @pytest.mark.asyncio
    async def test_stops_at_the_horizon_and_reports_only_new_actions(self) -> None:
        events = await record_trajectory(fetches=3)
        boundary = make_boundary(events, resumable_cuts(events)[1])  # after login + fetch a
        agent = Agent("worker", config=ContextAwareConfig())

        rollout = await run_rollout(agent, boundary.context, world_environment(), boundary.prefix, horizon=2, arm="pre")

        assert rollout.ending == "budget"
        assert [(a.name, json.loads(a.arguments)["item"]) for a in rollout.actions] == [("fetch", "b"), ("fetch", "c")]
        assert not {a.call_id for a in rollout.actions} & {a.call_id for a in boundary.prefix}

    @pytest.mark.asyncio
    async def test_new_calls_that_reuse_prefix_ids_are_counted(self) -> None:
        events = await record_trajectory(fetches=3, ollama_ids=True)
        boundary = make_boundary(events, resumable_cuts(events)[1])
        agent = Agent("worker", config=ContextAwareConfig(ollama_ids=True))

        rollout = await run_rollout(agent, boundary.context, world_environment(), boundary.prefix, horizon=2, arm="pre")

        assert {a.call_id for a in (*boundary.prefix, *rollout.actions)} == {"call_0"}
        assert [json.loads(a.arguments)["item"] for a in rollout.actions] == ["b", "c"]

    @pytest.mark.asyncio
    async def test_an_agent_that_finishes_early_ends_with_an_answer(self) -> None:
        events = await record_trajectory(fetches=7)
        boundary = make_boundary(events, resumable_cuts(events)[-1])  # 7 of 8 items fetched
        agent = Agent("worker", config=ContextAwareConfig())

        rollout = await run_rollout(agent, boundary.context, world_environment(), boundary.prefix, horizon=5, arm="pre")

        assert rollout.ending == "answer"
        assert len(rollout.actions) == 1

    @pytest.mark.asyncio
    async def test_an_environment_that_cannot_be_restored_is_recorded_not_raised(self) -> None:
        events = await record_trajectory(fetches=2)
        boundary = make_boundary(events, resumable_cuts(events)[-1])
        agent = Agent("worker", config=ContextAwareConfig())

        rollout = await run_rollout(
            agent, boundary.context, _BrokenAt(len(boundary.prefix)), boundary.prefix, horizon=2, arm="pre"
        )

        assert rollout.ending == "error"
        assert rollout.error is not None and rollout.error.startswith("ReplayMismatchError: ")
        assert rollout.actions == ()

    @pytest.mark.asyncio
    async def test_a_failing_run_is_recorded_not_raised(self) -> None:
        events = await record_trajectory(fetches=2)
        boundary = make_boundary(events, resumable_cuts(events)[-1])
        agent = Agent("worker", config=TestConfig(RuntimeError("provider down")))

        rollout = await run_rollout(
            agent, boundary.context, world_environment(), boundary.prefix, horizon=3, arm="post"
        )

        assert rollout.ending == "error"
        assert rollout.error == "RuntimeError: provider down"
        assert rollout.actions == ()


class TestCompactionVerifier:
    @pytest.mark.asyncio
    async def test_identity_scores_zero_and_forgetting_scores_burden(self) -> None:
        events = await record_trajectory(fetches=5)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=3, samples=2)

        report = await verifier.verify(
            [Recording(events, world_environment())],
            {"identity": IdentityCompact(), "tail": TailWindowCompact(target=5)},
            every=2,
            min_prefix=2,
        )

        identity, tail = report.strategies["identity"], report.strategies["tail"]
        assert identity.at(3).boundaries == tail.at(3).boundaries >= 2
        assert identity.at(3).wasted.mean == 0.0
        assert all(b.at(3).wasted == 0.0 for b in identity.boundaries)
        assert tail.at(3).wasted.mean > 0
        assert tail.at(3).refetch.mean > 0
        assert tail.median_token_ratio < 1.0
        assert identity.unchanged_boundaries == len(identity.boundaries)
        assert tail.unchanged_boundaries == 0
        assert report.failed_pre_rollouts == 0
        # two boundaries are too few to mark anything significant
        assert "*" not in report.summary().splitlines()[2]
        assert "without significance marks" in report.summary()

    @pytest.mark.asyncio
    async def test_ollama_ids_verify_exactly_like_unique_ids(self) -> None:
        strategies = {"identity": IdentityCompact(), "tail": TailWindowCompact(target=5)}
        reports = []
        for ollama in (False, True):
            events = await record_trajectory(fetches=5, ollama_ids=ollama)
            verifier = CompactionVerifier(
                Agent("worker", config=ContextAwareConfig(ollama_ids=ollama)), horizon=3, samples=2
            )
            reports.append(
                await verifier.verify([Recording(events, world_environment())], strategies, every=2, min_prefix=2)
            )

        def scores(report):  # type: ignore[no-untyped-def]
            return {
                name: [
                    (b.cut, b.unchanged, [(d.blocked, d.refetch, d.wasted, d.stopped) for d in b.deltas])
                    for b in s.boundaries
                ]
                for name, s in report.strategies.items()
            }

        unique, ollama = reports
        assert scores(ollama) == scores(unique)
        assert ollama.strategies["tail"].at(3).boundaries >= 2
        assert ollama.strategies["tail"].at(3).wasted.mean > 0

    @pytest.mark.asyncio
    async def test_failures_from_a_rejected_compacted_context_are_reported(self) -> None:
        # the window drops the original request; this model, like a strict provider, refuses such a context
        events = await record_trajectory(fetches=5)
        verifier = CompactionVerifier(
            Agent("worker", config=ContextAwareConfig(require_request=True)), horizon=2, samples=2
        )

        report = await verifier.verify(
            [Recording(events, world_environment())],
            {"identity": IdentityCompact(), "tail": TailWindowCompact(target=5)},
            every=2,
            min_prefix=2,
        )

        identity, tail = report.strategies["identity"], report.strategies["tail"]
        assert identity.failed_rollouts == 0 and identity.failure_asymmetry is None
        assert tail.failed_rollouts == tail.post_rollouts > 0
        assert tail.failure_asymmetry == "post"
        assert tail.unscored_boundaries == len(tail.boundaries)
        assert tail.post_errors == (
            ("RuntimeError: provider rejected the context: no user request", tail.post_rollouts),
        )
        text = report.summary()
        assert "no scorable boundaries" in text
        assert "! tail: failed more often after compaction" in text
        assert "provider rejected the context" in text

    @pytest.mark.asyncio
    async def test_a_strategy_that_raises_is_recorded_and_the_rest_still_scores(self) -> None:
        events = await record_trajectory(fetches=5)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=2, samples=2)

        report = await verifier.verify(
            [Recording(events, world_environment())],
            {"identity": IdentityCompact(), "raises": _Raises(RuntimeError("summary call rate-limited"))},
            every=2,
            min_prefix=2,
        )

        identity, raises = report.strategies["identity"], report.strategies["raises"]
        n = len(raises.boundaries)
        assert n >= 2 and identity.at(2).boundaries == n
        assert raises.failed_compactions == raises.unscored_boundaries == n
        assert raises.post_rollouts == 0
        assert raises.compaction_errors == (("RuntimeError: summary call rate-limited", n),)
        assert all(b.compaction_error and b.post_size is None for b in raises.boundaries)
        assert report.pre_rollouts == 2 * n  # PRE still ran, for identity
        text = report.summary()
        row = next(line for line in text.splitlines() if line.startswith("raises "))
        assert row.endswith(f"no scorable boundaries: compaction failed at {n} of {n} boundaries")
        assert f"! raises: compaction failed at {n} of {n} boundaries" in text
        assert f"raises: most common compaction error ({n}x): RuntimeError: summary call rate-limited" in text

    @pytest.mark.asyncio
    async def test_a_strategy_that_drops_the_trigger_is_recorded(self) -> None:
        events = await record_trajectory(fetches=4)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=1, samples=1)

        report = await verifier.verify(
            [Recording(events, world_environment())], {"empty": _Raises(None)}, every=2, min_prefix=2
        )

        (message, _), *_ = report.strategies["empty"].compaction_errors
        assert message.startswith("ValueError: _Raises did not keep the boundary's last ToolResultsEvent")

    @pytest.mark.asyncio
    async def test_a_strategy_failing_at_some_boundaries_scores_the_others(self) -> None:
        events = await record_trajectory(fetches=7)
        cuts = resumable_cuts(events)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=2, samples=1)

        report = await verifier.verify(
            [Recording(events, world_environment(), cuts=[cuts[2], cuts[4], cuts[6]])],
            {"late": _FailsAfter(limit=cuts[3])},
        )

        late = report.strategies["late"]
        assert late.failed_compactions == 2
        assert late.at(2).boundaries == 1
        row = next(line for line in report.summary().splitlines() if line.startswith("late "))
        assert " 1/3 " in row and row.rstrip().split()[-2].endswith("!")

    @pytest.mark.asyncio
    async def test_no_pre_rollouts_run_where_every_strategy_failed_to_compact(self) -> None:
        events = await record_trajectory(fetches=4)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=2, samples=3)

        report = await verifier.verify(
            [Recording(events, world_environment())], {"raises": _Raises(RuntimeError("down"))}, every=2, min_prefix=2
        )

        assert report.pre_rollouts == 0
        n = len(report.strategies["raises"].boundaries)
        assert f"no scorable boundaries: compaction failed at {n} of {n} boundaries" in report.summary()

    @pytest.mark.asyncio
    async def test_a_boundary_whose_environment_cannot_be_restored_is_recorded(self) -> None:
        events = await record_trajectory(fetches=5)
        cuts = resumable_cuts(events)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=2, samples=2)
        broken = make_boundary(events, cuts[3])

        report = await verifier.verify(
            [Recording(events, _BrokenAt(len(broken.prefix)), cuts=[cuts[1], cuts[3]])], {"identity": IdentityCompact()}
        )

        identity = report.strategies["identity"]
        assert identity.unscored_boundaries == 1
        assert identity.at(2).boundaries == 1
        assert report.failed_pre_rollouts == 2 and identity.failed_rollouts == 2
        assert report.pre_errors[0][0].startswith("ReplayMismatchError: ")
        assert "PRE: most common error (2x): ReplayMismatchError" in report.summary()

    @pytest.mark.asyncio
    async def test_each_recording_uses_its_own_environment(self) -> None:
        events = await record_trajectory(fetches=4)
        restored: list[int] = []

        class _Counting:
            def __init__(self, tag: int) -> None:
                self._tag = tag

            async def restore(self, prefix):  # type: ignore[no-untyped-def]
                restored.append(self._tag)
                return await world_environment().restore(prefix)

        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=1, samples=1)
        cut = resumable_cuts(events)[2]

        await verifier.verify(
            [Recording(events, _Counting(1), cuts=[cut]), Recording(events, _Counting(2), cuts=[cut])],
            {"identity": IdentityCompact()},
        )

        assert sorted(restored) == [1, 1, 2, 2]  # PRE + POST per recording

    @pytest.mark.asyncio
    async def test_a_guessed_token_shows_up_as_blocked(self) -> None:
        events = await record_trajectory(fetches=4)
        verifier = CompactionVerifier(
            Agent("worker", config=ContextAwareConfig(guess_token=True)), horizon=2, samples=1
        )
        cut = resumable_cuts(events)[3]

        report = await verifier.verify(
            [Recording(events, world_environment(), cuts=[cut])], {"drop-login": _DropUntilLast()}
        )

        (result,) = report.strategies["drop-login"].boundaries
        assert result.at(1).blocked == 1.0

    @pytest.mark.asyncio
    async def test_accepts_eval_traces_and_serializes(self) -> None:
        events = await record_trajectory(fetches=4)
        trace = Trace(events=events, exception=None, duration_ms=0)
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()), horizon=2, samples=1)

        report = await verifier.verify(
            [Recording(trace, world_environment())], {"identity": IdentityCompact()}, every=2, min_prefix=2
        )

        assert "identity" in report.summary()
        json.dumps(report.to_dict())

    @pytest.mark.asyncio
    async def test_rejects_bad_arguments(self) -> None:
        verifier = CompactionVerifier(Agent("worker", config=ContextAwareConfig()))
        with pytest.raises(ValueError, match="at least one strategy"):
            await verifier.verify([Recording([], world_environment())], {})
        with pytest.raises(ValueError, match="ToolResultsEvent"):
            await verifier.verify([Recording([], world_environment(), cuts=[1])], {"identity": IdentityCompact()})
        with pytest.raises(ValueError):
            CompactionVerifier(Agent("worker", config=ContextAwareConfig()), samples=0)


class _Raises:
    """Raises ``error`` instead of compacting; with ``None``, returns nothing at all."""

    def __init__(self, error: Exception | None) -> None:
        self._error = error

    async def compact(self, events, context, store):  # type: ignore[no-untyped-def]
        if self._error is None:
            return []
        raise self._error


class _FailsAfter:
    """Keeps the last tool round, until the history is longer than ``limit`` events; then raises."""

    def __init__(self, limit: int) -> None:
        self._limit = limit

    async def compact(self, events, context, store):  # type: ignore[no-untyped-def]
        if len(events) > self._limit:
            raise RuntimeError("context too long to summarize")
        return list(await TailWindowCompact(target=5).compact(events, context, store))


class _BrokenAt:
    """An environment that cannot be restored for a prefix of exactly ``size`` calls."""

    def __init__(self, size: int) -> None:
        self._size = size

    async def restore(self, prefix):  # type: ignore[no-untyped-def]
        if len(prefix) == self._size:
            raise ReplayMismatchError("recorded call fetch(...) succeeded but raised KeyError on replay")
        return await world_environment().restore(prefix)


class _DropUntilLast:
    """Keeps only the final tool round, so neither the token nor the login survive."""

    async def compact(self, events, context, store):  # type: ignore[no-untyped-def]
        return list(await TailWindowCompact(target=5).compact(events, context, store))
