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


class _DropUntilLast:
    """Keeps only the final tool round, so neither the token nor the login survive."""

    async def compact(self, events, context, store):  # type: ignore[no-untyped-def]
        return list(await TailWindowCompact(target=5).compact(events, context, store))
