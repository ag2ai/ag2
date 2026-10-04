# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from ag2.extensions.compaction_verifier import (
    Action,
    BoundaryResult,
    ContextSize,
    Rollout,
    StrategyReport,
    VerificationReport,
    call_signature,
    score_boundary,
)

HORIZON = 2


def ran(arm: str, sample: int) -> Rollout:
    actions = tuple(
        Action(
            call_id=f"{arm}{sample}{i}",
            name="f",
            arguments=f'{{"i": {i}}}',
            signature=call_signature("f", f'{{"i": {i}}}'),
            blocked=False,
        )
        for i in range(HORIZON)
    )
    return Rollout(arm=arm, sample=sample, actions=actions, ending="budget")  # type: ignore[arg-type]


def failed(arm: str, sample: int, error: str = "BadRequestError: orphaned tool result") -> Rollout:
    return Rollout(arm=arm, sample=sample, actions=(), ending="error", error=error)  # type: ignore[arg-type]


def boundary(
    cut: int, pre: list[Rollout], post: list[Rollout], *, trajectory: int = 0, post_tokens: int = 100
) -> BoundaryResult:
    scorable_pre = [r.actions for r in pre if r.ending != "error"]
    scorable_post = [r.actions for r in post if r.ending != "error"]
    deltas = (
        score_boundary(scorable_pre, scorable_post, frozenset(), range(1, HORIZON + 1))
        if scorable_pre and scorable_post
        else ()
    )
    return BoundaryResult(
        trajectory=trajectory,
        cut=cut,
        strategy="s",
        deltas=deltas,
        pre_size=ContextSize(events=20, tokens=400),
        post_size=ContextSize(events=5, tokens=post_tokens),
        pre=tuple(pre),
        post=tuple(post),
        compaction_tokens=0,
        unchanged=False,
    )


def report(strategies: dict[str, list[BoundaryResult]], pre_failed: int = 0, pre_total: int = 0) -> VerificationReport:
    return VerificationReport(
        horizon=HORIZON,
        samples=3,
        strategies={name: StrategyReport.build(name, rs, HORIZON) for name, rs in strategies.items()},
        failed_pre_rollouts=pre_failed,
        pre_rollouts=pre_total,
    )


class TestFailureCounts:
    def test_counts_unscored_boundaries_and_both_arms(self) -> None:
        results = [
            boundary(10, [ran("pre", 0), ran("pre", 1)], [ran("post", 0), failed("post", 1)]),
            boundary(20, [ran("pre", 0), failed("pre", 1)], [failed("post", 0), failed("post", 1)]),
        ]

        s = StrategyReport.build("s", results, HORIZON)

        assert (s.post_rollouts, s.failed_rollouts) == (4, 3)
        assert (s.pre_rollouts, s.failed_pre_rollouts) == (4, 1)
        assert s.unscored_boundaries == 1
        assert s.post_errors == (("BadRequestError: orphaned tool result", 3),)

    def test_failing_more_after_compaction_is_flagged(self) -> None:
        s = StrategyReport.build(
            "s", [boundary(10, [ran("pre", 0), ran("pre", 1)], [ran("post", 0), failed("post", 1)])], HORIZON
        )

        assert s.failure_asymmetry == "post"

    def test_failing_more_without_compaction_is_flagged(self) -> None:
        s = StrategyReport.build(
            "s",
            [
                boundary(
                    10,
                    [ran("pre", 0), failed("pre", 1, "ContextWindowExceeded: too long")],
                    [ran("post", 0), ran("post", 1)],
                )
            ],
            HORIZON,
        )

        assert s.failure_asymmetry == "pre"

    def test_equal_failure_rates_are_not_flagged(self) -> None:
        s = StrategyReport.build(
            "s", [boundary(10, [ran("pre", 0), failed("pre", 1)], [ran("post", 0), failed("post", 1)])], HORIZON
        )

        assert s.failure_asymmetry is None

    def test_errors_are_counted_most_common_first(self) -> None:
        post = [failed("post", 0, "A: x"), failed("post", 1, "B: y"), failed("post", 2, "B: y"), ran("post", 3)]

        s = StrategyReport.build("s", [boundary(10, [ran("pre", 0)], post)], HORIZON)

        assert s.post_errors == (("B: y", 2), ("A: x", 1))


class TestSummary:
    def test_shows_failed_counts_unscored_boundaries_and_pre_failures(self) -> None:
        results = [
            boundary(10, [ran("pre", 0), ran("pre", 1)], [ran("post", 0), failed("post", 1)]),
            boundary(20, [ran("pre", 0), ran("pre", 1)], [failed("post", 0), failed("post", 1)]),
        ]

        lines = report({"tail": results}, pre_failed=0, pre_total=4).summary().splitlines()

        assert lines[1] == "PRE rollouts failed: 0 of 4 (shared by every strategy)"
        row = next(line for line in lines if line.startswith("tail "))
        assert " 1/2 " in row  # boundaries scored / tested
        assert "3/4!" in row  # POST failed / total, flagged
        assert (
            "! tail: failed more often after compaction (POST 3/4 vs PRE 0/4). Failed rollouts are not scored, "
            "so the delta covers only the POST samples that ran and may understate the harm."
        ) in lines
        assert "tail: most common POST error (3x): BadRequestError: orphaned tool result" in lines

    def test_a_strategy_with_every_post_rollout_failed_says_why(self) -> None:
        results = [boundary(10, [ran("pre", 0)], [failed("post", 0), failed("post", 1)])]

        text = report({"tail": results}, pre_total=1).summary()

        assert "no scorable boundaries: 2 of 2 POST and 0 of 1 PRE rollouts failed" in text
        assert "! tail: failed more often after compaction (POST 2/2 vs PRE 0/1). No boundary could be scored." in text
        assert "most common POST error (2x)" in text

    def test_a_clean_run_has_no_failure_notes(self) -> None:
        results = [boundary(10, [ran("pre", 0)], [ran("post", 0)])]

        lines = report({"control": results}, pre_total=1).summary().splitlines()

        row = next(line for line in lines if line.startswith("control "))
        assert "0/1" in row and "!" not in row
        assert not any(line.startswith(("!", "control:")) for line in lines)

    def test_pre_errors_are_reported_once(self) -> None:
        r = VerificationReport(
            horizon=HORIZON,
            samples=1,
            strategies={},
            failed_pre_rollouts=2,
            pre_rollouts=3,
            pre_errors=(("TimeoutError: slow", 2),),
        )

        assert "PRE: most common error (2x): TimeoutError: slow" in r.summary()


class TestCompactionFailures:
    def test_counted_excluded_from_kept_and_flagged(self) -> None:
        ok = boundary(10, [ran("pre", 0)], [ran("post", 0)])
        broken = BoundaryResult(
            trajectory=0,
            cut=20,
            strategy="s",
            deltas=(),
            pre_size=ContextSize(events=20, tokens=400),
            post_size=None,
            pre=(ran("pre", 0),),
            post=(),
            compaction_tokens=0,
            unchanged=False,
            compaction_error="RateLimitError: 429",
        )

        s = StrategyReport.build("s", [ok, broken], HORIZON)

        assert (s.failed_compactions, s.unscored_boundaries) == (1, 1)
        assert s.compaction_errors == (("RateLimitError: 429", 1),)
        assert s.median_token_ratio == 0.25  # from the boundary that compacted only
        lines = report({"s": [ok, broken]}, pre_total=2).summary().splitlines()
        row = next(line for line in lines if line.startswith("s "))
        assert " 1/2 " in row and "0/1!" in row
        assert (
            "! s: compaction failed at 1 of 2 boundaries; no POST rollout ran there and those boundaries are not scored."
            in lines
        )
        assert "s: most common compaction error (1x): RateLimitError: 429" in lines
        d = report({"s": [ok, broken]}, pre_total=2).to_dict()["strategies"]["s"]
        assert d["failed_compactions"] == 1 and d["compaction_errors"] == [["RateLimitError: 429", 1]]
        assert (
            d["boundaries"][1]["post_size"] is None and d["boundaries"][1]["compaction_error"] == "RateLimitError: 429"
        )


class TestSampleNote:
    def test_several_boundaries_per_recording_are_flagged(self) -> None:
        results = [
            boundary(cut, [ran("pre", 0)], [ran("post", 0)], trajectory=t) for cut, t in ((10, 0), (20, 0), (30, 1))
        ]

        r = report({"a": results, "b": results}, pre_total=3)

        assert (r.boundaries, r.recordings) == (3, 2)
        assert (
            "3 boundaries from 2 recordings; intervals and p treat boundaries as independent, but boundaries "
            "from one recording are correlated, so both are optimistic"
        ) in r.summary().splitlines()
        assert (r.to_dict()["boundaries"], r.to_dict()["recordings"]) == (3, 2)

    def test_one_boundary_per_recording_needs_no_caveat(self) -> None:
        results = [boundary(10, [ran("pre", 0)], [ran("post", 0)], trajectory=t) for t in range(2)]

        lines = report({"a": results}, pre_total=2).summary().splitlines()

        assert "2 boundaries from 2 recordings" in lines
        assert not any("optimistic" in line for line in lines)

    def test_singular_wording(self) -> None:
        lines = report({"a": [boundary(10, [ran("pre", 0)], [ran("post", 0)])]}, pre_total=1).summary().splitlines()

        assert "1 boundary from 1 recording" in lines


class TestGrownContext:
    def test_a_strategy_that_lengthens_the_context_gets_a_note(self) -> None:
        results = [
            boundary(10, [ran("pre", 0)], [ran("post", 0)], post_tokens=600),
            boundary(20, [ran("pre", 0)], [ran("post", 0)], post_tokens=520),
            boundary(30, [ran("pre", 0)], [ran("post", 0)], post_tokens=100),
        ]

        s = StrategyReport.build("s", results, HORIZON)
        lines = report({"summarize": results}, pre_total=3).summary().splitlines()

        assert s.grown_boundaries == 2
        assert s.median_token_ratio == 1.3
        assert (
            "summarize: the compacted context was longer than the history it replaced at 2 of 3 boundaries "
            "(median kept 130%)"
        ) in lines
        assert report({"summarize": results}, pre_total=3).to_dict()["strategies"]["summarize"]["grown_boundaries"] == 2

    def test_no_note_when_every_context_shrank(self) -> None:
        results = [boundary(10, [ran("pre", 0)], [ran("post", 0)])]

        lines = report({"tail": results}, pre_total=1).summary().splitlines()

        assert StrategyReport.build("tail", results, HORIZON).grown_boundaries == 0
        assert not any("longer than the history" in line for line in lines)

    def test_failed_compactions_are_not_counted(self) -> None:
        grown = boundary(10, [ran("pre", 0)], [ran("post", 0)], post_tokens=600)
        broken = BoundaryResult(
            trajectory=0,
            cut=20,
            strategy="s",
            deltas=(),
            pre_size=ContextSize(events=20, tokens=400),
            post_size=None,
            pre=(ran("pre", 0),),
            post=(),
            compaction_tokens=0,
            unchanged=False,
            compaction_error="RuntimeError: down",
        )

        lines = report({"s": [grown, broken]}, pre_total=2).summary().splitlines()

        assert (
            "s: the compacted context was longer than the history it replaced at 1 of 1 boundaries (median kept 150%)"
            in lines
        )


class TestToDict:
    def test_failure_fields_are_serialized(self) -> None:
        results = [boundary(10, [ran("pre", 0)], [ran("post", 0), failed("post", 1)])]

        d = report({"tail": results}, pre_total=1).to_dict()

        s = d["strategies"]["tail"]
        assert (s["post_rollouts"], s["failed_rollouts"], s["unscored_boundaries"]) == (2, 1, 0)
        assert s["failure_asymmetry"] == "post"
        assert s["post_errors"] == [["BadRequestError: orphaned tool result", 1]]
        assert d["pre_rollouts"] == 1
        json.dumps(d)
