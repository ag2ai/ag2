# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
from fractions import Fraction

import pytest

from ag2.extensions.compaction_verifier import bootstrap_mean, exact_permutation_test, exact_sign_test


class TestBootstrapMean:
    def test_is_reproducible_for_a_seed(self) -> None:
        values = [0.0, 0.2, 0.4, 1.0, -0.2, 0.6]

        assert bootstrap_mean(values) == bootstrap_mean(values)
        assert bootstrap_mean(values, seed=3) == bootstrap_mean(values, seed=3)

    def test_interval_brackets_the_mean(self) -> None:
        est = bootstrap_mean([0.0, 0.2, 0.4, 1.0, -0.2, 0.6])

        assert est.mean == pytest.approx(2.0 / 6)
        assert est.low <= est.mean <= est.high
        assert est.n == 6

    def test_a_constant_sample_has_a_degenerate_interval(self) -> None:
        est = bootstrap_mean([0.5] * 8)

        assert (est.mean, est.low, est.high) == (0.5, 0.5, 0.5)
        assert est.excludes_zero

    def test_empty_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean([])


class TestExactTests:
    def test_sign_test_all_positive(self) -> None:
        # 5 of 5 positive: two-sided p = 2 * (1/32)
        assert exact_sign_test([0.2, 0.4, 0.2, 1.0, 0.6]) == pytest.approx(0.0625)

    def test_sign_test_drops_zeros(self) -> None:
        assert exact_sign_test([0.0, 0.0, 0.2]) == 1.0
        assert exact_sign_test([0.0, 0.0]) == 1.0

    def test_permutation_test_is_exact(self) -> None:
        # three equal positive differences: 2 of the 8 sign patterns reach |sum| = 3
        assert exact_permutation_test([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.25)

    def test_permutation_test_on_a_finer_grid(self) -> None:
        assert exact_permutation_test([0.2, 0.2, 0.2, 0.2]) == pytest.approx(2 / 16)

    def test_permutation_test_symmetric_null(self) -> None:
        assert exact_permutation_test([0.5, -0.5]) == 1.0
        assert exact_permutation_test([0.0, 0.0]) == 1.0
        assert exact_permutation_test([]) == 1.0

    def test_permutation_test_rejects_values_off_any_count_grid(self) -> None:
        with pytest.raises(ValueError, match="difference of means"):
            exact_permutation_test([0.123456789])


def _brute_force_p(exact: list[Fraction]) -> float:
    """Enumerate every sign pattern, in exact arithmetic: the reference the test compares against."""
    scale = math.lcm(*(d.denominator for d in exact))
    ints = [d.numerator * (scale // d.denominator) for d in exact]
    observed = abs(sum(ints))
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=len(ints))
        if abs(sum(s * v for s, v in zip(signs, ints))) >= observed
    )
    return hits / 2 ** len(ints)


def _boundary_delta(rnd: random.Random, n_pre: int, n_post: int, horizon: int = 5) -> tuple[float, Fraction]:
    """One boundary's wasted delta, as the verifier computes it (float) and exactly (Fraction).

    The float follows ``ArmBurden`` / ``BoundaryDelta``: each arm's mean of
    per-rollout integer counts, then POST minus PRE.
    """
    pre = [rnd.randint(0, horizon) for _ in range(n_pre)]
    post = [rnd.randint(0, horizon) for _ in range(n_post)]
    as_float = sum(post) / n_post - sum(pre) / n_pre
    exact = Fraction(sum(post), n_post) - Fraction(sum(pre), n_pre)
    return as_float, exact


class TestPermutationTestAgainstBruteForce:
    """Differential test against full enumeration, for the cases the review found wrong."""

    @pytest.mark.parametrize("samples", [2, 3, 4, 5])
    @pytest.mark.parametrize("arms", ["equal", "unequal"])
    def test_matches_enumeration(self, samples: int, arms: str) -> None:
        rnd = random.Random(samples * 100 + (arms == "unequal"))
        for _ in range(300):
            n = rnd.randint(1, 9)
            sizes = [
                (samples, samples) if arms == "equal" else (rnd.randint(1, samples), rnd.randint(1, samples))
                for _ in range(n)
            ]
            deltas = [_boundary_delta(rnd, n_pre, n_post) for n_pre, n_post in sizes]

            got = exact_permutation_test([f for f, _ in deltas])

            assert got == _brute_force_p([e for _, e in deltas]), (sizes, deltas)

    @pytest.mark.parametrize(
        ("deltas", "expected"),
        [
            # the review's counterexamples: samples=4, equal arms
            ([-0.5, 0.75, 0.0, -0.25, -0.25], 1.0),
            ([0.5, 0.25, 0.25, -0.75], 1.0),
            ([-0.5, -0.5, 0.25, -0.75, 0.25], 0.375),
            # and unequal arms, (pre, post) rollouts (4, 3), (4, 4), (3, 4): POST mean minus PRE mean
            ([0 / 3 - 2 / 4, 1 / 4 - 3 / 4, 1 / 4 - 1 / 3], 0.25),
        ],
    )
    def test_review_counterexamples(self, deltas: list[float], expected: float) -> None:
        assert exact_permutation_test(deltas) == expected

    def test_a_large_cohort_with_mixed_arm_sizes_is_exact_and_quick(self) -> None:
        rnd = random.Random(7)
        deltas = [_boundary_delta(rnd, rnd.randint(1, 5), rnd.randint(1, 5)) for _ in range(40)]

        p = exact_permutation_test([f for f, _ in deltas])

        # too many patterns to enumerate (2**40); check against the exact null built the slow way on 16 of them
        assert 0.0 < p <= 1.0
        head = deltas[:16]
        assert exact_permutation_test([f for f, _ in head]) == _brute_force_p([e for _, e in head])
