# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Resampling and exact tests over per-boundary deltas. Standard library only.

Burden deltas live on a coarse rational grid (differences of means of small
integer counts), so the paired tests here are exact rather than Monte Carlo:
the sign-flip null is obtained by integer convolution, and a p-value carries no
simulation noise.
Bootstrap draws come from a seeded :class:`random.Random`, so every interval is
reproducible bit for bit.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd, lcm

__all__ = (
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_RESAMPLES",
    "Interval",
    "bootstrap_mean",
    "exact_permutation_test",
    "exact_sign_test",
    "median",
)

DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260803

# A difference of means over arms of up to 100 rollouts has a denominator of at
# most 100 * 100; a value that needs more is not a burden delta.
_MAX_DENOMINATOR = 10_000


@dataclass(frozen=True, slots=True)
class Interval:
    """A mean with its percentile-bootstrap 95% interval over ``n`` units."""

    mean: float
    low: float
    high: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Interval:
    """Mean of ``values`` with a 95% percentile-bootstrap interval.

    The unit resampled is one value (one boundary), so boundaries are treated as
    independent. Boundaries cut from the same recording share one agent run and
    are correlated, so with several per recording the interval is narrower than
    it should be. The same holds for the p-values of :func:`exact_sign_test` and
    :func:`exact_permutation_test`, which also take each boundary as one unit.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rnd = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    draws = sorted(sum(values[rnd.randrange(n)] for _ in values) / n for _ in range(resamples))
    return Interval(mean=mean, low=draws[int(0.025 * resamples)], high=draws[int(0.975 * resamples) - 1], n=n)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of an empty sample")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def exact_sign_test(diffs: Sequence[float]) -> float:
    """Two-sided exact binomial sign test p-value; zero differences are dropped."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(1 for d in nonzero if d > 0)
    tail = sum(comb(n, i) for i in range(min(positive, n - positive) + 1))
    return min(1.0, 2 * tail / (1 << n))


def exact_permutation_test(diffs: Sequence[float]) -> float:
    """Two-sided exact sign-flip permutation p-value for a mean of paired differences.

    Under the null each difference's sign is flipped independently, giving
    ``2**n`` equally likely sums. A burden delta is a difference of two means of
    integer counts, so each difference is recovered as an exact fraction; all of
    them are then placed on one integer grid, the least common multiple of their
    denominators. Nothing is rounded, so the null distribution is convolved
    exactly and the p-value is exact for any number of samples per arm, equal or
    not.

    Raises:
        ValueError: A difference is not a fraction with a denominator of at most
            ``10_000``, so there is no grid the test could be exact on.
    """
    if not diffs:
        return 1.0
    fractions = [_as_fraction(d) for d in diffs]
    scale = lcm(*(f.denominator for f in fractions))
    grid = [f.numerator * (scale // f.denominator) for f in fractions]
    unit = gcd(*grid)
    if unit == 0:
        return 1.0  # every difference is zero: every sign pattern ties the observed sum
    return _sign_flip_p([g // unit for g in grid])


def _as_fraction(value: float) -> Fraction:
    exact = Fraction(value).limit_denominator(_MAX_DENOMINATOR)
    if abs(float(exact) - value) > 1e-9 * max(1.0, abs(value)):
        raise ValueError(f"{value!r} is not a difference of means of counts; the permutation test needs one")
    return exact


def _sign_flip_p(grid: Sequence[int]) -> float:
    """``P(|sum of randomly signed grid values| >= |observed sum|)``, by exact convolution."""
    observed = abs(sum(grid))
    if observed == 0:
        return 1.0
    values = [abs(v) for v in grid if v]  # a zero has both signs equal: it doubles every count and cancels
    span = sum(values)
    counts = [0] * (2 * span + 1)  # counts[s + span] = sign patterns whose sum is s
    counts[span] = 1
    reach = 0
    for v in values:
        step = [0] * len(counts)
        for i in range(span - reach, span + reach + 1):
            c = counts[i]
            if c:
                step[i - v] += c
                step[i + v] += c
        counts = step
        reach += v
    tail = sum(counts[: span - observed + 1]) + sum(counts[span + observed :])
    return tail / (1 << len(values))
