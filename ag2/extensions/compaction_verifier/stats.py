# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Resampling and exact tests over per-boundary deltas. Standard library only.

Burden deltas live on a coarse rational grid (a mean of small integer counts),
so the paired tests here are exact rather than Monte Carlo: the sign-flip null
is obtained by integer convolution, and a p-value carries no simulation noise.
Bootstrap draws come from a seeded :class:`random.Random`, so every interval is
reproducible bit for bit.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

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

    The unit resampled is one value (one boundary), so boundaries from the same
    trajectory are treated as independent; with few boundaries per trajectory
    this is the usual approximation.
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
    ``2**n`` equally likely sums. The differences are placed on an integer grid
    and the null distribution is convolved exactly, so the p-value is exact.
    """
    if not diffs:
        return 1.0
    scale = 3
    grid: list[int] = []
    for d in diffs:
        f = Fraction(d).limit_denominator(1000) * scale
        if f.denominator != 1:
            # Fold the whole scale into one integer before multiplying: scaling
            # in two steps rounds differently for some floats and moves a grid
            # point, which moves the p-value.
            scale *= f.denominator
            return _sign_flip_p([int(round(x * scale)) for x in diffs])
        grid.append(int(f))
    return _sign_flip_p(grid)


def _sign_flip_p(grid: Sequence[int]) -> float:
    observed = abs(sum(grid))
    counts = {0: 1}
    for v in grid:
        step: dict[int, int] = {}
        for s, c in counts.items():
            step[s + v] = step.get(s + v, 0) + c
            step[s - v] = step.get(s - v, 0) + c
        counts = step
    total = sum(counts.values())
    return sum(c for s, c in counts.items() if abs(s) >= observed) / total
