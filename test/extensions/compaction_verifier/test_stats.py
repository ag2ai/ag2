# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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
        # fifths are off the default grid of thirds, so the grid is refined
        assert exact_permutation_test([0.2, 0.2, 0.2, 0.2]) == pytest.approx(2 / 16)

    def test_permutation_test_symmetric_null(self) -> None:
        assert exact_permutation_test([0.5, -0.5]) == 1.0
        assert exact_permutation_test([]) == 1.0
