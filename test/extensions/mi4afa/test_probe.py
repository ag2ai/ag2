# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from ag2.extensions.mi4afa import LogisticProbe


def _separable(rows: int = 400, dims: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(rows, dims, generator=generator) * 5 + 3
    labels = (features[:, 2] > 9).float()  # rare positives, decided by one feature
    return features, labels


def test_probe_separates_a_linearly_separable_rare_class() -> None:
    features, labels = _separable()
    assert 0 < labels.sum() < len(labels) * 0.2

    probe = LogisticProbe.fit(features, labels)

    predictions = probe.score(features) > 0
    assert (predictions == labels.bool()).float().mean() > 0.95
    assert probe.weight.abs().argmax() == 2


def test_positive_class_is_up_weighted() -> None:
    """With one positive among many near-identical negatives, the positive still scores highest."""
    features = torch.zeros(50, 4)
    features[:, 0] = torch.linspace(0, 1, 50)
    features[17, 1] = 1.0
    labels = torch.zeros(50)
    labels[17] = 1.0

    probe = LogisticProbe.fit(features, labels)

    assert int(probe.score(features).argmax()) == 17
    assert float(probe.score(features)[17]) > 0


def test_fit_accepts_inference_mode_tensors() -> None:
    features, labels = _separable(rows=64)
    with torch.inference_mode():
        frozen = features * 1.0

    probe = LogisticProbe.fit(frozen, labels, epochs=5)

    assert probe.score(features).shape == (64,)


def test_fit_is_deterministic() -> None:
    features, labels = _separable(rows=64)
    first = LogisticProbe.fit(features, labels, epochs=20)
    second = LogisticProbe.fit(features, labels, epochs=20)
    torch.testing.assert_close(first.weight, second.weight)


def test_state_dict_round_trip() -> None:
    features, labels = _separable(rows=64)
    probe = LogisticProbe.fit(features, labels, epochs=20)

    restored = LogisticProbe.from_state_dict(probe.state_dict())

    torch.testing.assert_close(restored.score(features), probe.score(features))
    assert set(probe.state_dict()) == {"weight", "bias", "mean", "std"}


def test_score_accepts_bfloat16_features() -> None:
    features, labels = _separable(rows=64)
    probe = LogisticProbe.fit(features, labels, epochs=20)

    scores = probe.score(features.bfloat16())

    assert scores.dtype == torch.float32
