# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Standardized logistic probe that scores each turn as "is this the mistake?"."""

import torch

__all__ = ("LogisticProbe",)

_STD_EPSILON = 1e-6


class LogisticProbe:
    """A linear probe over standardized activations.

    Features are standardized with the training mean and standard deviation,
    then scored as ``x @ weight + bias``. Higher scores mean "more likely the
    decisive mistake"; attribution takes the arg-max over a conversation's turns.

    Args:
        weight: ``(d,)`` weight vector over standardized features.
        bias: ``(1,)`` bias.
        mean: ``(1, d)`` training feature mean.
        std: ``(1, d)`` training feature standard deviation (epsilon included).
    """

    __slots__ = ("bias", "mean", "std", "weight")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        epochs: int = 300,
        lr: float = 0.05,
        l2: float = 1e-3,
    ) -> "LogisticProbe":
        """Train a probe with full-batch AdamW on class-weighted logistic loss.

        The rare positive class (the mistake turn) is up-weighted by the
        negative-to-positive ratio. Training runs on ``features.device``.

        Args:
            features: ``(n, d)`` float activations, one row per turn.
            labels: ``(n,)`` float labels, ``1.0`` for mistake turns.
            epochs: Optimization steps.
            lr: AdamW learning rate.
            l2: Coefficient of the explicit squared-norm penalty on ``weight``.

        Returns:
            The trained probe.
        """
        features = features.float()
        labels = labels.to(device=features.device, dtype=torch.float32)
        mean = features.mean(0, keepdim=True)
        std = features.std(0, keepdim=True) + _STD_EPSILON
        standardized = (features - mean) / std

        with torch.enable_grad():
            weight = torch.zeros(features.shape[1], device=features.device, requires_grad=True)
            bias = torch.zeros(1, device=features.device, requires_grad=True)
            positives = labels.sum().clamp(min=1)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(len(labels) - positives) / positives)
            optimizer = torch.optim.AdamW([weight, bias], lr=lr)
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(standardized @ weight + bias, labels) + l2 * (weight @ weight)
                loss.backward()
                optimizer.step()

        return cls(weight.detach(), bias.detach(), mean, std)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit per row of ``features`` (moved to the probe's device)."""
        features = features.to(device=self.weight.device, dtype=torch.float32)
        return ((features - self.mean) / self.std) @ self.weight + self.bias

    def to(self, device: torch.device | str) -> "LogisticProbe":
        """Return a copy of the probe on ``device``."""
        return LogisticProbe(self.weight.to(device), self.bias.to(device), self.mean.to(device), self.std.to(device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the probe tensors, on CPU, keyed for :meth:`from_state_dict`."""
        return {
            "weight": self.weight.cpu().contiguous(),
            "bias": self.bias.cpu().contiguous(),
            "mean": self.mean.cpu().contiguous(),
            "std": self.std.cpu().contiguous(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "LogisticProbe":
        """Rebuild a probe from :meth:`state_dict` output."""
        return cls(state["weight"], state["bias"], state["mean"], state["std"])
