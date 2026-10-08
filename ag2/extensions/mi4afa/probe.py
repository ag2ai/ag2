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

        Raises:
            ValueError: If there are fewer than two rows, the labels lack a mistake
                row or a row that is not one, the features or their mean and
                standard deviation are not finite, or training diverges.
        """
        features = features.float()
        labels = labels.to(device=features.device, dtype=torch.float32)
        if len(features) < 2:
            raise ValueError(f"a probe needs at least two training rows, got {len(features)}")
        positives = int(labels.sum())
        if positives == 0 or positives == len(labels):
            raise ValueError(
                f"a probe needs mistake and non-mistake rows, got {positives} of {len(labels)} rows labelled "
                "as the mistake; one-turn conversations have no non-mistake turn"
            )
        if not torch.isfinite(features).all():
            raise ValueError("training features contain NaN or infinity")
        mean = features.mean(0, keepdim=True)
        std = features.std(0, keepdim=True) + _STD_EPSILON
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError(
                "training features are too large to standardize: their mean or standard deviation overflows"
            )
        standardized = (features - mean) / std

        with torch.enable_grad():
            weight = torch.zeros(features.shape[1], device=features.device, requires_grad=True)
            bias = torch.zeros(1, device=features.device, requires_grad=True)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=labels.new_tensor((len(labels) - positives) / positives))
            optimizer = torch.optim.AdamW([weight, bias], lr=lr)
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(standardized @ weight + bias, labels) + l2 * (weight @ weight)
                loss.backward()
                optimizer.step()

        weight, bias = weight.detach(), bias.detach()
        if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError("probe training diverged to a non-finite weight; try a lower lr")
        return cls(weight, bias, mean, std)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit per row of ``features`` (moved to the probe's device).

        Raises:
            ValueError: If a logit is NaN or infinite, so no turn can be ranked.
        """
        features = features.to(device=self.weight.device, dtype=torch.float32)
        scores = ((features - self.mean) / self.std) @ self.weight + self.bias
        if not torch.isfinite(scores).all():
            raise ValueError("probe scores contain NaN or infinity; check the activations and the probe tensors")
        return scores

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
