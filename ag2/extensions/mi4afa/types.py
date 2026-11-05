# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain data types shared by the mi4afa probing pipeline.

Nothing here imports ``torch``, so conversations can be built and converted
from AG2 traces without the probing dependencies installed.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = (
    "ActivationSite",
    "Component",
    "Conversation",
    "FitReport",
    "SiteScore",
    "Turn",
)

Component: TypeAlias = Literal["resid_pre", "mlp_in", "resid_final"]
"""Where in the transformer an activation is read.

* ``resid_pre`` — the residual stream entering decoder layer ``layer``
  (``hidden_states[layer]`` in Hugging Face terms; layer 0 is the embeddings).
* ``mlp_in`` — the input to layer ``layer``'s MLP: the post-attention residual
  after that layer's pre-MLP normalization.
* ``resid_final`` — the final hidden state after the model's last norm
  (``hidden_states[num_layers]``); ``layer`` equals ``num_layers``.
"""


@dataclass(frozen=True, slots=True)
class Turn:
    """One step of a multi-agent conversation.

    Attributes:
        name: The agent (or tool) that produced the step.
        content: The step's text.
    """

    name: str
    content: str


@dataclass(frozen=True, slots=True)
class Conversation:
    """A multi-agent conversation to attribute, optionally with its label.

    Attributes:
        question: The task the agents were solving.
        ground_truth: The correct final answer, shown to the probed model.
        history: The conversation steps, in order.
        mistake_step: Index into ``history`` of the decisive mistake, when
            known. Required for training and evaluation.
        mistake_agent: Name of the agent responsible for the mistake, when known.
    """

    question: str
    ground_truth: str
    history: tuple[Turn, ...]
    mistake_step: int | None = None
    mistake_agent: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(self.history))
        if not self.history:
            raise ValueError("a conversation needs at least one turn")
        if self.mistake_step is not None and not 0 <= self.mistake_step < len(self.history):
            raise ValueError(f"mistake_step {self.mistake_step} is outside the {len(self.history)}-turn history")

    @classmethod
    def from_who_and_when(cls, record: Mapping[str, Any]) -> "Conversation":
        """Build a conversation from a Who&When-format record.

        The record carries ``question``, ``ground_truth`` and a ``history`` of
        ``{"name", "content"}`` mappings, plus the optional labels
        ``mistake_step`` and ``mistake_agent``. Other keys are ignored.

        Args:
            record: One entry of a Who&When (or Who&When-Pro) dataset.

        Returns:
            The equivalent :class:`Conversation`.
        """
        mistake_step = record.get("mistake_step")
        return cls(
            question=str(record["question"]),
            ground_truth=str(record["ground_truth"]),
            history=_turns(record["history"]),
            mistake_step=int(mistake_step) if mistake_step is not None else None,
            mistake_agent=record.get("mistake_agent"),
        )


@dataclass(frozen=True, slots=True, order=True)
class ActivationSite:
    """A location in the model whose activations a probe reads.

    Attributes:
        layer: Decoder layer index (see :data:`Component` for its meaning per component).
        component: Which activation of that layer.
    """

    layer: int
    component: Component

    def __str__(self) -> str:
        return f"L{self.layer}.{self.component}"


@dataclass(frozen=True, slots=True)
class SiteScore:
    """Attribution accuracy of a probe at one site over a set of conversations.

    Attributes:
        site: The probed site.
        agent_accuracy: Fraction of conversations whose responsible agent was identified.
        step_accuracy: Fraction of conversations whose decisive step was identified exactly.
        count: Number of conversations scored.
    """

    site: ActivationSite
    agent_accuracy: float
    step_accuracy: float
    count: int


@dataclass(frozen=True, slots=True)
class FitReport:
    """Outcome of :meth:`ProbeAttributor.fit`.

    Attributes:
        site: The site selected on the validation split.
        validation: Validation scores of every candidate site, in site order.
        train_count: Conversations the final probe was trained on.
        validation_count: Conversations used to select the site.
    """

    site: ActivationSite
    validation: tuple[SiteScore, ...] = field(repr=False)
    train_count: int
    validation_count: int


def _turns(history: Iterable[Mapping[str, Any]]) -> tuple[Turn, ...]:
    return tuple(
        Turn(name=str(entry.get("name", "Unknown Agent")), content=str(entry.get("content", ""))) for entry in history
    )
