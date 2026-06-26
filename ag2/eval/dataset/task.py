# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task — one unit of evaluation, loaded from a dataset."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

__all__ = ("Task",)


@dataclass(frozen=True, slots=True)
class Task:
    """A single task in an evaluation suite.

    Tasks are typically loaded from JSONL via :meth:`Suite.from_jsonl` or
    built inline via :meth:`Suite.from_list`. The runner passes
    ``inputs["input"]`` to ``agent.ask(...)``; every other field is plumbed
    through to scorers unchanged.

    Args:
        task_id: Stable identifier for this task. Auto-generated as
            ``"task-{index:04d}"`` by ``Suite.from_*`` when the source
            dict omits one.
        inputs: The task's input payload. Must contain at least an
            ``"input"`` key — that string is the user prompt the agent is asked.
        reference_outputs: Expected outputs, consumed by reference-based
            scorers (e.g. ``final_answer_matches``). A dict; a Pydantic model or
            dataclass (e.g. a ``response_schema`` instance) is accepted and
            coerced to a dict. ``None`` for tasks scored reference-free.
        tags: Free-form labels, useful for filtering or slicing
            (``"happy-path"``, ``"adversarial"``).
        metadata: Anything else the dataset carries — surfaces in the
            run JSON so scorers and reports can consume it.
    """

    inputs: dict[str, Any]

    task_id: str = field(default_factory=lambda: str(uuid4()))
    reference_outputs: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_outputs", _coerce_reference_outputs(self.reference_outputs))


def _coerce_reference_outputs(value: object) -> dict[str, Any] | None:
    """Normalise ``reference_outputs`` to a plain dict (or ``None``).

    A Pydantic model or dataclass — e.g. a ``response_schema`` instance — is dumped to a
    dict, so it stores as data and reads correctly in scorers (which treat it as a mapping).
    Anything that is not a mapping, model, dataclass, or ``None`` raises ``TypeError`` rather
    than being accepted and then scoring silently wrong.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "reference_outputs must be a dict, a Pydantic model / dataclass, or None; "
        f"got {type(value).__name__}. Pass e.g. {{'answer': 'Paris'}} or your_model.model_dump()."
    )
