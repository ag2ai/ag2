# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task — one unit of evaluation, loaded from a dataset."""

from dataclasses import dataclass, field
from typing import Any

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
            scorers (e.g. ``final_answer_matches``). ``None`` for tasks
            scored reference-free.
        tags: Free-form labels, useful for filtering or slicing
            (``"happy-path"``, ``"adversarial"``).
        metadata: Anything else the dataset carries — surfaces in the
            run JSON so scorers and reports can consume it.
    """

    task_id: str
    inputs: dict[str, Any]
    reference_outputs: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
