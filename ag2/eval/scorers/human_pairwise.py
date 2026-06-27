# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Human pairwise comparators — the same A-vs-B unit, decided by a person.

Two modes (both are :class:`PairwiseComparator`, interchangeable with
``pairwise_judge``):

* **Offline** — :func:`export_pairwise_cases` writes a *blinded* JSONL manifest
  (Response 1/2 order randomized and recorded as ``first_variant``); a person /
  UI fills in ``preferred`` per line; :func:`human_labels` reads it back and
  de-blinds to an a/b/tie outcome. Scales, UI-agnostic, and is the calibration
  workflow (run alongside ``pairwise_judge`` and compare with
  ``PairwiseRunResult.agreement``).
* **Inline** — :func:`human_pairwise` prompts during the run via an ``ask``
  callback (terminal by default), randomizing + de-blinding per case.

Position handling for humans is a single blinded randomized order (asking twice
is wasteful and itself biasing) — unlike the LLM judge's dual-order swap.
"""

import inspect
import json
import random
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from ag2.events import ModelResponse

from ..dataset import Suite, Task
from ..pairwise import PairwiseComparator, PairwiseOutcome
from ..sources import TraceRef, TraceSource
from ..trace import Trace

__all__ = (
    "export_pairwise_cases",
    "human_labels",
    "human_pairwise",
)

# A human's per-case answer and the renderer it's shown: returns "1" / "2" / "tie".
AskHuman = Callable[[Task, str, str], "str | Awaitable[str]"]


async def export_pairwise_cases(
    source_a: TraceSource,
    source_b: TraceSource,
    *,
    criteria: Iterable[str],
    out: str,
    suite: Suite | None = None,
    seed: int | None = None,
) -> Path:
    """Write a blinded JSONL labeling manifest for the paired traces.

    One line per (task, criterion): ``{case_id, task_id, criterion, task_input,
    response_1, response_2, first_variant}``. ``first_variant`` records which
    variant is Response 1 (de-blinding key — a labeling UI must not show it).
    A labeler adds ``preferred`` ("1"/"2"/"tie") per line; feed the result to
    :func:`human_labels`.
    """
    criteria = list(criteria)
    rng = random.Random(seed)
    tasks_by_id = {task.task_id: task for task in suite} if suite is not None else {}

    refs_a = [ref async for ref in source_a.list()]
    b_by_task: dict[str, TraceRef] = {}
    async for ref in source_b.list():
        if ref.task_id is not None:
            b_by_task[ref.task_id] = ref

    lines: list[dict[str, Any]] = []
    for ref_a in refs_a:
        if ref_a.task_id is None or ref_a.task_id not in b_by_task:
            continue
        answer_a = _final_text(await source_a.load(ref_a))
        answer_b = _final_text(await source_b.load(b_by_task[ref_a.task_id]))
        task = tasks_by_id.get(ref_a.task_id) or Task(task_id=ref_a.task_id, inputs={})
        first_variant = rng.choice(["a", "b"])
        response_1, response_2 = (answer_a, answer_b) if first_variant == "a" else (answer_b, answer_a)
        for criterion in criteria:
            lines.append({
                "case_id": f"{ref_a.task_id}::{criterion}",
                "task_id": ref_a.task_id,
                "criterion": criterion,
                "task_input": task.inputs.get("input"),
                "response_1": response_1,
                "response_2": response_2,
                "first_variant": first_variant,
            })

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def human_labels(path: str, *, criterion: str, key: str) -> PairwiseComparator:
    """A comparator reading human labels for one criterion from a manifest JSONL."""
    return _HumanLabels(path, criterion, key)


def human_pairwise(*, key: str, ask: AskHuman | None = None, seed: int | None = None) -> PairwiseComparator:
    """An inline comparator that asks a human per case (terminal ``ask`` by default)."""
    return _HumanInline(key, ask or _terminal_ask, random.Random(seed))


class _HumanLabels:
    """Offline human comparator: reads de-blinded labels lazily on first use."""

    def __init__(self, path: str, criterion: str, key: str) -> None:
        self.key = key
        self._path = Path(path)
        self._criterion = criterion
        self._by_task: dict[str, tuple[str, Any]] | None = None

    def _labels(self) -> dict[str, tuple[str, Any]]:
        if self._by_task is None:
            by_task: dict[str, tuple[str, Any]] = {}
            for raw in self._path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if record.get("criterion") != self._criterion:
                    continue
                by_task[record["task_id"]] = (record.get("first_variant", "a"), record.get("preferred"))
            self._by_task = by_task
        return self._by_task

    async def compare(
        self, *, task: Task, trace_a: Trace, trace_b: Trace, reference_outputs: dict[str, Any] | None
    ) -> PairwiseOutcome:
        entry = self._labels().get(task.task_id)
        if entry is None or entry[1] is None:
            return PairwiseOutcome(winner="tie", reasoning="no human label", detail={"missing": True})
        first_variant, preferred = entry
        return PairwiseOutcome(
            winner=_deblind(preferred, first_variant),
            reasoning="human label",
            detail={"preferred": preferred, "first_variant": first_variant},
        )


class _HumanInline:
    """Inline human comparator: blinded single presentation via an ask callback."""

    def __init__(self, key: str, ask: AskHuman, rng: random.Random) -> None:
        self.key = key
        self._ask = ask
        self._rng = rng

    async def compare(
        self, *, task: Task, trace_a: Trace, trace_b: Trace, reference_outputs: dict[str, Any] | None
    ) -> PairwiseOutcome:
        answer_a, answer_b = _final_text(trace_a), _final_text(trace_b)
        first_variant = self._rng.choice(["a", "b"])
        response_1, response_2 = (answer_a, answer_b) if first_variant == "a" else (answer_b, answer_a)
        preferred = self._ask(task, response_1, response_2)
        if inspect.isawaitable(preferred):
            preferred = await preferred
        return PairwiseOutcome(
            winner=_deblind(preferred, first_variant),
            reasoning="human (inline)",
            detail={"preferred": preferred, "first_variant": first_variant},
        )


def _deblind(preferred: Any, first_variant: str) -> str:
    other = "b" if first_variant == "a" else "a"
    if str(preferred) == "1":
        return first_variant
    if str(preferred) == "2":
        return other
    return "tie"


def _final_text(trace: Trace) -> str:
    responses = trace.events_of(ModelResponse)
    if responses and responses[-1].content is not None:
        return responses[-1].content
    return "(no answer)"


def _terminal_ask(task: Task, response_1: str, response_2: str) -> str:
    print(f"\nTask: {task.inputs.get('input')}\n[1] {response_1}\n[2] {response_2}")
    return input("Which is better? 1 / 2 / tie: ").strip() or "tie"
