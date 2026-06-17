# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Suite — a collection of evaluation tasks, loaded from JSONL or memory.

A Suite is an immutable, iterable container of :class:`Task` records.
Build one with :meth:`Suite.from_jsonl` (load from disk) or
:meth:`Suite.from_list` (inline). Both factories validate each entry
has at least an ``inputs`` dict; missing ``task_id`` values are
auto-filled as ``task-0000`` / ``task-0001`` / … in dataset order.
"""

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .task import Task

__all__ = ("Suite",)


class Suite:
    """An immutable collection of :class:`Task` records.

    Use :meth:`from_jsonl` to load from disk or :meth:`from_list` to
    build inline. Suites are iterable and sized, so they can be passed
    directly to :func:`autogen.beta.eval.run_agent`.
    """

    __slots__ = ("_tasks", "_name", "_source")

    def __init__(
        self,
        tasks: Sequence[Task] = (),
        *,
        name: str = "inline",
        source: str = "inline",
    ) -> None:
        """Direct constructor — most callers should use :meth:`from_jsonl` or :meth:`from_list`."""
        self._tasks: tuple[Task, ...] = tuple(tasks)
        self._name = name
        self._source = source

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike[str]) -> "Suite":
        """Load a Suite from a JSONL file (one task per line).

        Each line must be a JSON object containing at least an
        ``"inputs"`` key (a dict). Missing ``"task_id"`` values are
        filled in from dataset order. Blank lines are skipped.

        Args:
            path: Filesystem path to the JSONL file.

        Returns:
            A Suite whose ``name`` is the filename stem and whose
            ``source`` is the full path string.
        """
        p = Path(path)
        items = _read_jsonl(p)
        tasks = _items_to_tasks(items, source=str(p))
        return cls(tasks=tasks, name=p.stem, source=str(p))

    @classmethod
    def from_list(
        cls,
        items: list[dict[str, Any]],
        *,
        name: str = "inline",
    ) -> "Suite":
        """Build a Suite from a list of dicts (same shape as JSONL lines).

        Args:
            items: One dict per task. Each must contain ``"inputs"`` (a
                dict). Missing ``"task_id"`` values are filled from list
                order.
            name: Optional name for the suite, surfaces in the run JSON.
                Defaults to ``"inline"``.

        Returns:
            A Suite whose ``source`` is ``"inline"``.
        """
        tasks = _items_to_tasks(items, source="inline")
        return cls(tasks=tasks, name=name, source="inline")

    @property
    def name(self) -> str:
        """Human-readable name (filename stem for JSONL, ``"inline"`` otherwise)."""
        return self._name

    @property
    def source(self) -> str:
        """Where the tasks came from — a path string, or ``"inline"``."""
        return self._source

    @property
    def tasks(self) -> tuple[Task, ...]:
        """The tasks in this suite, in dataset order."""
        return self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, with line-numbered errors."""
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for index, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {index} is not valid JSON: {e.msg}") from e
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}: line {index} must be a JSON object, got {type(parsed).__name__}")
            items.append(parsed)
    return items


def _items_to_tasks(items: Sequence[dict[str, Any]], *, source: str) -> tuple[Task, ...]:
    """Validate and convert dict items into :class:`Task` records.

    Each item must have ``inputs`` as a dict; missing ``task_id`` values
    are filled from list order. Returns tasks in input order.
    """
    tasks: list[Task] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{source}: task at index {index} must be a dict, got {type(item).__name__}")
        inputs = item.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"{source}: task at index {index} is missing required field 'inputs' (must be a dict)")
        task_id = item.get("task_id") or f"task-{index:04d}"
        reference_outputs = item.get("reference_outputs")
        tags = tuple(item.get("tags", ()))
        metadata = dict(item.get("metadata", {}))
        tasks.append(
            Task(
                inputs=inputs,
                task_id=task_id,
                reference_outputs=reference_outputs,
                tags=tags,
                metadata=metadata,
            )
        )
    return tuple(tasks)
