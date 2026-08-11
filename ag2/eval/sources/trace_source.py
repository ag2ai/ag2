# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TraceSource — where the evaluator gets traces from.

The trace-based evaluator grades a :class:`Trace`; it does not run an agent.
A :class:`TraceSource` is the seam between "grade a trace" and "where the
trace's spans physically live" — an in-memory batch from a just-finished run,
a directory of files on disk, or (via a user implementation) a cloud
observability backend. ``list`` enumerates what's available (cheaply, for
backends where loading is expensive); ``load`` materializes one :class:`Trace`.

Two backends ship here, both SDK-free (they build a :class:`Trace` via the
pure ``spans_to_trace`` adapter, never the OpenTelemetry SDK):

* :class:`InMemoryTraceSource` — already-built traces, for the live
  produce-then-evaluate path.
* :class:`DirectoryTraceSource` — a directory of one-trace-per-file JSON span
  documents. The on-disk shape is **provisional** pending the cross-team OTLP
  interchange decision (see ``design/research/otel-trace-eval.md``); use
  :func:`save_trace` to write it so reader and writer stay in lockstep.
"""

import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..trace import Trace
from ._spans import SpanConvention, SpanData, span_data_from_dict, span_data_to_dict, spans_to_trace

__all__ = (
    "DirectoryTraceSource",
    "InMemoryTraceSource",
    "TraceRef",
    "TraceSource",
    "save_trace",
)


@dataclass(frozen=True, slots=True)
class TraceRef:
    """A handle to one trace in a source, plus the bits needed to grade it.

    ``task_id`` joins the trace back to a :class:`~ag2.eval.Task` in a
    :class:`~ag2.eval.Suite` (for reference-based scorers); it is
    ``None`` for traces with no associated dataset task (e.g. captured
    production traffic, graded reference-free).
    """

    trace_id: str
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TraceSource(Protocol):
    """A backend that supplies traces for evaluation."""

    def list(self) -> AsyncIterator[TraceRef]:
        """Yield a :class:`TraceRef` for each available trace (cheap; no payload)."""
        ...

    async def load(self, ref: TraceRef) -> Trace:
        """Materialize the :class:`Trace` for ``ref``."""
        ...


class InMemoryTraceSource:
    """A :class:`TraceSource` over already-built traces held in memory.

    Used by the live produce-then-evaluate path: a producer captures spans,
    reconstructs each :class:`Trace`, and hands them here — no disk round-trip.
    """

    def __init__(self, traces: Sequence[tuple[TraceRef, Trace]]) -> None:
        self._refs: tuple[TraceRef, ...] = tuple(ref for ref, _ in traces)
        self._by_id: dict[str, Trace] = {ref.trace_id: trace for ref, trace in traces}

    async def list(self) -> AsyncIterator[TraceRef]:
        for ref in self._refs:
            yield ref

    async def load(self, ref: TraceRef) -> Trace:
        return self._by_id[ref.trace_id]


class DirectoryTraceSource:
    """A :class:`TraceSource` over a directory of one-trace-per-file JSON spans.

    Each ``<trace_id>.json`` holds ``{"trace_id", "task_id"?, "metadata"?,
    "spans": [...]}`` where each span is a :func:`span_data_to_dict` object.
    Write the files with :func:`save_trace`. Format is provisional.
    """

    def __init__(self, path: str | os.PathLike[str], *, conventions: Sequence[SpanConvention] | None = None) -> None:
        self._path = Path(path)
        self._conventions = conventions

    async def list(self) -> AsyncIterator[TraceRef]:
        for file in sorted(self._path.glob("*.json")):
            doc = json.loads(file.read_text(encoding="utf-8"))
            yield TraceRef(
                trace_id=doc.get("trace_id", file.stem),
                task_id=doc.get("task_id"),
                metadata=dict(doc.get("metadata", {})),
            )

    async def load(self, ref: TraceRef) -> Trace:
        doc = json.loads((self._path / f"{ref.trace_id}.json").read_text(encoding="utf-8"))
        spans = [span_data_from_dict(s) for s in doc.get("spans", [])]
        return spans_to_trace(spans, conventions=self._conventions)


def save_trace(
    directory: str | os.PathLike[str],
    trace_id: str,
    spans: Sequence[SpanData],
    *,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``spans`` as one ``<trace_id>.json`` under ``directory`` (creating it).

    The companion writer for :class:`DirectoryTraceSource`. Returns the path written.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    doc = {
        "trace_id": trace_id,
        "task_id": task_id,
        "metadata": dict(metadata or {}),
        "spans": [span_data_to_dict(s) for s in spans],
    }
    file = path / f"{trace_id}.json"
    file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return file
