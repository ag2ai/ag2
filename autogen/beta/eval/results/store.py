# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run-store serialization — schema version ``"0.1"``.

This is the wire format a future hosted dashboard reads. Field names and
shapes are forward-compatible: new fields land at the end of an object,
existing fields keep their names and types.

The serializer relies on :class:`~autogen.beta.events.BaseEvent.to_dict`
for event payload shapes rather than inventing a parallel event
serializer. Anything outside ``autogen.beta.events`` (``Feedback``,
exceptions) is handled here.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._types import Feedback
from ..trace import TokenUsage

if TYPE_CHECKING:
    from ..sources.trace_source import TraceRef
    from .result import Aggregates, RunResult, ScoreStats, TaskResult


__all__ = (
    "dump",
    "load_run",
    "to_dict",
)


def to_dict(result: "RunResult") -> dict[str, Any]:
    """Serialize a :class:`RunResult` to a schema-0.1 JSON-safe dict.

    The result is composed of plain JSON types (dict, list, str, int,
    float, bool, None). Pass it through :mod:`json` for the wire form.
    """
    return {
        "schema_version": result.schema_version,
        "run_id": result.run_id,
        "label": result.label,
        "created_at": result.created_at,
        "duration_ms": result.duration_ms,
        "suite": {
            "name": result.suite.name,
            "size": len(result.suite),
            "source": result.suite.source,
        },
        "target": result.target_path,
        "concurrency": result.concurrency,
        "tasks": [_task_to_dict(tr) for tr in result.tasks],
        "aggregates": _aggregates_to_dict(result.aggregates),
    }


def dump(result: "RunResult", path: str | os.PathLike[str]) -> Path:
    """Write a :class:`RunResult` to ``path`` as JSON, creating parent dirs.

    Returns the resolved :class:`Path` that was written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_dict(result), indent=2, default=str), encoding="utf-8")
    return target


def _task_to_dict(tr: "TaskResult") -> dict[str, Any]:
    return {
        "task_id": tr.task.task_id,
        "inputs": tr.task.inputs,
        "reference_outputs": tr.task.reference_outputs,
        "tags": list(tr.task.tags),
        "metadata": tr.task.metadata,
        "duration_ms": tr.trace.duration_ms,
        "events": [_event_to_dict(e) for e in tr.trace.events],
        "exception": _exception_to_dict(tr.trace.exception),
        "tokens": _tokens_to_dict(tr.trace.tokens),
        "feedback": [_feedback_to_dict(fb) for fb in tr.feedback],
        "budget_violation": tr.budget_violation,
        "trace_ref": _trace_ref_to_dict(tr.trace_ref),
    }


def _trace_ref_to_dict(ref: "TraceRef | None") -> dict[str, Any] | None:
    """Serialize a :class:`TraceRef` (or ``None``) — the real OTEL trace id + join key."""
    if ref is None:
        return None
    return {"trace_id": ref.trace_id, "task_id": ref.task_id, "metadata": dict(ref.metadata)}


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Serialize one event. Uses ``BaseEvent.to_dict()`` if available.

    Unknown event types fall back to ``{"type": ClassName, ...vars}``.
    """
    if hasattr(event, "to_dict"):
        payload = event.to_dict()
        if isinstance(payload, dict) and "type" in payload:
            return payload
        return {"type": type(event).__name__, **(payload if isinstance(payload, dict) else {})}
    return {"type": type(event).__name__, **{k: v for k, v in vars(event).items() if not k.startswith("_")}}


def _feedback_to_dict(fb: Feedback) -> dict[str, Any]:
    return {
        "key": fb.key,
        "score": fb.score,
        "value": fb.value,
        "comment": fb.comment,
        "detail": fb.detail,
    }


def _exception_to_dict(exc: BaseException | None) -> dict[str, Any] | None:
    if exc is None:
        return None
    return {"type": type(exc).__name__, "message": str(exc)}


def _tokens_to_dict(tokens: TokenUsage) -> dict[str, int]:
    return {
        "input": tokens.input,
        "output": tokens.output,
        "cache_creation": tokens.cache_creation,
        "cache_read": tokens.cache_read,
    }


def _aggregates_to_dict(aggregates: "Aggregates") -> dict[str, Any]:
    return {
        "pass_rate": dict(aggregates.pass_rate),
        "score_stats": {key: _score_stats_to_dict(stats) for key, stats in aggregates.score_stats.items()},
        "value_counts": {key: dict(counts) for key, counts in aggregates.value_counts.items()},
        "tokens": {
            "input": aggregates.tokens.input,
            "output": aggregates.tokens.output,
            "total": aggregates.tokens.total,
        },
        "errors": aggregates.errors,
        "budget_violations": aggregates.budget_violations,
    }


def _score_stats_to_dict(stats: "ScoreStats") -> dict[str, float | int]:
    return {
        "mean": stats.mean,
        "p50": stats.p50,
        "p95": stats.p95,
        "n": stats.n,
    }


def load_run(path: str | os.PathLike[str]) -> "RunResult":
    """Load a persisted run JSON back into a :class:`RunResult`, for comparison.

    Reconstructs each task's identity (id, inputs, reference, tags, metadata) and its
    scorer :class:`Feedback` — enough for :meth:`RunResult.diff` and the pass-rate /
    score / value accessors. Event traces are **not** reconstructed: the per-task
    :class:`~autogen.beta.eval.Trace` carries no events, so ``aggregates.tokens`` reads
    zero on a loaded run. The JSON is a record for cross-time comparison, not a full
    trace round-trip — read it directly if you need event-level detail.
    """
    # Local imports: store.py is the writer; importing the result/dataset types at
    # module top would invert the result.py -> store.py dependency (AGENTS.md exempts
    # circular-import shims).
    from ..dataset import Suite, Task
    from ..sources.trace_source import TraceRef
    from ..trace import Trace
    from .result import RunResult, TaskResult

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    task_results: list[TaskResult] = []
    for t in doc.get("tasks", []):
        task = Task(
            task_id=t["task_id"],
            inputs=dict(t.get("inputs") or {}),
            reference_outputs=t.get("reference_outputs"),
            tags=tuple(t.get("tags", ())),
            metadata=dict(t.get("metadata") or {}),
        )
        feedback = tuple(
            Feedback(
                key=f["key"],
                score=f.get("score"),
                value=f.get("value"),
                comment=f.get("comment"),
                detail=f.get("detail"),
            )
            for f in t.get("feedback", [])
        )
        trace = Trace(
            events=[], exception=_exception_from_dict(t.get("exception")), duration_ms=int(t.get("duration_ms", 0))
        )
        ref_doc = t.get("trace_ref")
        trace_ref: TraceRef | None = (
            TraceRef(
                trace_id=ref_doc.get("trace_id", ""),
                task_id=ref_doc.get("task_id"),
                metadata=dict(ref_doc.get("metadata") or {}),
            )
            if ref_doc is not None
            else None
        )
        task_results.append(
            TaskResult(
                task=task,
                trace=trace,
                feedback=feedback,
                budget_violation=bool(t.get("budget_violation", False)),
                trace_ref=trace_ref,
            )
        )

    suite_doc = doc.get("suite", {})
    suite = Suite(
        tasks=tuple(tr.task for tr in task_results),
        name=suite_doc.get("name", "loaded"),
        source=suite_doc.get("source", str(path)),
    )
    return RunResult(
        run_id=doc.get("run_id", ""),
        tasks=tuple(task_results),
        suite=suite,
        target_path=doc.get("target", ""),
        concurrency=int(doc.get("concurrency", 1)),
        duration_ms=int(doc.get("duration_ms", 0)),
        created_at=doc.get("created_at", ""),
        label=doc.get("label"),
        store_dir=None,
    )


def _exception_from_dict(exc: dict[str, Any] | None) -> BaseException | None:
    """Rebuild a placeholder exception (carrying the serialized type + message)."""
    if not exc:
        return None
    return RuntimeError(f"{exc.get('type', 'Exception')}: {exc.get('message', '')}")
