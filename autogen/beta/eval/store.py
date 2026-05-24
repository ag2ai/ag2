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
exceptions, reply bodies) is handled here.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._types import Feedback
from .trace import TokenUsage

if TYPE_CHECKING:
    from .result import Aggregates, RunResult, ScoreStats, TaskResult


__all__ = (
    "dump",
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
        "created_at": result.created_at,
        "duration_ms": result.duration_ms,
        "suite": {
            "name": result.suite.name,
            "size": len(result.suite),
            "source": result.suite.source,
        },
        "target_factory": result.target_factory_path,
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
        "reply": _reply_to_dict(tr.trace.reply),
        "exception": _exception_to_dict(tr.trace.exception),
        "tokens": _tokens_to_dict(tr.trace.tokens),
        "feedback": [_feedback_to_dict(fb) for fb in tr.feedback],
        "budget_violation": tr.budget_violation,
    }


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


def _reply_to_dict(reply: Any) -> dict[str, Any] | None:
    """Serialize the target's reply as ``{"body": str | None, "response": dict | None}``."""
    if reply is None:
        return None
    body = getattr(reply, "body", None)
    response = getattr(reply, "response", None)
    if not isinstance(response, dict):
        response = None
    return {"body": body, "response": response}


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
