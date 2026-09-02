# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry ``ReadableSpan`` → :class:`Trace` bridge (producer-side).

Unlike :mod:`ag2.eval.sources._spans` (which is SDK-free), this module needs the
OpenTelemetry SDK: it turns the spans a live run produced — or spans read back
from an exporter — into the normalized :class:`SpanData` the pure adapter
consumes.

Importing it is still safe without the SDK installed: the ``ReadableSpan``
import is guarded by :func:`~ag2._import_utils.optional_import_block`, and both
functions are wrapped in :func:`~ag2._import_utils.require_optional_import`, so
a *call* made without the SDK raises an ``ImportError`` naming the
``ag2[tracing]`` extra rather than failing at import time. Both are re-exported
from :mod:`ag2.eval.sources` (and :func:`readable_spans_to_trace` from
:mod:`ag2.eval`) so callers that produce their own spans can build a
:class:`Trace` that grades identically to one from ``run_agent``.
"""

from collections.abc import Sequence

from ag2._import_utils import optional_import_block, require_optional_import

from ..trace import Trace
from ._spans import SpanConvention, SpanData, SpanEvent, spans_to_trace

with optional_import_block():
    from opentelemetry.sdk.trace import ReadableSpan

__all__ = (
    "readable_span_to_data",
    "readable_spans_to_trace",
)


@require_optional_import("opentelemetry.sdk", "tracing")
def readable_span_to_data(span: "ReadableSpan") -> SpanData:
    """Normalize one OpenTelemetry ``ReadableSpan`` into a :class:`SpanData`."""
    context = span.context
    parent = span.parent
    status = span.status.status_code.name if span.status is not None else "UNSET"
    return SpanData(
        name=span.name or "",
        span_id=format(context.span_id, "016x") if context is not None else "",
        parent_id=format(parent.span_id, "016x") if parent is not None else None,
        start_ns=span.start_time or 0,
        end_ns=span.end_time or 0,
        attributes=dict(span.attributes or {}),
        status=status,
        events=tuple(SpanEvent(e.name, dict(e.attributes or {})) for e in span.events),
    )


@require_optional_import("opentelemetry.sdk", "tracing")
def readable_spans_to_trace(
    spans: "Sequence[ReadableSpan]",
    *,
    conventions: Sequence[SpanConvention] | None = None,
    duration_ms: int | None = None,
) -> Trace:
    """Reconstruct a :class:`Trace` from a list of OpenTelemetry ``ReadableSpan``."""
    return spans_to_trace([readable_span_to_data(s) for s in spans], conventions=conventions, duration_ms=duration_ms)
