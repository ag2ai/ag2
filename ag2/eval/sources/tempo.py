# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TempoTraceSource — read traces from a Grafana Tempo HTTP API.

A cloud :class:`~ag2.eval.TraceSource`: ``list()`` enumerates traces
via Tempo's TraceQL search (``GET /api/search``); ``load()`` fetches one trace
(``GET /api/traces/{id}`` as OTLP-JSON) and reconstructs a :class:`Trace` via
the generic ``otlp_json_to_spans`` parser + ``spans_to_trace`` adapter. The
span→Trace machinery is shared with the disk/in-memory sources — only the
fetch layer is Tempo-specific, so other OTLP backends can follow this shape.

This lets the evaluator grade traces another system emitted into a shared
store, correlated by ``trace_id`` — execution and grading fully decoupled.
"""

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from ..trace import Trace
from ._otlp_json import otlp_json_to_spans
from ._spans import SpanConvention, spans_to_trace
from .trace_source import TraceRef

__all__ = ("TempoTraceSource",)


class TempoTraceSource:
    """A :class:`~ag2.eval.TraceSource` backed by a Tempo HTTP API.

    Args:
        base_url: Tempo query base, e.g. ``"http://localhost:3200"``.
        query: TraceQL passed to ``/api/search``. Defaults to ``"{}"`` (all).
        lookback_seconds: Search window ending now. Default 1 hour.
        limit: Max traces returned by ``list``.
        headers: Extra headers (e.g. auth) merged into every request.
        task_id_attribute: Span/search attribute holding the eval task id, if
            the producer stamped one (else refs carry ``task_id=None``).
        client: Optional injected ``httpx.AsyncClient`` (for tests); when given,
            it is used as-is and not closed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        query: str = "{}",
        lookback_seconds: int = 3600,
        limit: int = 20,
        headers: Mapping[str, str] | None = None,
        task_id_attribute: str = "ag2.eval.task_id",
        conventions: Sequence[SpanConvention] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._query = query
        self._lookback = lookback_seconds
        self._limit = limit
        self._headers = {"Accept": "application/json", **(headers or {})}
        self._task_id_attribute = task_id_attribute
        self._conventions = conventions
        self._client = client

    async def list(self) -> AsyncIterator[TraceRef]:
        now = int(time.time())
        params = {"q": self._query, "limit": self._limit, "start": now - self._lookback, "end": now}
        data = await self._get("/api/search", params=params)
        for summary in data.get("traces", []):
            trace_id = summary.get("traceID")
            if not trace_id:
                continue
            yield TraceRef(
                trace_id=trace_id,
                task_id=_summary_attribute(summary, self._task_id_attribute),
                metadata={"root_service": summary.get("rootServiceName"), "root_span": summary.get("rootTraceName")},
            )

    async def load(self, ref: TraceRef) -> Trace:
        doc = await self._get(f"/api/traces/{ref.trace_id}")
        return spans_to_trace(otlp_json_to_spans(doc), conventions=self._conventions)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        if self._client is not None:
            response = await self._client.get(url, params=params, headers=self._headers)
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=self._headers)
            response.raise_for_status()
            return response.json()


def _summary_attribute(summary: Mapping[str, Any], key: str) -> str | None:
    """Pull an attribute value out of a Tempo search summary's spanSets, if present."""
    for span_set in summary.get("spanSets", []):
        for span in span_set.get("spans", []):
            for attr in span.get("attributes", []):
                if attr.get("key") == key:
                    return attr.get("value", {}).get("stringValue")
    return None
