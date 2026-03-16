# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for OpenTelemetry tests."""

import threading
from collections.abc import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter


@pytest.fixture(autouse=True)
def _enable_otel_content_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable content recording for all OTel tests.

    Production default is OFF (to prevent secret leakage). Tests need it ON
    to verify that content is correctly attached to spans.
    """
    import autogen.opentelemetry.instrumentators.agent_instrumentators._config as _cfg

    monkeypatch.setattr(_cfg, "RECORD_CONTENT", True)


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for tests.

    Collects finished spans in a list so tests can inspect them.
    """

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        self.clear()
