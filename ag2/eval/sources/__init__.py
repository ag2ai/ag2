# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trace sources: ingest stored/OTEL traces (in-memory, directory, Tempo) for evaluation."""

from ._otel import readable_span_to_data, readable_spans_to_trace
from ._spans import (
    DEFAULT_CONVENTIONS,
    AG2GenAIConvention,
    OpenInferenceConvention,
    SpanConvention,
    SpanData,
    spans_to_trace,
)
from .tempo import TempoTraceSource
from .trace_source import DirectoryTraceSource, InMemoryTraceSource, TraceRef, TraceSource

__all__ = (
    "DEFAULT_CONVENTIONS",
    "AG2GenAIConvention",
    "DirectoryTraceSource",
    "InMemoryTraceSource",
    "OpenInferenceConvention",
    "SpanConvention",
    "SpanData",
    "TempoTraceSource",
    "TraceRef",
    "TraceSource",
    "readable_span_to_data",
    "readable_spans_to_trace",
    "spans_to_trace",
)
