# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trace sources: ingest stored/OTEL traces (in-memory, directory, Tempo) for evaluation."""

from ._spans import (
    DEFAULT_CONVENTIONS,
    AG2GenAIConvention,
    OpenInferenceConvention,
    SpanConvention,
    SpanData,
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
)
