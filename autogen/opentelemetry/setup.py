# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Attributes
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import SpanKind

from .consts import INSTRUMENTING_LIBRARY_VERSION, INSTRUMENTING_MODULE_NAME, OTEL_SCHEMA


class DropNoiseSampler(Sampler):
    def should_sample(
        self,
        parent_context: Optional["Context"],
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence["trace.Link"] | None = None,
        trace_state: trace.TraceState | None = None,
    ) -> "SamplingResult":
        decision = Decision.RECORD_ONLY if name.startswith("a2a.") else Decision.RECORD_AND_SAMPLE
        return SamplingResult(decision, attributes=None, trace_state=trace_state)

    def get_description(self) -> str:
        return "Drop a2a.server noisy spans"


def get_tracer(tracer_provider: TracerProvider) -> Tracer:
    return tracer_provider.get_tracer(
        instrumenting_module_name=INSTRUMENTING_MODULE_NAME,
        instrumenting_library_version=INSTRUMENTING_LIBRARY_VERSION,
        schema_url=OTEL_SCHEMA,
    )
