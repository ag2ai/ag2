# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for autogen.opentelemetry.setup module."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Decision

from autogen.opentelemetry.consts import (
    INSTRUMENTING_LIBRARY_VERSION,
    INSTRUMENTING_MODULE_NAME,
    OTEL_SCHEMA,
)
from autogen.opentelemetry.setup import DropNoiseSampler, get_tracer


# ---------------------------------------------------------------------------
# DropNoiseSampler
# ---------------------------------------------------------------------------
class TestDropNoiseSampler:
    """Tests for the DropNoiseSampler which filters noisy a2a spans."""

    def setup_method(self) -> None:
        self.sampler = DropNoiseSampler()

    def test_a2a_span_is_record_only(self) -> None:
        """Spans starting with 'a2a.' should be RECORD_ONLY (not exported)."""
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="a2a.server.request",
        )
        assert result.decision == Decision.RECORD_ONLY

    def test_a2a_dot_prefix_is_record_only(self) -> None:
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=99999,
            name="a2a.something.else",
        )
        assert result.decision == Decision.RECORD_ONLY

    def test_non_a2a_span_is_record_and_sample(self) -> None:
        """Non-a2a spans should be RECORD_AND_SAMPLE (fully exported)."""
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="conversation my_agent",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_invoke_agent_span_is_record_and_sample(self) -> None:
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="invoke_agent assistant",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_chat_span_is_record_and_sample(self) -> None:
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="chat gpt-4",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_empty_name_is_record_and_sample(self) -> None:
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_a2a_without_dot_is_record_and_sample(self) -> None:
        """'a2a' without trailing dot should NOT be filtered."""
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="a2a_something",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_trace_state_is_passed_through(self) -> None:
        ts = trace.TraceState([("key", "value")])
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="some_span",
            trace_state=ts,
        )
        assert result.trace_state == ts

    def test_attributes_not_passed_through(self) -> None:
        """Input attributes should not be passed through to the SamplingResult."""
        result = self.sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="some_span",
            attributes={"key": "value"},
        )
        # SamplingResult converts attributes=None to an empty mapping
        assert not result.attributes

    def test_get_description(self) -> None:
        assert "a2a" in self.sampler.get_description().lower()


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------
class TestGetTracer:
    """Tests for the get_tracer factory function."""

    def test_returns_tracer(self) -> None:
        provider = TracerProvider()
        tracer = get_tracer(provider)
        # The returned tracer should be a valid tracer instance
        assert tracer is not None

    def test_tracer_has_correct_instrumenting_module(self) -> None:
        provider = TracerProvider()
        tracer = get_tracer(provider)
        scope = tracer._instrumentation_scope
        assert scope.name == INSTRUMENTING_MODULE_NAME

    def test_tracer_has_correct_version(self) -> None:
        provider = TracerProvider()
        tracer = get_tracer(provider)
        scope = tracer._instrumentation_scope
        assert scope.version == INSTRUMENTING_LIBRARY_VERSION

    def test_tracer_has_correct_schema_url(self) -> None:
        provider = TracerProvider()
        tracer = get_tracer(provider)
        scope = tracer._instrumentation_scope
        assert scope.schema_url == OTEL_SCHEMA

    def test_different_providers_return_different_tracers(self) -> None:
        provider1 = TracerProvider()
        provider2 = TracerProvider()
        tracer1 = get_tracer(provider1)
        tracer2 = get_tracer(provider2)
        # Tracers from different providers should be distinct objects
        assert tracer1 is not tracer2

    def test_same_provider_returns_equivalent_tracer(self) -> None:
        provider = TracerProvider()
        tracer1 = get_tracer(provider)
        tracer2 = get_tracer(provider)
        # Both should have the same instrumentation scope
        assert tracer1._instrumentation_scope == tracer2._instrumentation_scope

    def test_tracer_used_with_drop_noise_sampler(self) -> None:
        """Verify the tracer works with the DropNoiseSampler."""
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from test.opentelemetry.conftest import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider(sampler=DropNoiseSampler())
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        tracer = get_tracer(provider)

        # Non-a2a span should be exported
        with tracer.start_as_current_span("conversation test") as span:
            span.set_attribute("test", True)

        # a2a span should NOT be exported
        with tracer.start_as_current_span("a2a.server.something") as span:
            span.set_attribute("test", True)

        exported = exporter.get_finished_spans()
        span_names = [s.name for s in exported]
        assert "conversation test" in span_names
        assert "a2a.server.something" not in span_names
