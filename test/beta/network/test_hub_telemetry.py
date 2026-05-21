# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub-level OpenTelemetry tracing: ``HubTelemetryListener`` spans, the
Hub-owned ``network.envelope`` span, WAL-truth traceparent persistence,
and end-to-end propagation into the agent-side ``invoke_agent`` span.
"""

import json
import threading
from collections.abc import Sequence as SequenceType

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.middleware.builtin import TelemetryMiddleware
from autogen.beta.network import (
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.hub import HubTelemetryListener
from autogen.beta.network.hub._envelope_tracing import EnvelopeTracer
from autogen.beta.network.hub.layout import spans_path

from ._helpers import ScriptedConfig, wait_for_text_count


class _InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: SequenceType[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def shutdown(self) -> None:
        with self._lock:
            self._spans.clear()


@pytest.fixture()
def otel_setup():
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


async def _read_disk_spans(store: MemoryKnowledgeStore) -> list[dict]:
    data = await store.read(spans_path())
    return [json.loads(line) for line in (data or "").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_hub_emits_envelope_span_and_persists_traceparent(otel_setup) -> None:
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, tracer_provider=provider, ttl_sweep_interval=0)
    hub.register_listener(HubTelemetryListener(store, tracer_provider=provider))

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(Agent("alice", config=ScriptedConfig("")), Passport(name="alice"), Resume())
    await bob_hc.register(Agent("bob", config=ScriptedConfig("")), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi there")
    await wait_for_text_count(hub, channel.channel_id, 1)

    # Hub-owned envelope span exists, is PRODUCER, and has the channel SpanLink.
    spans = exporter.get_finished_spans()
    envelope_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "envelope"]
    assert envelope_spans, "expected at least one network.envelope span"
    env_span = envelope_spans[0]
    assert env_span.name.startswith("network.envelope")
    assert env_span.kind.name == "PRODUCER"
    assert env_span.attributes["ag2.network.sender_id"] == alice.agent_id

    # WAL is the source of truth: the persisted envelope carries the traceparent.
    wal = await hub.read_wal(channel.channel_id)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert text_envelopes and text_envelopes[0].trace_id, "traceparent must be persisted on the WAL envelope"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_listener_emits_channel_and_agent_spans_to_disk(otel_setup) -> None:
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, tracer_provider=provider, ttl_sweep_interval=0)
    listener = HubTelemetryListener(store, tracer_provider=provider)
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(Agent("alice", config=ScriptedConfig("")), Passport(name="alice"), Resume())
    await bob_hc.register(Agent("bob", config=ScriptedConfig("")), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await alice_hc.close_channel(channel.channel_id, reason="done")

    # Channel span carries opened + closed events; agent.lifetime spans emit on close.
    disk = await _read_disk_spans(store)
    by_type: dict[str, list[dict]] = {}
    for rec in disk:
        by_type.setdefault(rec["attributes"].get("ag2.span.type", "?"), []).append(rec)

    assert by_type.get("channel"), "expected a network.channel span on disk"
    channel_rec = by_type["channel"][0]
    event_names = [e["name"] for e in channel_rec["events"]]
    assert "opened" in event_names
    assert "closed" in event_names

    # bytes counter reflected in Hub.health()
    assert hub.health()["telemetry_log_bytes"] > 0

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_task_terminal_emits_single_shot_span(otel_setup) -> None:
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    listener = HubTelemetryListener(store, tracer_provider=provider)

    # on_task_event is the public listener surface; only terminal kinds fire.
    await listener.on_task_event(
        "task-1",
        "completed",
        {
            "owner_id": "alice",
            "channel_id": "chan-1",
            "capability": "synthesis",
            "outcome": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "at": "2026-01-01T00:00:05Z",
        },
    )

    spans = exporter.get_finished_spans()
    task_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "task"]
    assert len(task_spans) == 1
    span = task_spans[0]
    assert span.name == "network.task synthesis"
    assert span.status.status_code.name == "OK"
    # start_time backdated to started_at → ~5s duration despite single-shot emit.
    assert (span.end_time - span.start_time) == pytest.approx(5 * 1_000_000_000, rel=1e-3)


@pytest.mark.asyncio
async def test_started_and_progress_task_kinds_do_not_emit(otel_setup) -> None:
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    listener = HubTelemetryListener(store, tracer_provider=provider)

    await listener.on_task_event("t", "started", {"capability": "x"})
    await listener.on_task_event("t", "progress", {"capability": "x"})

    assert exporter.get_finished_spans() == []


@pytest.mark.asyncio
async def test_traceparent_propagates_to_agent_invoke_span(otel_setup) -> None:
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, tracer_provider=provider, ttl_sweep_interval=0)
    hub.register_listener(HubTelemetryListener(store, tracer_provider=provider))

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    # alice stays silent (empty reply halts the exchange); bob replies once.
    alice = await alice_hc.register(
        Agent("alice", config=ScriptedConfig(""), middleware=[TelemetryMiddleware(tracer_provider=provider)]),
        Passport(name="alice"),
        Resume(),
    )
    await bob_hc.register(
        Agent("bob", config=ScriptedConfig("ok"), middleware=[TelemetryMiddleware(tracer_provider=provider)]),
        Passport(name="bob"),
        Resume(),
    )

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi")
    await wait_for_text_count(hub, channel.channel_id, 2)

    spans = exporter.get_finished_spans()
    envelope_spans = {s.context.span_id: s for s in spans if s.attributes.get("ag2.span.type") == "envelope"}
    invoke_spans = [s for s in spans if s.attributes.get("ag2.span.type") == "agent"]
    assert invoke_spans, "expected at least one invoke_agent span"

    # At least one invoke_agent span is a child of a network.envelope span:
    # same trace, parent == the envelope span.
    nested = [
        s
        for s in invoke_spans
        if s.parent is not None
        and s.parent.span_id in envelope_spans
        and s.context.trace_id == envelope_spans[s.parent.span_id].context.trace_id
    ]
    assert nested, "invoke_agent should nest under the hub's network.envelope span via the relayed traceparent"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_envelope_span_is_root_even_when_posted_under_active_span(otel_setup) -> None:
    # Regression for the "envelope span isn't always a trace root" issue:
    # an envelope posted while another span is current (e.g. set_context →
    # EV_CONTEXT_SET from inside a tool) must still root its own trace, with
    # the causal relationship preserved as a triggered_by SpanLink — not as a
    # parent edge that merges the two traces.
    exporter, provider = otel_setup
    store = MemoryKnowledgeStore()
    et = EnvelopeTracer(provider, store)
    env = Envelope(
        channel_id="c1",
        sender_id="agent_x",
        audience=None,
        event_type="ag2.context.set",
        event_data={},
    )

    dummy_tracer = provider.get_tracer("test")
    with dummy_tracer.start_as_current_span("execute_tool synthesise") as dummy:
        dummy_ctx = dummy.get_span_context()
        span = et.start_envelope_span(env)
        await et.finish_envelope_span(span)

    [es] = [s for s in exporter.get_finished_spans() if s.name == "network.envelope ag2.context.set"]
    # Root despite being created while `dummy` was the current span.
    assert es.parent is None
    assert es.context.trace_id != dummy_ctx.trace_id
    # Causality kept as a SpanLink back to the triggering span.
    triggered = [link for link in es.links if link.attributes.get("ag2.link.kind") == "triggered_by"]
    assert len(triggered) == 1
    assert triggered[0].context.span_id == dummy_ctx.span_id


@pytest.mark.asyncio
async def test_hub_without_tracer_provider_stays_otel_free() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)  # no tracer_provider

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(Agent("alice", config=ScriptedConfig("")), Passport(name="alice"), Resume())
    await bob_hc.register(Agent("bob", config=ScriptedConfig("")), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi there")
    await wait_for_text_count(hub, channel.channel_id, 1)

    # No envelope spans emitted, no traceparent stamped, no spans file written.
    wal = await hub.read_wal(channel.channel_id)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert text_envelopes and text_envelopes[0].trace_id is None
    assert await store.read(spans_path()) in (None, "")
    assert hub.health()["telemetry_log_bytes"] == 0

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
