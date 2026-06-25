---
status: accepted
date: 2026-06-26
---

# Network tracing: bounded per-entity traces, WAL-truth traceparent, opt-in by layering

How the hub emits OpenTelemetry traces for cross-agent activity (envelope dispatch,
channels, agent lifecycles, tasks) without making OpenTelemetry a hard dependency,
without collapsing everything onto one unbounded trace, and without losing trace
context when the write-ahead log is replayed.

## Context

Agent-side telemetry (`TelemetryMiddleware`) already traces what happens *inside*
one agent. The network adds a second layer — what happens *between* agents — and it
brings three tensions a naive design gets wrong:

- **Trace shape.** A hub can run for hours with long-lived channels and thousands of
  envelopes. Parenting all of it under one root span produces a trace that never ends
  and that no collector or viewer ingests well.
- **Dependency surface.** The hub and `autogen.beta.task` are core; most users never
  enable tracing. OpenTelemetry must not become a hard import on the hot path.
- **Replay.** Channel state is rebuilt by re-folding the WAL on `hydrate()`. If trace
  context lived only in memory, replayed envelopes would either lose their original
  trace or spuriously emit new spans.

## Decision

**1. Bounded traces joined by links, not one tree.** Each entity roots its *own*
trace — `network.envelope` (one per dispatched message, `SpanKind.PRODUCER`),
`network.channel`, `agent.lifetime`, `network.task`. They are correlated by
`SpanLink`s (`in_channel`, `triggered_by`) and by shared resource attributes
(`service.instance.id`, `ag2.hub.id`), not by a common parent. The envelope span is a
root even when posted under an active span, so a caller's span never swallows it.

**2. The WAL is the source of truth for traceparent.** The hub starts the
`network.envelope` span and injects its W3C `traceparent` into `envelope.trace_id`
*before* the envelope is appended to the WAL. The receiving agent's middleware
extracts it to parent its `invoke_agent` span. Because the traceparent is persisted on
the envelope, replay during `hydrate()` re-folds deterministically through
`adapter.fold()` and does **not** re-emit spans — historical context is preserved on
the records, not reconstructed.

**3. Opt-in by layering.** Nothing is traced unless a `tracer_provider` is passed to
`Hub.open`. OpenTelemetry is imported only in the middleware and network seams
(guarded), never in core. The shared string vocabulary lives in an import-free
`autogen.beta._telemetry_consts` module so an OTel-free consumer (the network handler,
the eval reader) can reference a key without pulling in OpenTelemetry.

**4. Span vocabulary — align with GenAI semconv, extend under `ag2.*`.** Agent-side
spans follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(`invoke_agent` / `chat` / `execute_tool` / `await_human_input` and `gen_ai.*`
attributes) for backend interop. On top of that we:

- add an `ag2.span.type` discriminator (a closed value set) so eval/query consumers
  filter on a stable field rather than the operation name;
- namespace every AG2-specific attribute under `ag2.*` (`network.*`, `agent.*`,
  `checkpoint.*`) to avoid colliding with semconv;
- pin a fixed schema URL and instrumentation-scope name so every AG2 tracer reports
  consistently.

The GenAI semconv is still experimental, so all of these strings are isolated in
`_telemetry_consts.py` to absorb churn in one place. One span, abbreviated:

```
network.envelope ag2.msg.text   ag2.span.type=envelope            (PRODUCER, trace root)
  └─ invoke_agent bob           ag2.span.type=agent  gen_ai.operation.name=invoke_agent
       └─ chat                  ag2.span.type=llm    gen_ai.request.model=gpt-4o-mini
```

The full span/attribute catalog is **not** duplicated here — it is reference material
maintained in the telemetry docs (`website/docs/beta/telemetry/agent.mdx` and
`website/docs/beta/telemetry/network.mdx`).

**Related — checkpoints as span-events, not spans.** Task checkpoints bypass the
envelope path (direct `KnowledgeStore` writes), so they would otherwise be invisible.
`HubBackedCheckpointStore` records them as `checkpoint.write` / `checkpoint.read`
*span-events* on the active span rather than as their own spans — keeping cardinality
low and keeping OpenTelemetry out of `autogen.beta.task` (the same layering rule as #3).

## Consequences

- Replay and `hydrate()` are deterministic and never double-emit; trace context rides
  the WAL, so the WAL stays the single source of truth.
- Collectors ingest cleanly — no trace is held open indefinitely.
- Core stays OTel-free; the tracing extras are genuinely optional.
- The cost: there is **no single root span** for "everything the hub did." Correlation
  in the backend relies on `SpanLink`s and the `ag2.hub.id` resource attribute. A
  consumer that expects one tree per hub will not find it — this is intentional.
- Aligning with an experimental semconv means `gen_ai.*` names may shift upstream; the
  blast radius is confined to `_telemetry_consts.py`.
