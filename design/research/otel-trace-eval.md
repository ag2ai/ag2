# Trace-Based Evaluation (OTEL)

Status: **design**, accepted direction (2026-05-22). Supersedes the v0 "offline,
in-memory events" model in `eval_development/AG2 Evaluation Specification.md`
(now out of date). PR #2797 will **not** ship as-is; it becomes the vehicle for
the full trace-based solution.

## Decision

Evaluation is a **pure function of a trace**:

```
evaluate(trace) -> feedback
```

The evaluator never executes an agent and has no idea how the trace was made.
A trace can come from a live run we assisted with, a file on disk, or a cloud
observability backend — all graded identically. The **substrate is OTEL spans**;
where those spans physically live is a backend concern hidden behind a protocol.

This inverts today's architecture, where `runner.py` *is* the center (executes
`target.ask`, captures in-memory events, builds the `Trace`, scores). The live
runner **demotes to an optional trace *producer***: ensure telemetry is on, run
the agent, locate the resulting trace, hand it to the evaluator.

### What's frozen (the durable surface)

1. **Trace serialization format** — a span tree + the `ag2.*` vocabulary owned by
   `autogen/beta/_telemetry_consts.py` (see "Relationship to feat/beta-network-tracing").
   The portable, on-disk/over-the-wire contract. Encoding: likely span-JSONL (that
   branch already forward-references "the disk-JSONL reader") — confirm vs OTLP-JSON.
2. **`Scorer` signature** — unchanged (`inputs / outputs / reference_outputs /
   trace / task`, async, returns `Feedback | list[Feedback] | bool | ... `).
3. **`TraceSource` protocol** — NEW. The seam for pluggable trace backends.

`EvalTarget` / `target.ask` is **demoted** from "the contract the runner drives"
to "an optional helper the producer uses." It is no longer central and may evolve.

## Relationship to `feat/beta-network-tracing`

That in-flight branch (2 commits) is upstream of this work and already builds
much of our telemetry foundation. It is **eval-aware** — its constants file names
"the disk-JSONL reader, eval / query code" and "eval scorecards" as the consumers
it centralises strings for. Dependency direction: **tracing → eval**.

Already done by that branch (adopt, don't rebuild):

- **`autogen/beta/_telemetry_consts.py`** — the single, import-free (OTel-free)
  source of truth for the `ag2.*` vocabulary: `ATTR_SPAN_TYPE` + the closed
  span-type set (agent/llm/tool/human_input/envelope/channel/task/agent_lifetime/
  agent_event), network/agent/error attribute keys, tracer identity, and the
  `TRACEPARENT_DEP_KEY` propagation contract. **This resolves our "ag2.* naming"
  frozen-surface question — we import these verbatim in the span→Trace adapter.**
- **Tool-error capture** — tool spans get `StatusCode.ERROR` + recorded exception
  (incl. `ToolErrorEvent`), so `no_tool_errors` reconstructs from span status.
- **Cross-trace nesting** — `TelemetryMiddleware.on_turn` parents `invoke_agent`
  under an inbound network span via `TRACEPARENT_DEP_KEY` in dependencies.
- **Network/hub spans** — `network.envelope/channel/task`, `agent.lifetime` etc.
  (`network/hub/telemetry.py`, `_envelope_tracing.py`) + task TTL (`expires_at`).

Still our gap (residual Phase 1):

- **No disk span sink + reader, no on-disk format** — the branch forward-references
  it; we build it. Lean toward aligning with their span-JSONL shape.
- **No halt span** — `HaltEvent` is not emitted; `trace.halted` won't round-trip.
  Add `SPAN_TYPE_HALT` to the shared consts + emit it.
- **Single-agent `run_subtask` traceparent** — only the network handler
  (`handlers.py:228`) sets `TRACEPARENT_DEP_KEY`; single-agent sub-tasks don't nest
  yet. Add the same dep-set in `run_subtask` for single-agent subtask grading.
- **`ag2.output.*` final answer on root span** — only the last llm span carries
  `gen_ai.output.messages`; derivable, so optional (a root attr is cleaner).
- **`ag2.eval.*` reference stamping** — eval-specific; add to the shared consts
  so the vocabulary stays centralised.

Coordinate with that author on: on-disk format, reserving `ag2.eval.*`, halt-as-span,
and `run_subtask` propagation. Merge order: land network-tracing first, then rebase
eval onto it; or merge its branch in now to unblock and reconcile on land.

## Architecture

```
   PRODUCE (optional helper)                 EVALUATE (the core)
   ─────────────────────────                 ────────────────────
   target.ask(prompt)                        TraceSource.list() ─┐
     │  TelemetryMiddleware                                       │ TraceRef(s)
     ▼  emits spans                           TraceSource.load(ref)
   span sink ──► trace artifact ───────────►  span→Trace adapter
   (file / in-mem / OTLP backend)                   │  Trace
                                                     ▼
                                              scorers / agent-judges
                                                     ▼
                                              TaskResult ─► RunResult (persisted)
```

The universal substrate is **spans**. `TraceSource` yields `Trace` objects built
from spans; backends differ only in *where the spans live*.

## Contracts

### TraceSource protocol (new)

```python
@dataclass(frozen=True, slots=True)
class TraceRef:
    trace_id: str
    task_id: str | None = None          # join key to a Suite Task, if stamped
    metadata: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class TraceSource(Protocol):
    """A backend that supplies traces for evaluation."""

    def list(self) -> AsyncIterator[TraceRef]: ...   # cheap; what's available
    async def load(self, ref: TraceRef) -> Trace: ...  # materialize one Trace
```

`list` / `load` are split so cloud backends can enumerate cheaply and load lazily
/ in parallel. Shipped implementations:

- `DirectoryTraceSource(path)` — OTLP-JSON files under a directory. `trace_id`
  from the root span; `task_id` from an `ag2.eval.task_id` root attribute.
- `InMemoryTraceSource(spans)` — for the producer's fast live path (no disk).
- Cloud backends (Tempo / Honeycomb / Langfuse / Jaeger / …) implement the
  protocol against their query API. We ship the protocol + one reference HTTP
  example; vendors/users plug in their own.

### Trace (in-memory view) — unchanged shape, new source

`Trace` stays the read-only "what happened" object scorers grade against
(`eval/trace.py`). Today it's filled by `EventCapture`; now it is also
constructible from a span tree. `events_of(...)`, `.tokens`, `.duration_ms`,
`.halted`, `.reply` all reconstruct from spans — so **scorers do not change**.

`reply` becomes a lightweight projection exposing `.body` / `.response` read from
the root agent span's output attributes. `outputs` (the dict scorers receive)
becomes a **projection of the trace** rather than a separate input.

## Telemetry extensions needed

Today `TelemetryMiddleware` (`middleware/builtin/telemetry.py`) emits only 4 span
types (agent / llm / tool / human_input). For a trace to be a *lossless* eval
substrate we add:

| Need | Today | Add |
|------|-------|-----|
| `trace.halted` | no halt span | `ag2.halted` attr + `ag2.halt` span event on the agent span (reason, severity) |
| sub-task / network grading | no subtask spans | child spans `ag2.span.type=subtask` nesting the sub-agent's spans (also unlocks multi-stream Trace) |
| `no_tool_errors` / `no_tool_not_found` | result only | `error.type` (semconv) + `ag2.tool.status` (ok\|error\|not_found) on tool spans |
| final answer (`final_answer_matches`, outputs) | derive from last llm span | `ag2.output.body` + `ag2.output.response` (JSON) on the root agent span |
| reference join (eval-produced only) | n/a | `ag2.eval.task_id`, `ag2.eval.reference_outputs` (JSON) on root span; absent for prod traces |

Mapping existing prebuilt scorers to spans confirms coverage: `token_budget` →
`gen_ai.usage.*` (exists); `tool_called` → tool spans (exist); the rest covered
by the additions above.

**Consequence accepted:** telemetry completeness becomes a *correctness
requirement of eval*, not just observability polish. Fidelity is bounded by what
the emitter writes, so the span→Trace round-trip must be fidelity-tested against
a live `EventCapture` of the same run.

## span → Trace adapter

Parse OTLP `ResourceSpans` → flat span list → order by start time → map each span
to typed event(s): llm → `ModelResponse` (+ `Usage` from `gen_ai.usage.*`); tool →
`ToolCallEvent` + `ToolResultEvent` (status from `error.type`); halt event →
`HaltEvent`; subtask spans → `TaskStarted/Completed/Failed`. Build
`Trace(events=…, reply=<projection>, duration_ms=<root span>, …)`.

## Producer (demoted runner)

Thin glue: install a capturing span sink (file exporter → directory, or in-memory),
run `target.ask`, flush, return a `TraceRef`. This is the "assist with running the
agent + know where the trace is stored" piece the user described. The live `run()`
becomes `produce → evaluate` over an `InMemoryTraceSource` / `DirectoryTraceSource`.

## Reference / task join

A trace records *what the agent did*, not *what the right answer was*. Two ways to
supply `reference_outputs`:

1. **Stamp into the trace** at produce time (`ag2.eval.reference_outputs`) → the
   trace is self-contained. Preferred for eval-harness runs.
2. **External pairing** — the evaluator joins `TraceRef.task_id` to a `Suite`.

Captured *production* traces carry no reference → they grade with reference-free
scorers and **agent-judges** only. This is why trace-based eval and the
agent-as-judge thread reinforce each other. See `agent-as-judge.md`.

## Evaluator entry point

```python
async def evaluate(
    source: TraceSource,
    *,
    scorers: Iterable[Scorer],
    suite: Suite | None = None,        # for external reference pairing
    store_dir: str | os.PathLike[str],
    budgets: BudgetThresholds | None = None,
    concurrency: int = 4,
    run_id: str | None = None,
) -> RunResult: ...
```

Iterate `source.list()` → for each ref load the `Trace`, resolve
`reference_outputs` (stamp → suite fallback), project `outputs`, run scorers,
build `TaskResult`; aggregate → `RunResult`; persist (unchanged store schema).

## Migration of PR #2797

- **Keep**: `Trace`, `Scorer` / `@scorer`, `Feedback`, `RunResult` / `Aggregates`
  / `ScoreStats`, `Suite`, `Task`, `scorers/`, `store`.
- **Add**: trace serialization (OTLP-JSON), span→Trace adapter, `TraceSource` +
  `DirectoryTraceSource` + `InMemoryTraceSource`, telemetry span additions,
  `evaluate()` entry, `agent_judge` scorer + `Verdict`.
- **Refactor**: `runner.py` splits into producer + evaluator; `EvalTarget`
  demoted; `run()` reframed as produce-then-evaluate.

## Phased build plan

1. **Telemetry completeness (residual)** — most is done on `feat/beta-network-tracing`
   (span vocabulary, tool-error status, network/task spans, cross-trace nesting).
   Residual: halt span (`SPAN_TYPE_HALT`), single-agent `run_subtask` traceparent
   propagation, `ag2.output.*` on root span, `ag2.eval.*` namespace. All additive to
   `_telemetry_consts.py`; independently testable; no eval coupling yet.
2. **Trace serialization + span→Trace adapter** — DONE (in part): pure
   `spans_to_trace` over `SpanData` (`eval/_spans.py`); OTel `ReadableSpan`
   bridge (`eval/_otel.py`); live round-trip fidelity test asserting the
   span-reconstructed `Trace` matches the `EventCapture` `Trace` on
   scorer-relevant projections (tokens, final response, tool call/result).
   Remaining: load `SpanData` from stored JSON (format TBD); live error-path
   fidelity (telemetry emitting ERROR tool spans on real failures — not easily
   triggered under `TestConfig`; adapter side is unit-covered).
3. **`TraceSource` protocol + Directory/InMemory backends + `evaluate()`** —
   demote runner to producer; `run()` = produce + evaluate.
4. **Reference join + outputs-from-trace projection.**
5. **Agent-as-judge** scorer (`Verdict`, compose-not-subclass) — see
   `agent-as-judge.md`.
6. **Cloud `TraceSource` reference impl + docs**; update the eval spec; update PR.

## Open questions / risks

- **Trace format encoding**: OTLP-JSON (standard, verbose, tool-interop) is the
  lean; confirm vs an AG2-native span dump.
- **OTEL dependency floor**: keep the SDK out of the core eval import path — the
  span→Trace adapter parses a format, it should not require the live SDK.
- **Fidelity test** is the make-or-break: a round-trip diff (live events vs
  span-reconstructed events) must be part of CI.
