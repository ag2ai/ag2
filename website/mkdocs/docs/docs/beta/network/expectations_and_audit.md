---
title: Governance, Audit and Observability
sidebarTitle: Governance & Audit
---

The hub enforces governance and surfaces observability through a few layered seams:

1. **Evaluators** — pure functions over channel state that return zero or more `#!python Violation` records when their thresholds are breached.
2. **Violation handlers** — what to do when a violation fires: log to the audit trail, notify the channel, or auto-close.
3. **The audit log** — append-only record of every governance-relevant event the hub processes (itself a `#!python HubListener`).
4. **`#!python HubListener`** — read-only observers the hub fans state transitions out to, after the fact.
5. **`#!python HubArbiter`** — the gatekeeper the hub consults *before* committing register / channel-open / send / dispatch decisions.

## Evaluators

Three evaluators ship today, addressed by name in adapter manifests:

| Name | Class | Threshold |
|---|---|---|
| `#!python "acks_within"` | `#!python AcksWithinEvaluator` | All invitees must ack within `params["seconds"]` of channel creation. |
| `#!python "reply_within"` | `#!python ReplyWithinEvaluator` | The respondent must reply within `params["seconds"]` of the initiator's first send (consulting only). |
| `#!python "max_silence"` | `#!python MaxSilenceEvaluator` | No participant may go silent for longer than `params["seconds"]`. |
| `#!python "turn_within"` | (composes from the above) | The next speaker must speak within `params["seconds"]` of being scheduled. |

Each evaluator implements:

```python
class ExpectationEvaluator(Protocol):
    name: ClassVar[str]
    def evaluate(self, ctx: ExpectationContext) -> list[Violation]: ...
```

`#!python ExpectationContext` is a small dataclass holding the metadata, WAL slice, current time, and the expectation's params. Evaluators are pure — no I/O, no mutation — so they're trivially testable.

The default registry exposes them as `#!python default_evaluators()`. Custom evaluators register similarly to custom transition targets.

## Adapter-Declared Expectations

Each adapter's manifest declares its defaults (see [Adapters Overview](adapters_overview.md) for the table). Examples:

```python
# ConsultingAdapter
expectations = [
    Expectation(name="acks_within",  on_violation="auto_close", params={"seconds": 30}),
    Expectation(name="reply_within", on_violation="auto_close", params={"seconds": 600}),
]

# ConversationAdapter
expectations = [
    Expectation(name="max_silence", on_violation="audit", params={"seconds": 3600}),
]
```

`#!python Expectation.on_violation` selects the handler:

| `on_violation` | Handler | Effect |
|---|---|---|
| `#!python "audit"` | `#!python AuditHandler` | Write to the audit log only. Channel continues. |
| `#!python "warn"` | `#!python NotifyChannelHandler` | Post `#!python EV_EXPECTATION_VIOLATED` on the channel WAL. |
| `#!python "auto_close"` | `#!python AutoCloseHandler` | Close the channel with `reason="expectation_violated:<name>"`; record to audit. |
| `#!python "hide"` | (custom) | Hide later turns from the offending participant; not yet implemented as a built-in. |

The default registry exposes them as `#!python default_handlers()`.

## The Sweeper Loop

When the hub is open, an expectation sweeper task wakes every `expectation_sweep_interval` (default 10 s), walks every active channel, runs each expectation's evaluator, and dispatches any violations to the configured handler.

For deterministic tests / examples:

```python linenums="1" hl_lines="4-7 10-11"
from autogen.beta.network import Hub
from autogen.beta.knowledge import MemoryKnowledgeStore

hub = await Hub.open(
    MemoryKnowledgeStore(),
    expectation_sweep_interval=0,  # disable background loop
)

# Manually advance state and tick:
clock.advance(45)                       # mock-clock pattern
await hub._expectation_tick()           # operator API
```

`#!python hub._expectation_tick()` is a public-by-convention test entry point — a leading underscore, but exercised explicitly by the test suite.

## Audit Log

`#!python hub.audit_log` is an `#!python AuditLog` instance — append-only, and itself a registered `#!python HubListener` (so every state transition the hub fans out also lands as one structured record). It writes a single `audit.jsonl` under the hub's `#!python KnowledgeStore`.

```python linenums="1"
records = await hub.audit_log.read_all()
for r in records:
    print(r["kind"], r["at"], r)
```

Each record is a plain dict with at minimum `kind` and `at`; `kind`-specific fields appear alongside.

| Member | Purpose |
|---|---|
| `#!python await hub.audit_log.read_all()` | Read + parse the whole log. `#!python []` if absent. |
| `#!python await hub.audit_log.append(record)` | Write one record. The kind set is **open** — tenants/subclasses append their own `kind` values here. |
| `#!python hub.audit_log.subscribe(cb)` / `#!python unsubscribe(cb)` | Live tail — `#!python cb(record)` fires per appended record (no polling). Subscriber exceptions are logged and swallowed. |
| `#!python hub.audit_log.bytes_written` | Process-local byte counter (resets on hub restart). Surfaced by `#!python hub.health()` as `audit_log_bytes`. |
| `#!python hub.replace_audit_log(custom)` | Swap in a tenant-provided `#!python AuditLog` subclass (e.g. a different on-disk format). The replacement is registered as the first listener so audit writes still complete before tenant listeners observe the same event. |

### Audit kinds

Re-exported as constants from `#!python autogen.beta.network`:

| Constant | Notes |
|---|---|
| `#!python AUDIT_KIND_AGENT_REGISTERED` | Records `agent_id`, `name`. |
| `#!python AUDIT_KIND_AGENT_UNREGISTERED` | Records `agent_id`, `name`. |
| `#!python AUDIT_KIND_RESUME_SET` | Records the source: `#!python RESUME_SOURCE_TENANT` (a `#!python set_resume` call) or `#!python RESUME_SOURCE_OBSERVED` (a `#!python record_observation`). |
| `#!python AUDIT_KIND_SKILL_SET` | Records updated skill markdown. |
| `#!python AUDIT_KIND_RULE_SET` | Records the new rule. |
| `#!python AUDIT_KIND_CHANNEL_CREATED` | Records `creator_id`, manifest type/version, participants. |
| `#!python AUDIT_KIND_CHANNEL_CLOSED` | Records `reason`. |
| `#!python AUDIT_KIND_CHANNEL_EXPIRED` | Records the TTL details. |
| `#!python AUDIT_KIND_TASK_TERMINATED` | Records `owner_id`, `capability`, `outcome`, `latency_ms`. |
| `#!python AUDIT_KIND_EXPECTATION_VIOLATED` | Records `expectation`, `channel_id`, evaluator details. |
| `#!python AUDIT_KIND_TURN_FAILED` | A notify handler crashed processing an inbound envelope. Records `channel_id`, `agent_id`, `envelope_id`, `exc_type`, `exc_message`. |

### Inspection patterns

```python linenums="1"
# Filter to violations only.
violations = [
    r for r in await hub.audit_log.read_all()
    if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED
]

# Filter to one channel.
channel_records = [
    r for r in await hub.audit_log.read_all()
    if r.get("channel_id") == channel_id
]

# Live tail.
async def on_record(record: dict) -> None:
    print("[audit]", record["kind"], record)
hub.audit_log.subscribe(on_record)
```

The `#!python AuditLog` is durable when the hub is backed by `#!python DiskKnowledgeStore`; with `#!python MemoryKnowledgeStore` it lives only as long as the hub.

## HubListener — observing state transitions

A `#!python HubListener` is a **read-only** observer. Attach one with `#!python hub.register_listener(...)`; the hub `await`s the matching method *after* the corresponding state change commits — listeners observe, they don't gate (that's `#!python HubArbiter`, below). Registration itself is inert: there's no startup hook, and methods only fire on subsequent transitions. Each listener call is wrapped in its own `#!python try/except` so a throwing listener can't stall dispatch — the exception is logged at `ERROR` and the next listener still runs.

`#!python BaseHubListener` is a no-op base — subclass it and override only the events you care about (you don't have to implement the full Protocol surface). The built-in `#!python AuditLog` *is* a `#!python HubListener`, pre-registered on every hub, which is why a fresh hub already reports `registered_listeners: 1`.

```python linenums="1"
from autogen.beta.network import BaseHubListener

class MetricsListener(BaseHubListener):
    async def on_envelope_posted(self, envelope, metadata) -> None:
        metrics.incr("network.envelopes", tags={"type": envelope.event_type})

    async def on_envelope_rejected(self, envelope, reason) -> None:
        metrics.incr("network.rejected", tags={"reason": type(reason).__name__})

    async def on_turn_failed(self, channel_id, agent_id, envelope_id, exc) -> None:
        logger.error("turn failed: agent=%s channel=%s", agent_id, channel_id, exc_info=exc)

hub.register_listener(MetricsListener())
# hub.unregister_listener(listener)   # detach later — no-op if absent
```

| Method | Fires when |
|---|---|
| `#!python on_envelope_posted(envelope, metadata)` | An envelope was validated, WAL-appended, folded, and dispatched. |
| `#!python on_envelope_rejected(envelope, reason)` | An envelope was rejected before WAL append. `reason` is the typed `#!python NetworkError` the sender saw. |
| `#!python on_dispatch_failed(envelope, recipient_id, reason)` | Delivery of an accepted envelope to one recipient failed (the rest of the audience may have received it). |
| `#!python on_channel_event(channel_id, kind, payload)` | `kind` ∈ `opened` / `closed` / `expired` / `participant_removed` / `participant_hidden`. |
| `#!python on_agent_event(agent_id, kind, payload)` | `kind` ∈ `registered` / `unregistered` / `resume_set` / `skill_set` / `rule_set` / `observation_recorded`. |
| `#!python on_expectation_fired(channel_id, expectation, violation)` | An evaluator emitted a violation (deduped per `(channel, expectation, violator)`). |
| `#!python on_turn_failed(channel_id, agent_id, envelope_id, exc)` | A notify handler crashed processing an inbound envelope. |
| `#!python on_task_event(task_id, kind, payload)` | `kind` ∈ `started` / `progress` / `completed` / `failed` / `expired` / `cancelled` / `mirror_failed`. |
| `#!python on_inbox_pressure(agent_id, pending, cap)` | A recipient's pending count first crossed its inbox high-water mark (fires once per crossing). |

### Subclassing the Hub instead of registering a listener

The same `#!python on_*` methods are available on `#!python Hub` itself (with no-op defaults) — so if you're building a custom hub you can override them directly rather than registering the hub as a listener of itself:

```python linenums="1"
from autogen.beta.network import Hub

class ObservingHub(Hub):
    async def on_envelope_posted(self, envelope, metadata) -> None:
        metrics.incr("network.envelopes", tags={"type": envelope.event_type})

    async def on_inbox_pressure(self, agent_id, pending, cap) -> None:
        logger.warning("inbox pressure: %s at %d/%d", agent_id, pending, cap)

hub = await ObservingHub.open(store)
```

Subclass overrides fire **alongside** any externally-registered listeners, with the same per-callee `#!python try/except` isolation. Use a subclass when the observation logic belongs to your hub implementation; use `#!python register_listener(...)` when it's a separate concern (metrics shipper, audit tap) you want to attach and detach independently.

`#!python on_task_event` is the one listener hook that isn't purely hub-driven — `#!python TaskMirror` (and other tenant code) emit `#!python "mirror_failed"` and other kinds through it. The public way to fan one out is `#!python await hub_client.fire_task_event(task_id, kind, payload)` (from a registered tenant) or `#!python await hub.fire_task_event(...)` (direct); neither touches the hub's private fan-out.

### Health snapshot

`#!python hub.health()` is a cheap, in-memory operational snapshot — wire it to a `/health` endpoint or dashboard:

```python linenums="1"
hub.health()
# {
#   "active_channels": 2,
#   "registered_agents": 5,
#   "pending_inbox_total": 3,
#   "max_pending_inbox_depth": 2,        # None when nothing queued — indicative of a stuck agent
#   "registered_listeners": 1,           # the built-in AuditLog counts
#   "adapters_loaded": 4,
#   "audit_log_bytes": 8192,
# }
```

## HubArbiter — the decision seam

Where a `#!python HubListener` only observes, a `#!python HubArbiter` **decides**. The hub consults the active arbiter inline before committing register / channel-open / send / dispatch decisions. Exactly one arbiter is active at a time; install yours with `#!python hub.register_arbiter(arbiter)` (and read it back via `#!python hub.arbiter`).

Each gate returns a `#!python Decision` — `#!python Allow()` or `#!python Deny(reason, error=...)`, where `error` selects which `#!python NetworkError` subclass the hub raises back to the caller (defaults to `#!python AccessDeniedError`):

| Gate | Called before | Default `RuleBasedArbiter` checks |
|---|---|---|
| `#!python authorize_send(envelope, sender, sender_rule, recipients)` | `#!python post_envelope` WAL append | `access.outbound_to`, `limits.delegation_depth` |
| `#!python authorize_inbox(envelope, recipient, recipient_rule, current_pending)` | per-recipient, on `#!python post_envelope` | `limits.inbox.max_pending` (denies with `#!python InboxFull`) |
| `#!python authorize_dispatch(envelope, sender, recipient, recipient_rule)` | each notify frame | `access.inbound_from` (deny ⇒ silently skip that recipient) |
| `#!python authorize_channel_open(manifest, creator, creator_rule, invitees, invitee_rules, active_creator_channels)` | `#!python create_channel` | each invitee's `access.inbound_from`, creator's `limits.max_concurrent_channels` |
| `#!python authorize_register(passport, resume, rule)` | *(reserved — not yet wired by the hub)* | always `#!python Allow` |
| `#!python resolve_unknown_audience(envelope, unknown_ids)` | dispatch to ids the hub doesn't know | returns `#!python None` (drop silently) — the federation hook |

The default `#!python RuleBasedArbiter` enforces the per-agent `#!python Rule` (access + limits) — exactly the behavior the hub had inline before this seam existed. Two ways to customise:

- **Add policy on top** — subclass `#!python RuleBasedArbiter` and `#!python await super()` in the gates you extend.
- **Start from scratch** — subclass `#!python BaseHubArbiter` (all gates return `#!python Allow` by default) and implement only the ones you need.

```python linenums="1"
from autogen.beta.network import RuleBasedArbiter, Allow, Deny, EV_TEXT

class ContentGuardArbiter(RuleBasedArbiter):
    BANNED = ("password", "ssn")

    async def authorize_send(self, envelope, sender, sender_rule, recipients):
        base = await super().authorize_send(envelope, sender, sender_rule, recipients)
        if isinstance(base, Deny):
            return base
        if envelope.event_type == EV_TEXT:
            text = str(envelope.event_data.get("text", "")).lower()
            hit = next((b for b in self.BANNED if b in text), None)
            if hit is not None:
                return Deny(reason=f"message blocked: contains {hit!r}")
        return Allow()

hub.register_arbiter(ContentGuardArbiter())
```

A `#!python Deny` from `#!python authorize_send` / `#!python authorize_inbox` / `#!python authorize_channel_open` surfaces to the caller as the chosen `#!python NetworkError` (so `#!python channel.send(...)` raises `#!python AccessDeniedError`); a `#!python Deny` from `#!python authorize_dispatch` just drops that one recipient. `#!python resolve_unknown_audience` is the seam a federated arbiter uses to re-route to a local proxy id instead of dropping.

## Turn-failure resilience

The default notify handler wraps its **entire** substantive path — channel resolve, view projection, `#!python adapter.extract_turn_input`, `#!python agent.ask`, round-envelope build, outbound send. If any step raises, the handler:

1. routes the failure through `#!python HubClient.report_turn_failure` → `#!python Hub.report_turn_failure`,
2. which fans `#!python on_turn_failed(channel_id, agent_id, envelope_id, exc)` out to every `#!python HubListener` — the built-in `#!python AuditLog` writes an `#!python AUDIT_KIND_TURN_FAILED` record,
3. then returns cleanly. **No reply envelope is posted, but the channel stays active and the next envelope flows normally** — a buggy turn no longer takes down the receive loop.

React however you like — retry, escalate, surface to a UI — by registering a listener that overrides `#!python on_turn_failed`.

## Custom Evaluators

Same shape as the built-ins:

```python linenums="1"
from typing import ClassVar
from autogen.beta.network.hub import (
    ExpectationContext,
    ExpectationEvaluator,
    Violation,
)

class TooManyMessagesEvaluator:
    name: ClassVar[str] = "too_many_messages"

    def evaluate(self, ctx: ExpectationContext) -> list[Violation]:
        threshold = ctx.params["max"]
        text_count = sum(1 for e in ctx.wal if e.event_type == EV_TEXT)
        if text_count > threshold:
            return [Violation(
                expectation=self.name,
                channel_id=ctx.channel.channel_id,
                detail=f"text count {text_count} exceeds {threshold}",
            )]
        return []
```

Register on a custom registry and pass to `#!python Hub.open(..., evaluators=registry)`. The default registry can also be mutated via the module-level `#!python register_evaluator(...)` helper.
