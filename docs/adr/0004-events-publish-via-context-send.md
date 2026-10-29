---
status: accepted
date: 2026-06-17
---

# Application code publishes events via `context.send`, not `stream.send`

Surfaced while migrating beta callers off the two-argument
`stream.send(event, context)` form, but the rule is framework-wide. The
migration is the worked example.

## Context

Beta has two objects in scope on almost every execution path:

- **`Stream`** — the event bus. Its primitive is
  `send(event, context) -> None` (`ag2/stream.py`): it runs the
  registered interrupters and subscribers, threading `context` through so each
  handler gets the live `dependency_provider` and the `Context` itself as a
  dependency.
- **`ConversationContext`** — wraps a stream plus the per-run state
  (`variables`, `dependencies`, prompt). It exposes the one-argument
  convenience `send(event)`, which publishes through
  `await self.stream.send(event, self)` (`ag2/context.py`). It also gives each
  locally published `UsageEvent` a fresh source identity.

So `context.send(event)` and `stream.send(event, context)` publish onto the
*same* stream with the *same* context — **when `context.stream is stream`**.
The two-argument form additionally lets the caller pass a context that is *not*
the stream's owner, which is a sharp tool: useful in exactly one place,
a footgun everywhere else.

## Decision

**Application code publishes events with `context.send(event)`.** It reads as
the intent ("emit this event in my context"), it can't desync the stream from
the context, and it's the single obvious seam.

**`stream.send(event, context)` is reserved for two cases:**

1. **`Stream` implementations themselves.** It is the primitive that
   `context.send` delegates to. `ConversationContext.send`,
   `SubStream.send`, and `RedisStream.send` (`super().send(...)`) all call it
   directly because they *are* the plumbing — there is no higher-level seam
   below them.

2. **A deliberate cross-stream forward**, where the stream published to and the
   context used for handler dispatch / reply routing are *intentionally
   different*. The subagent HITL bridge is implemented by
   `Forwarder.bridge` in `ag2/tools/subagents/_forwarding.py`.

## Consequences / things that look wrong but are deliberate

- **`Forwarder.bridge` calls `self.parent.stream.send(forwarded, ctx)` and
  must NOT be "fixed" to `self.parent.send(forwarded)`.** This is the
  trap the migration was written to prevent. When a subagent without its own
  HITL hook calls `context.input()`, it `await`s a `HumanMessage` reply on its
  **child** stream (`ag2/context.py`, `input()`). The bridge forwards
  the `HumanInputRequest` to the **parent** stream so the parent's hook handles
  it — but it passes the **child** `ctx` as the context. The parent hook replies
  with `await context.send(reply)` (`ag2/hitl.py`), and because that
  `context` is the child `ctx`, the reply lands on the child stream where
  `input()` is waiting. Rewrite it to `self.parent.send(forwarded)` and the hook
  receives the parent context, the reply goes to the *parent* stream, the child's
  `input()` never resolves — silent deadlock. The stream/context mismatch is the
  whole point of using `stream.send` here.

- **Observation forwarding is separate from the HITL bridge.** Ancestors
  observe descendants through inert `ChildEvent` envelopes under their own
  contexts. `event.source_event` exposes the source for inspection without
  executing its local handlers. Only the explicit bridge dispatches a bare
  `HumanInputRequest` under the requester's context; a local hook takes
  priority. This preserves one answer channel while still making both the
  question and answer observable. See
  [ADR 0027](0027-subagent-events-use-direct-forwarding.md).

- **Tests under `test/stream/` keep using `stream.send(event, context)`.**
  They exist to exercise the `Stream` API directly, and at least one
  (`test_context_propagates_to_substream`) passes a *custom* context distinct
  from the stream's owner on purpose. Migrating them would erase the coverage
  they exist for. Behavioural tests elsewhere use `context.send`, matching real
  application code.

To emit an event: call `context.send(event)`. Reach for
`stream.send(event, context)` only when you are implementing a `Stream`, or when
you genuinely need to publish onto one stream while dispatching under a
different context — and if you do, leave a comment saying why, because the next
person running a `context.send` migration will otherwise assume it's an
oversight.
