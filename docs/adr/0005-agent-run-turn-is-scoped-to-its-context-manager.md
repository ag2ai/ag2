---
status: accepted
date: 2026-06-21
---

# `Agent.run` turns are scoped to their `async with` block (cancel-on-exit)

`Agent.run(...)` is the non-blocking counterpart to `Agent.ask(...)`. Where `ask`
awaits the whole turn and returns an `AgentReply`, `run` is an async context manager
that starts the turn in the background and yields an `AgentRun` handle so the caller can
observe events live and abandon the turn early:

```python
async with agent.run("Hi!") as run:
    async for event in run:          # live event feed, ends when the turn ends
        ...
    result = await run.result()      # AgentReply, same as ask() would return
```

## Context

A background turn outlives the statement that launched it, so its **lifetime** has to be
defined explicitly. Two coherent models exist:

- **Detached / await-on-exit** (trio-nursery style): normal exit *waits* for the turn to
  finish even if the caller never asked for the result; only an exception cancels it.
- **Scoped / cancel-on-exit**: the turn may not outlive the `async with` block. If it has
  not completed by the time the block exits, it is cancelled.

The primary motivating use cases for `run` are *live observation* and *early
abandonment* (cancellation). Under the detached model, the natural "I've seen enough"
gesture — `break` out of the event loop — would silently block on a completion the caller
explicitly walked away from, and abandoning a turn would require raising an exception.

There is also a failure-visibility concern unique to background execution: an exception
raised by a turn that nobody retrieves is silently lost (the orphaned-task footgun).

## Decision

**The turn is scoped to the `async with` block.** On block exit:

- If the turn is still running, it is **cancelled**, and exit awaits the cancellation —
  no task escapes the block.
- To let a turn finish, the caller awaits `run.result()` (or iterates the event feed to
  completion) **inside** the block.

`run.result()` is the **authoritative, idempotent** outcome: it awaits completion,
returns the `AgentReply` on success, and re-raises the turn's exception on failure;
repeated calls return the same reply / re-raise the same exception and never re-run the
turn.

The `AgentRun` handle's **async iteration is a pure event feed**: `async for event in
run` yields every emitted event and then ends cleanly when the turn finishes, whether it
succeeded or failed. It does not itself raise the turn's error. The handle also exposes
`.stream` so the existing `where()` / `get()` / `subscribe()` primitives remain reachable
for filtered observation.

**Failures are never swallowed.** If the turn raised on its own (not a cancellation the
caller triggered by leaving early) and the caller exits the block without ever calling
`result()`, `__aexit__` re-raises that exception. A cancellation caused by the caller's
own early exit is expected and is swallowed.

The turn starts on `__aenter__`, not when `run()` is called — a `run()` whose context
manager is never entered launches nothing and leaks nothing.

## Consequences

- The block is a hard lifetime boundary: a `run` turn can never outlive its `async with`,
  mirroring how `ask()` guarantees the turn is complete when it returns. This is the
  property that makes `run` safe to reach for casually.
- Cancellation is ergonomic: `break`, `return`, a `timeout`, or any propagating exception
  all abandon the turn by leaving the block.
- It is **surprising**: a reader may assume the background turn keeps running after the
  block. The cost of getting this wrong (a half-finished turn the caller expected to
  complete) is why this is recorded here rather than left implicit.
- A cancelled turn may leave a partially-written history / orphaned `tool_use` records on
  a shared stream, the same way any cancelled `ask` would. Callers who need a turn to
  complete must await `result()` before exiting.
- The choice is **hard to reverse**: callers will write `break`-to-cancel code against
  this guarantee. Switching to the detached model later would silently change the meaning
  of that code. Superseding ADR required to change it.
