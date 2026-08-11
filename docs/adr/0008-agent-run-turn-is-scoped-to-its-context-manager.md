---
status: accepted
date: 2026-06-22
---

# `Agent.run` opens a turn scope; the turn advances only while its result is awaited

> **Amended by [0009](0009-agent-run-start-drives-in-background.md):** `AgentRun` now drives
> the turn through a single scope-owned `asyncio.Task` (created by `start()` or `result()`),
> and `start()` kicks it off without awaiting. The core decision below — entering the block
> does not auto-start the turn — still holds; the "no background task" mechanism (driving
> inline in the caller's task) is superseded.

`Agent.run(...)` is the observable counterpart to `Agent.ask(...)`. Where `ask` opens a
turn, drives it to completion, and returns an `AgentReply`, `run` is an async context
manager that *opens the turn scope* and yields an `AgentRun` handle, letting the caller
observe the turn as it runs:

```python
async with agent.run("Hi!") as run:
    run.stream.where(ModelMessageChunk).subscribe(on_token)  # observe live
    result = await run.result()                              # drives the turn inline
```

## Context

A turn needs a scope: middlewares are instantiated, and the turn's subscribers (the LLM
callback, HITL, the tool executor) and the observer lifecycle are registered on the
stream. The whole turn is then driven by a single `await agent_turn(...)` — it is
*subscriber-driven*, not an explicit loop. `ask` does scope-open, drive, and scope-close
in one call, so a caller never gets between "subscribers are live" and "the turn runs."

To observe a turn live, a caller needs to attach observation *after* the scope is open but
*before* the turn is driven. The question is how to expose that window.

The tempting design is to run the turn in a background `asyncio.Task` so the caller can
`async for event in run` (pull events) while the turn progresses. But pull-iteration
fundamentally needs concurrency: the turn is one `await`, and yielding its intermediate
events to a caller in the *same* task requires the turn to run concurrently with the
draining — i.e. a background task, or rewriting `context.send` as generator `yield`s
through the entire middleware stack (not viable; the model calls are real I/O). A
background task also brings its own hazards: orphaned-task exceptions, cancellation
plumbing, and a turn that keeps running after the caller has moved on.

## Decision

**`run` opens a turn scope; it does not start a background task.** Entering the
`async with` block registers the turn's subscribers (so observation is live) but does
**not** advance the turn. `AgentRun` is built on the same `_turn_scope` context manager
that backs `ask`/`_ask`/`resume`/`_execute` — there is one launch primitive.

- **The turn advances only while `result()` is awaited.** `result()` drives the turn
  inline, in the caller's task, and returns the `AgentReply`. It is idempotent: the turn
  runs once; repeated calls return the same reply or re-raise the same failure.
- **Observation is push-based.** Because the scope's subscribers are live, a caller
  subscribes to `run.stream` (or attaches `observers=`) before driving, and the callbacks
  fire inline as the turn runs. `run.stream` also exposes `where()` / `get()` / `join()`.
  There is no handle-level `async for`; a caller who wants pull-iteration uses
  `run.stream.join()` and owns the concurrency.
- **Cancellation is inline.** Cancelling the `await result()` (e.g. an `asyncio.timeout`)
  cancels the turn where it is suspended, propagating into the running tool. Leaving the
  block without ever driving runs nothing — there is no turn to cancel and nothing to
  clean up but the scope's subscriptions.

`ask` is the degenerate case: `async with self._open_run(...) as handle: return await
handle.result()` — open the scope, drive once, close.

## Consequences

- **No background task** means no orphaned-task exceptions, no task-cancellation plumbing,
  and no turn outliving the caller. The lifecycle is just the scope: open on `__aenter__`,
  closed on `__aexit__`.
- It is **surprising**: a reader may expect `run()` to start work immediately. It does
  not — `agent.run(...)` and entering the block only open the scope; the model is not
  called until `result()`. This is the core reason for recording the decision.
- Live observation covers the real use case (streaming a turn's events to a UI) via
  push callbacks during `result()`. What it does **not** offer is the turn making progress
  in the background while the caller does unrelated work — that was the concurrency case we
  set aside, and it is the price of dropping the task.
- Splitting *open the scope* from *drive the turn* (`_turn_scope` yielding a `drive`
  callable) is the single seam the whole design rests on. `_execute` — used by
  `AgentReply.ask` and the A2A executor — is now just "enter the scope and drive," so those
  callers are unaffected.
- The choice is **hard to reverse**: callers will subscribe-then-`result()` and rely on
  the turn not running until driven. Reintroducing a background task later would change the
  meaning of that code. Superseding ADR required to change it.
