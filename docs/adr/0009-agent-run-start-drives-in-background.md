---
status: accepted
date: 2026-06-24
---

# `AgentRun.start()` drives the turn in a scope-owned background task

Amends [0008](0008-agent-run-turn-is-scoped-to-its-context-manager.md). The core
decision of 0008 stands: entering the `async with agent.run(...)` block opens the
turn *scope* but does **not** advance the turn. This ADR adds an opt-in way to
drive it without awaiting `result()` inline.

## Context

0008 decided `run` opens a scope and the turn advances only while `result()` is
awaited, explicitly setting aside "the turn making progress in the background
while the caller does unrelated work." The practical consequence is that any
caller who wants to *pull* events while the turn runs must own the concurrency
themselves:

```python
async with agent.run("Hi!") as run:
    driver = asyncio.create_task(run.result())   # caller-owned background task
    with run.stream.join() as events:
        async for event in events:
            ...
    reply = await driver
```

That `asyncio.create_task(run.result())` is boilerplate every pull-iteration
caller repeats, and getting the teardown right (cancel on early exit, retrieve
the task's exception) is exactly the "task hazard" plumbing 0008 wanted to keep
out of user code. The hazards are real but bounded: they only need solving
*once*, inside `AgentRun`.

## Decision

**`AgentRun` drives the turn through a single `asyncio.Task` owned by the scope,
created lazily by either `start()` or `result()`, and adds `start()` to kick it
off without awaiting.** Entering the block still does not auto-start anything —
the task is created only when something drives the run — so 0008's
surprising-but-deliberate "nothing runs until you drive it" holds.

- `start()` creates the task and returns `None` (the task is not exposed). It is
  idempotent and reconciles with `result()`: a no-op once the turn is already
  being driven, whether by an earlier `start()` or a `result()`.
- `await result()` creates the task if `start()` did not, then awaits it,
  returning the same reply or re-raising the same failure (0008's idempotency is
  now a property of awaiting the same task). Cancelling the await cancels the
  turn.
- **The task is owned by the scope.** `__aexit__` cancels it if still running and
  retrieves its outcome, swallowing the turn's cancellation/failure — a caller
  that leaves the block without awaiting `result()` opted out of the result. The
  turn never outlives the block, which is the property 0008 cared about.

This **supersedes 0008's "no background task" mechanism.** 0008 drove the inline
`result()` in the caller's own task to avoid any `asyncio.Task`; collapsing the
two paths onto one task removed the duplicate state (`__driven` / `__reply` /
`__error`) that the split required. The observable trade-off: inline
cancellation is now delivered through the task (at the cancelled await and, as a
backstop, at scope exit) rather than unwinding the turn's call stack
synchronously. The turn is still cancelled and the running tool still receives
`CancelledError`; only the *timing* shifts by at most one event-loop hop.

## Consequences

- The background-task hazards 0008 cited (orphaned-task exceptions, a turn
  outliving the caller, cancellation plumbing) are solved once, in `AgentRun`'s
  `__aexit__`, instead of pushed onto every caller. Every turn — inline or
  started — is bounded by the `async with` block.
- One drive path means one place that holds turn state (the task) and one
  teardown rule, rather than parallel inline/background bookkeeping.
- Pull-iteration becomes first-class: `run.start()` then `async for event in
  run.stream.join()` reads naturally, with no caller-managed task or teardown.
- This does **not** reintroduce auto-start. `agent.run(...)` and entering the
  block remain inert; the model is called only once `start()` or `result()` runs.
