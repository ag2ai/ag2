---
status: accepted
date: 2026-10-30
---

# 0027. An A2A cancel that races the task's completion keeps the first terminal state

## Context

a2a-sdk does not order `AgentExecutor.cancel()` against `execute()`. `ActiveTask.cancel`
only checks that its consumer has not yet *processed* a terminal event, so a cancel can
arrive after the turn has already enqueued `COMPLETED`. The base `TaskUpdater` raises
`RuntimeError` on a second terminal transition, and in a2a-sdk 1.2.x–1.3.0 an error out of
`cancel()` makes the SDK write `FAILED` over the `COMPLETED` a streaming client already
received (a2aproject/a2a-python#1397).

The spec makes `COMPLETED` terminal (§4.1.3) and closes the stream on it (§3.1.2), so a
client is entitled to treat it as final. §3.1.5 is ambiguous about the reply to a cancel
that loses: "success is not guaranteed … returns its updated state" against
`TaskNotCancelableError` for a task "already completed, failed, or canceled". The SDK
answers both ways: its local path returns the finished task, its cross-replica path
(`_cancel_remote`) raises `TaskNotCancelableError`.

## Decision

- **The first terminal state wins.** The turn and `cancel()` share one `_ClosingTaskUpdater`;
  a transition after a terminal one is a no-op, not a `RuntimeError`, so nothing triggers
  the SDK's `FAILED` overwrite. Lifecycle observers are fed from the same updater and never
  see the state that lost.
- **`cancel()` with no running turn writes nothing.** The turn may have just ended with its
  terminal event not yet consumed; the SDK re-reads the task after `cancel()` returns and
  closes it if it is still open.
- **`CancelTask` answers `TaskNotCancelableError` when the cancel loses**
  (`_RequestHandler` in `ag2/a2a/transports/_common.py`), so a client never reads a
  successful cancel that cancelled nothing, and the local and cross-replica paths agree.

## Consequences

- A cancel in flight when the task finishes is reported as an error, not as the finished
  task. If upstream settles #1397 the other way, only `_RequestHandler` changes.
- The guarantees hold on the replica running the task. A cancel through another replica is
  written to the shared store by the SDK; the running turn is stopped without sending the
  closing `last_chunk`, and observers there may already have seen a `COMPLETED` that the
  store then rejects.
- A task parked in `INPUT_REQUIRED` has no running turn, so its `CANCELED` is written by the
  SDK and is not delivered to the parking request's observers.
