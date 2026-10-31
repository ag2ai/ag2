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

- **The first terminal state wins.** The turn and `cancel()` write through one `_TaskWriter`,
  a single queue drained by a single task, so their transitions are ordered without a lock.
  A transition after a terminal one is a no-op, not a `RuntimeError`, so nothing triggers
  the SDK's `FAILED` overwrite. Lifecycle observers are fed from the same writer, by a
  separate notifier, and never see the state that lost. Neither the writer nor a transition
  waits for that notifier: a transition resolves once it is on the wire, so a slow observer
  delays neither the turn nor a cancel. The end of `execute()` waits for delivery, so an
  observer can still cancel a task it sees enter `INPUT_REQUIRED`, and subscriptions close
  after the last notification. A cancelled `execute()` leaves that wait to a background task.
- **`cancel()` with no running request writes nothing.** The turn may have just ended with its
  terminal event not yet consumed; the SDK re-reads the task after `cancel()` returns and
  closes it if it is still open. A request counts as running until its notifications are
  delivered, or until its `execute()` is cancelled. So a `CancelTask` that arrives in that
  window, after the turn wrote `INPUT_REQUIRED`, is written by the request's writer: the
  `CANCELED` the SDK would write anyway, plus a notification. After a terminal state it is a
  no-op. The executor cannot tell an observer's cancel from a client's, so closing the window
  would drop the observer's.
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
- A task parked in `INPUT_REQUIRED` has no running request once delivery is done, so its
  `CANCELED` is written by the SDK and is not delivered to the parking request's observers.
- A cancel an observer starts without awaiting it, and that reaches the writer only after the
  last notification was delivered, is dropped: the writer has stopped. An observer that wants
  its cancel written awaits it.
