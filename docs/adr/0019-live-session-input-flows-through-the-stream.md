---
status: accepted
date: 2026-09-24
---

# 0019. Live session input flows through the stream, not a session method

## Context

A `LiveAgent` needs to accept user turns (`ModelRequest`) while its session runs — typed
by a human next to the voice channel, or produced by a program (background tasks and
subagents that `enqueue` into the stream's inbox). The provider has to receive them, and
when it may answer them is provider-specific (OpenAI rejects `response.create` while a
response is active).

Two shapes were on the table: `RealtimeConfig.session()` yields a handle with an explicit
`push(request)` that `LiveAgent` calls, or the provider's session subscribes to
`ModelRequest` on the context's stream itself.

## Decision

The provider's session subscribes to `ModelRequest` on the stream, exactly as it already
subscribes to `RecordedAudioEvent` and `ToolResultEvent`. The `session()` signature does
not change; the `RealtimeConfig` docstring states that a session consumes `ModelRequest`.
`LiveAgent` owns only the inbox: at session open and on every `MessageEnqueued` it drains
`pending_messages` and publishes them as a plain `ModelRequest`, which then takes the same
path. `context.enqueue(*inputs)` is the public way to hand a running agent input — the same
call for `Agent` and `LiveAgent`, for a human and for a program.

The inbox used to be passive — `enqueue` only appended, and `Agent` looked at it before
each model call. A live session has no model calls of its own to hang that look on, so a
message enqueued while the model is silent would wait for the user's next utterance.
`ConversationContext.enqueue` therefore stays synchronous (it is called from subscribers
and from sync code, and making it `async` would turn every un-updated call into a silently
dropped coroutine) and announces the append by publishing `MessageEnqueued` from a
background task. `Stream.enqueue` has no context to publish with, so it remains a
low-level append that announces nothing. `Agent` ignores the event and keeps draining
before each model call.

A provider that itself publishes a user turn built from captured audio (the cascade's
transcript) publishes it as a marker subclass and skips that subclass in its own
subscription, so it never feeds its own speech back in.

## Consequences

- Every input into a live session — audio, tool results, user turns — is an event on the
  stream, so observers, history and middleware see pushed input with no extra wiring.
- `MessageEnqueued` lands one event-loop tick after the append, not inline. A direct
  `stream.enqueue(...)` wakes nobody; code that needs a live session to react goes through
  the context.
- The contract is passive: a provider that forgets to subscribe drops pushed input
  silently, and no type check catches it. Each provider needs a test that a pushed
  `ModelRequest` reaches its connection.
- Timing rules (add to the conversation now, answer at the next response boundary, one
  answer for several pushes) live in each provider, which is where the constraints are.
  Every request for a response — for pushed input or for tool results — goes through that
  one per-provider rule, which also accounts for responses the provider starts on its own
  (turn detection), so no request collides with an already active response.
