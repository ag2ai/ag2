---
title: Resuming a Turn
sidebarTitle: Resuming
---

`agent.ask(...)` starts a turn from a new message on the agent's current stream. `agent.resume(...)` does the same job from a **recorded trajectory**: you hand it a list of past events, and it re-enters the agent loop driven by the **last** event in that list. The trigger can be *any* event (a fresh user message, a tool result, a human reply) so `resume` is the general way to rebuild a conversation from stored state and carry it forward.

!!! tip "Most multi-turn chats do **not** need `resume`"
    To continue a conversation within a running process, just keep using the reply: call `agent.ask(...)` once, then `reply.ask(...)` for each follow-up. That is the normal continuation path — `resume` is **not** a step you run after `ask`.

    ```python linenums="1"
    reply = await agent.ask("Plan a trip to Kyoto.")
    reply = await reply.ask("Make it five days.")   # continue — no resume needed
    ```

    Reach for `resume` only when you **can't** hold onto that reply: the process ended, another worker picks the turn up, or you need to drive the loop from a non-message event such as a tool result. See [How it relates to `ask` / `reply.ask`](#how-resume-works) below.

## How `resume` works

`resume` takes the full trajectory as positional `events`:

```python
await agent.resume(*events, ...)
```

The list is split in two:

- **All events except the last** seed the stream's history — the conversation state the model will see.
- **The last event** is the **trigger**: the event that drives the next LLM call. It can be any [event](advanced/stream.md) — typically a `ModelRequest` (a new user message) or a `ToolResultsEvent` (a tool result the model should react to).

```python linenums="1"
*history, trigger = events
# history  -> replaces the stream's history (the prefix the model sees)
# trigger  -> drives the next LLM call (any BaseEvent)
```

Everything after that is identical to `ask`: the agent calls the model, may issue tool calls, and returns an `AgentReply`. The `resume` signature mirrors `ask` exactly — `stream`, `dependencies`, `variables`, `prompt`, `config`, `tools`, `middleware`, `observers`, `response_schema`, and `hitl_hook` all behave the same way.

!!! note "How it relates to `ask` / `reply.ask`"
    [`reply.ask(...)`](agents.md) continues the **same live stream** the turn ran on — you keep the conversation going as long as you still hold the `AgentReply` object in the running process. That stream may itself be durable (a `RedisStream`, say); what `reply.ask` needs is the in-process handle, not where the history happens to be stored. `resume` is for when you no longer have that handle — a process restart, a worker on another machine, a turn rebuilt from a store — so you supply the events yourself and drive the next one.

## Continue a stored conversation

The most common use is multi-turn that outlives the process: persist a conversation, load it back later, and continue with a new user message as the trigger.

```python linenums="1" hl_lines="12 15-16"
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import ModelRequest, TextInput

agent = Agent(
    "assistant",
    prompt="You are a helpful travel assistant.",
    config=OpenAIConfig("gpt-4o-mini"),
)

# A conversation you stored earlier and just loaded back from your database.
past_events = load_conversation("thread-42")

# Drive the next turn with a fresh user message as the trigger.
trigger = ModelRequest([TextInput("And what about getting there by train?")])
reply = await agent.resume(*past_events, trigger)
print(reply.body)
```

This is `reply.ask("...")`, except you drive it from events you reload yourself rather than from a live `AgentReply` handle held in the running process.

## Resume from a tool result

A turn can also stop **mid-loop** — the model asked to run a tool whose result is produced elsewhere: a webhook, a queue worker, a long-running job, or a human approving a request. You record the tool call, run the work separately, then hand the result back as the trigger. The model reacts to the result without the tool being re-executed.

```python linenums="1" hl_lines="18 25 28"
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)

agent = Agent("support", prompt="Answer using the tool result.", config=OpenAIConfig("gpt-4o-mini"))

# The trajectory recorded earlier: the user asked, and the model responded
# by requesting a tool call (whose result is not yet known).
call = ToolCallEvent(name="lookup_order", arguments='{"id": "A-1001"}', id="call-1")
history = [
    ModelRequest([TextInput("Where is order A-1001?")]),
    ModelResponse(message=ModelMessage(""), tool_calls=ToolCallsEvent([call])),
]

# The tool result, produced out of band, becomes the trigger.
trigger = ToolResultsEvent([ToolResultEvent.from_call(call, "Shipped, arriving Tuesday.")])

# Re-enter the loop: the model sees the history + result and writes the answer.
reply = await agent.resume(*history, trigger)
print(reply.body)  # -> grounded answer using "Shipped, arriving Tuesday."
```

`ToolResultEvent.from_call(call, result)` pairs the result with the original call so the model can match it to its request. Because `resume` re-enters the **live** loop, the model is free to react by calling more tools — those continuation calls execute normally; only the trigger event is replayed.

## Capture, persist, resume

A trajectory is just a list of events, so you can store it anywhere and pass it back to `resume` later — even from another process. Pull the events from a live stream with `await stream.history.get_events()`, or build them yourself, as long as the prefix ends at the point you want to continue from.

```python linenums="1" hl_lines="2 7"
# --- First process: capture the trajectory and persist it ---
events = list(await reply.context.stream.history.get_events())
save_to_store(stream_id="thread-42", events=events)

# --- Later / another process: load it back and drive the next event ---
events = load_from_store(stream_id="thread-42")
reply = await agent.resume(*events, ModelRequest([TextInput("Carry on.")]))
print(reply.body)
```

!!! note "Reading back the event log"
    `resume` reads only the events you pass in — it does not load history from a store on its own. The [event log](advanced/knowledge_store.md) written by `KnowledgeConfig(write_event_log=True)` is one place these trajectories can come from: it persists each turn to `/log/{stream_id}.jsonl`, and `EventLogWriter(store).load(stream_id)` reads it back as typed events ready to pass straight to `resume`.

    ```python linenums="1"
    from autogen.beta.knowledge import EventLogWriter

    events = await EventLogWriter(store).load(stream_id)
    reply = await agent.resume(*events, ModelRequest([TextInput("Carry on.")]))
    ```

## The stream is replaced, not appended

`resume` **replaces** the target stream's history with the seeded prefix. If you omit `stream`, a fresh `MemoryStream` is created. If you pass an existing stream, any history it already held is discarded in favour of the trajectory you provide.

```python linenums="1" hl_lines="7"
from autogen.beta.stream import MemoryStream

stream = MemoryStream()
await stream.history.replace([ModelRequest([TextInput("stale prior turn")])])

# The stale turn is dropped; only the events you pass remain.
reply = await agent.resume(*history, trigger, stream=stream)
```

This makes the recorded trajectory the single source of truth for the resumed turn — the events you pass are exactly what the model sees.

!!! warning "Durable-backed streams are overwritten in place"
    The replacement is applied to the stream's **storage**, not just an in-memory copy. For a persistent stream such as `RedisStream`, seeding the prefix overwrites the durable history stored under that stream id — the backing store clears the key and rewrites it. Resume into a fresh stream, or use a new stream id, when you need to keep the original conversation's stored history intact.

## Caveats — resuming isn't always possible

`resume` replays provider-native events back to the model, and some providers attach opaque, **required** metadata to those events. Gemini 3.x, for example, binds a *thought signature* to each function call and rejects a replayed call that arrives without it:

```
400 INVALID_ARGUMENT — Function call is missing a thought_signature in functionCall parts
```

OpenAI reasoning items and Anthropic thinking signatures carry similar provider-specific state. As long as you resume from the **original** events — the ones the agent emitted, written verbatim by the [event log](#capture-persist-resume) — this metadata travels with them and `resume` round-trips cleanly. The risk appears when a trajectory is rebuilt from a lossy form: a hand-rolled `{name, args, result}` record, or a trace store that keeps only the fields it understands. The required metadata is silently dropped, and the provider rejects the resumed turn.

!!! warning
    Resuming is not guaranteed for every provider and every trajectory. Preserve the events exactly as the agent emitted them; do not assume a trajectory reconstructed from a reduced or normalized form can be resumed.

## Related

- [Agent Communication](agents.md) — `ask` / `reply.ask`, the in-memory entry points.
- [Human in the Loop](context/human_in_the_loop.md) — pausing for human input, a common reason a turn is continued out of band.
- [Stream & Events](advanced/stream.md) — the event types that make up a trajectory and how to subscribe to them.
