# Context: Agent Invocation

Glossary for how a caller invokes an `ag2` agent and consumes what comes back.
Glossary only — no implementation detail.

## Terms

### Ask
A **blocking** invocation of an agent. The caller awaits the entire turn (model calls,
tool execution, schema retries) and only then receives a **Reply**. The caller cannot
interleave any work between the start of the turn and its completion.

### Run
An **observable** invocation of an agent. The caller obtains a **Run handle** that opens
the turn's scope but does not advance it; the caller then observes the turn live and drives
it to its **Reply**. A Run follows the same invocation signature as an Ask. (An Ask is the
degenerate Run that opens the scope and drives it in one step.)

### Run handle
The object a Run yields to the caller, scoped to the block that opened it. Opening the
handle makes the turn's observation live, but the turn does not advance until the caller
asks for its **Reply**. Through the handle the caller reaches the underlying stream to
**observe** the turn and drives the turn to obtain its authoritative **Reply**. The turn
runs only inside the block and only while its Reply is awaited; abandoning the block
without asking for the Reply runs nothing. The handle's name is `AgentRun`.

### Observe
To watch a turn's events as they happen by subscribing to its stream (push): the caller
attaches a callback before driving, and the callback fires inline as the turn runs. The
stream is the single observation surface — there is no separate event feed on the handle.

### Reply
The outcome of one completed turn — the agent's response for that turn, from which the
caller reads the raw text body, the schema-validated content, generated files, history,
and token usage. A Reply is produced identically whether the turn was an Ask or a Run. For
a Run it is the authoritative, idempotent outcome: requesting it **drives** the turn to
completion and re-raises the turn's failure if there was one; requesting it again yields
the same Reply without re-running the turn.

### Inbox
A per-stream queue of follow-up messages waiting to be fed to the model. A caller adds to
it through a **Run handle** (or the stream directly) while a turn is in flight; the turn
drains the inbox at its next or final model call, so a message added while the turn runs is
consumed by that same turn. Anything added when no turn is running waits for the next one.

### Continuation
A follow-up turn on the conversation a Reply belongs to, reusing that Reply's context,
stream, and client. A Reply can be continued as an **Ask** (blocking) or as a **Run**
(observable) — the same two forms as a fresh invocation, so a conversation is a chain of
turns each begun from the previous turn's Reply.
