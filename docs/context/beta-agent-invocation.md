# Context: Agent Invocation

Glossary for how a caller invokes an `autogen.beta` agent and consumes what comes back.
Glossary only — no implementation detail.

## Terms

### Ask
A **blocking** invocation of an agent. The caller awaits the entire turn (model calls,
tool execution, schema retries) and only then receives a **Reply**. The caller cannot
interleave any work between the start of the turn and its completion.

### Run
A **non-blocking** invocation of an agent. The caller obtains a **Run handle** the moment
the turn starts, while the turn proceeds in the background. This lets the caller observe
the turn's events live as they happen, and abandon the turn before it finishes. A Run
follows the same invocation signature as an Ask.

### Run handle
The object a Run yields to the caller. It represents one in-flight turn, scoped to the
block that started it — the turn may not outlive that block, and leaving the block early
abandons (cancels) the turn. Through the handle the caller does three things: watch the
**Live event feed**, reach the underlying stream for filtered observation, and obtain the
authoritative **Reply** once the turn completes. The handle's name is `AgentRun`.

### Live event feed
The stream of events a Run handle exposes as the turn happens, consumed by iterating the
handle. It is a pure feed: it surfaces every emitted event and then ends when the turn
finishes — whether the turn succeeded or failed. The turn's success or failure is not
adjudicated here; that is the **Reply**'s role.

### Reply
The outcome of one completed turn — the agent's response for that turn, from which the
caller reads the raw text body, the schema-validated content, generated files, history,
and token usage. A Reply is produced identically whether the turn was an Ask or a Run. For
a Run it is the authoritative, idempotent outcome: requesting it awaits completion and
re-raises the turn's failure if there was one; requesting it again yields the same Reply
without re-running the turn.
