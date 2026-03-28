# Task Delegation

## Decision

Add a Task concept to the beta framework that enables agents to delegate work to other agents (or themselves). A Task is not a new primitive -- it's a pattern built from existing ones: `@tool`, `Agent.ask()`, and `BaseEvent`.

## Motivation

An orchestrating agent needs to break work into discrete units and delegate them. Each unit should:
- Run on its own stream (isolated history)
- Have an objective and return a result
- Be observable via lifecycle events on the parent stream
- Be triggerable by the LLM (via tool calling) or programmatically

## API

### `agent.as_tool()` — LLM-triggered delegation

```python
researcher = Agent("researcher", config=config, tools=[search_tool])

coordinator = Agent(
    "coordinator",
    config=config,
    prompt="Delegate research to the researcher, then summarize.",
    tools=[
        researcher.as_tool(
            description="Research a topic thoroughly",
            name="task_researcher",       # optional, defaults to "task_{agent.name}"
            max_depth=3,                  # optional, defaults to 1
            stream=lambda: RedisStream(   # optional, defaults to MemoryStream
                url, prefix="ag2:sub"
            ),
        ),
    ],
)

reply = await coordinator.ask("Write about AI safety")
```

The coordinator's LLM sees a tool called `task_researcher` with parameters `objective` (required) and `context` (optional). When it calls the tool:

1. A fresh stream is created (via the factory, or MemoryStream by default)
2. `TaskStarted` event is emitted on the parent stream
3. `run_task()` executes the researcher agent
4. `TaskCompleted` or `TaskFailed` is emitted on the parent stream
5. The result returns to the coordinator's LLM as a tool result

### Layer 3: Self-delegation

An agent delegates to itself by calling `as_tool()` on itself:

```python
analyst = Agent("analyst", config=config, tools=[search])
analyst.tools.append(
    analyst.as_tool(description="Handle a focused sub-task independently", name="sub_task")
)
```

The sub-task is a fresh copy: same prompt, tools, config, but clean stream and history. The LLM decides when to decompose work into sub-tasks.

## Task Events

Lifecycle events emitted on the **parent** stream:

```python
class TaskStarted(BaseEvent):
    task_id: str       # unique ID for this task execution
    agent_name: str    # name of the agent handling the task
    objective: str     # what the task should accomplish

class TaskCompleted(BaseEvent):
    task_id: str
    agent_name: str
    objective: str
    result: str        # the agent's final response
    task_stream: Any   # reference to the sub-task's stream (in-memory only, not serializable)

class TaskFailed(BaseEvent):
    task_id: str
    agent_name: str
    objective: str
    error: str         # the exception message
```

**TaskCompleted vs TaskFailed**: `TaskCompleted` means the agent ran and produced a response. `TaskFailed` means the agent raised an exception. There is no `successful` field -- the orchestrating LLM judges whether the result is good enough.

## Recursion Guard

Self-delegation and mutual delegation (A→B→A) can cause infinite recursion. The guard uses `contextvars.ContextVar` (matching the existing Hub pattern in `autogen/beta/network/hub.py`):

```python
_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("task_depth", default=0)
DEFAULT_MAX_TASK_DEPTH = 1
```

**Depth-aware schema hiding**: When `depth >= max_depth`, the task tool's `schemas()` returns an empty list. The LLM never sees the tool, so it can't call it. No wasted API calls, no error messages.

The error check inside `delegate()` stays as a safety net but should never fire in practice.

**Default `max_depth=1`**: One level of delegation. The top-level agent can delegate (depth 0→1). Sub-tasks work directly (depth 1, tool hidden). Users set higher values explicitly when needed.

## Context Flow

| What | Inherited? | Why |
|---|---|---|
| Dependencies | Yes | Infrastructure (DB, API clients). Shared resources. |
| Variables | No | Mutable conversation state. Fresh scope per task. |
| History | No | Clean stream. LLM passes relevant context via `context` parameter. |
| Agent prompt/tools/config | Yes | The sub-agent brings its own capabilities. |

## Stream Factory

By default, sub-tasks use `MemoryStream`. To persist sub-task events (e.g., to Redis), pass a callable:

```python
researcher.as_tool(
    description="Research a topic",
    stream=lambda: RedisStream(redis_url, prefix="ag2:sub"),
)
```

Each invocation calls the factory to create a fresh stream. The factory pattern avoids sharing a single stream across multiple sub-tasks.

## Implementation

### Files

| File | Purpose |
|---|---|
| `autogen/beta/task.py` | `run_task()`, `TaskResult`, `_TaskTool`, `_make_task_tool()`, depth guard |
| `autogen/beta/events/task_events.py` | `TaskStarted`, `TaskCompleted`, `TaskFailed` |
| `autogen/beta/agent.py` | `Agent.as_tool()` method (delegates to `_make_task_tool`) |
| `autogen/beta/events/__init__.py` | Exports task events |
| `autogen/beta/__init__.py` | Exports `run_task`, `TaskResult` |

### Why this fits the framework

1. **`@tool` is the mechanism.** `_make_task_tool` uses the existing `@tool` decorator. No new Tool subclass for the inner function -- just a thin `_TaskTool` wrapper for schema hiding.
2. **`Agent.as_tool()` parallels `Agent.as_conversable()`.** Same adapter pattern.
3. **`run_task()` is just a function.** Composable, testable, no framework coupling beyond `Agent.ask()`.
4. **Events are just data.** Plain `BaseEvent` subclasses on streams.
5. **Depth guard follows Hub pattern.** `contextvars.ContextVar` with `set()`/`reset()` in try/finally.

## Alternatives Considered

### Task as a new primitive (rejected)
Adding Task alongside Agent/Stream/Event as a core concept. Rejected because Task is naturally expressed as a tool that calls `agent.ask()`. No new machinery needed.

### `_SELF_AGENT_KEY` in context.dependencies (rejected)
For self-delegation, injecting the agent into dependencies via a private sentinel. Rejected because `as_tool()` captures the agent reference in a closure -- standard Python, no framework changes.

### Automatic context copying (rejected)
Automatically copying conversation history or variables into sub-tasks. Rejected because the LLM can selectively pass relevant context via the `context` tool parameter. Mechanical copying pollutes sub-tasks with irrelevant history.

### `successful` field on TaskCompleted (rejected)
A boolean indicating whether the task achieved its objective. Rejected because without a response schema, we can only know if the agent finished (completed) or crashed (failed). The orchestrating LLM judges result quality.

## Related

- `autogen/beta/network/hub.py` — Hub delegation with depth tracking (same `contextvars` pattern)
- `design/research/delegate-tool.md` — Earlier DelegateTool research (deferred, superseded by this)
- `design/research/group-chat-shared-streams.md` — Shared stream patterns (complementary)
