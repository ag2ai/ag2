---
name: ag2-subagent-delegation
description: Delegate work from one AG2 beta `Agent` to another. Two patterns — auto-injected `run_subtask` / `run_subtasks(parallel=True)` (opt in via `tasks=TaskConfig(...)`) for self-delegation and parallel fan-out, and `Agent.as_tool()` for named delegates between distinct agents. Use when one coordinator should spawn sub-tasks, fan out concurrent work, or hand off to a specialist agent. Covers context flow, recursion safety, and `persistent_stream` for sub-task history.
license: Apache-2.0
---

# Subagent delegation

## When to use

- "Coordinator + specialists" — a parent agent should hand parts of a task to a research agent, math agent, etc.
- "Fan out then collect" — multi-part questions where each part is independent and parallel execution saves wall time.
- "Self-delegation" — one agent breaks complex work into focused sub-tasks for itself.

## Two patterns

| Pattern | Reach for it when | API |
|---|---|---|
| **Auto-injected** `run_subtask` / `run_subtasks` | Lightweight self-delegation, dynamic fan-out, parallel sub-questions | `tasks=TaskConfig(...)` on the parent |
| **`Agent.as_tool()`** | Distinct named delegates the LLM should reason about ("call the researcher", "call the writer") | Wrap a child Agent as a tool on the parent |

The two compose — a coordinator can have both.

## Pattern 1 — auto-injected `run_subtasks`

Subtask tools are **off by default** (`tasks=False`). Opt in with `tasks=TaskConfig(...)` and the agent gains:

- `run_subtask(task: str)` — one isolated sub-task agent.
- `run_subtasks(tasks: list[str], parallel: bool = True)` — fan out multiple in one tool call (default concurrent).

```python
from autogen.beta import Agent, TaskConfig
from autogen.beta.config import GeminiConfig

config = GeminiConfig(model="gemini-3-flash-preview")

coordinator = Agent(
    "coordinator",
    prompt=(
        "You answer multi-part questions by dispatching run_subtasks "
        "with parallel=True. Use one tool call with every sub-question "
        "packed into the 'tasks' list."
    ),
    config=config,
    tasks=TaskConfig(),  # opt in
)

reply = await coordinator.ask(
    "In one run_subtasks call, answer: "
    "(a) tallest waterfall, (b) Eiffel Tower year, (c) boiling point of nitrogen."
)
```

`TaskConfig` controls how the sub-task agents are built:

```python
@dataclass
class TaskConfig:
    config: ModelConfig | None = None    # falls back to parent's config
    prompt: str = "You are a task agent..."
    include_tools: Iterable[str] | None = None   # None = inherit all parent tools
    exclude_tools: Iterable[str] = ()
    extra_tools: Iterable[Callable | Tool] = ()
```

Common shape — cheaper model for sub-tasks, narrow tool surface:

```python
TaskConfig(
    config=worker_config,                  # smaller model
    prompt="You are a focused worker; one step only.",
    include_tools=["search", "fetch_url"], # don't expose `summarize` to children
)
```

**Sub-task agents are built with `tasks=False`** — they never gain `run_subtask` tools themselves. Recursive delegation is structurally impossible; no depth limit needed.

## Pattern 2 — `Agent.as_tool()`

Expose a whole agent as a tool the LLM can name and call:

```python
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig

config = AnthropicConfig(model="claude-sonnet-4-6")

researcher = Agent("researcher", prompt="Provide concise factual findings.", config=config, tools=[search_tool])
writer     = Agent("writer", prompt="Turn research into clear prose.", config=config)

coordinator = Agent(
    "coordinator",
    prompt="First delegate research, then pass findings to the writer.",
    config=config,
    tools=[
        researcher.as_tool(description="Research a topic and return findings."),
        writer.as_tool(description="Write an article. Pass research notes in the context parameter."),
    ],
)
```

The coordinator's LLM sees `task_researcher` and `task_writer`. Each call has two parameters:

- `objective` (required) — what the sub-task should do.
- `context` (optional) — relevant info the parent wants to share.

`as_tool()` accepts:

| Parameter | Description |
|---|---|
| `description` | Tool description shown to the LLM (required) |
| `name` | Override the default `task_{agent.name}` |
| `stream` | `StreamFactory` for custom sub-task streams (see below) |
| `middleware` | `ToolMiddleware` callables (e.g. `approval_required`) |

For more control, use `subagent_tool()` directly:

```python
from autogen.beta.tools.subagents import subagent_tool

coordinator = Agent("coordinator", config=config, tools=[
    subagent_tool(researcher, description="Research a topic."),
])
```

## Self-delegation via `as_tool()`

If you want a named self-delegate (`sub_task` instead of generic `run_subtask`), give an agent its own tool:

```python
analyst = Agent(
    "analyst",
    prompt=(
        "You have search and sub_task tools. "
        "Only use sub_task when the task has clearly independent parts."
    ),
    config=config,
    tools=[search_tool],
)

analyst.add_tool(
    analyst.as_tool(
        description="Break work into a focused sub-task for independent analysis.",
        name="sub_task",
    )
)
```

### Recursion safety

Self-delegation via `as_tool()` *can* recurse — the child has the same `sub_task` tool, so without a guard the LLM may chain calls indefinitely.

The simplest safe pattern is to **prefer the auto-injected `run_subtask` / `run_subtasks` path** for self-delegation. Sub-tasks spawned that way are constructed with `tasks=False`, so they have no `run_subtask` tools and recursion is structurally impossible.

If you genuinely need recursive `as_tool()` self-delegation, write a tool middleware that increments a depth counter in `context.dependencies` and short-circuits past a threshold. The `subagents` module exports `subagent_tool`, `persistent_stream`, and `StreamFactory` from `autogen.beta.tools.subagents` — verify the current public surface there before relying on a built-in depth-limiting helper.

## Sub-task streams

By default, each sub-task gets a fresh `MemoryStream` — its history is isolated and starts empty. Context flow:

| What | Behaviour | Why |
|---|---|---|
| **Dependencies** | Copied (top-level shallow) | Isolated; treat dependencies as read-only inside subtasks |
| **Variables** | Copied; synced back on success | Concurrent-safe — sibling subtasks won't race-clobber a shared dict |
| **History** | Fresh stream | Clean context; relevant info passes via the `context` tool parameter |
| **Tools** | Inherited from parent (filtered by `TaskConfig`) | Sub-tasks need real capabilities to do work |

### `persistent_stream()`

When a sub-agent benefits from seeing its prior calls (e.g. avoid repeating searches), give it a stream that persists across invocations within the parent context:

```python
from autogen.beta.tools.subagents import persistent_stream

researcher.as_tool(
    description="Research a topic",
    stream=persistent_stream(),
)
```

Stores stream id in `context.dependencies` keyed by `f"ag:{agent.name}:stream"` and reuses the parent stream's storage backend.

### Custom factory

```python
from autogen.beta import Agent, Context
from autogen.beta.streams.redis import RedisStream

def make_redis_stream(agent: Agent, ctx: Context) -> RedisStream:
    return RedisStream(MY_REDIS_URL, prefix=f"ag2:sub:{agent.name}")

researcher.as_tool(description="Research a topic", stream=make_redis_stream)
```

## Going deeper

- Working starter: `assets/research_squad.py` (mirrors `code_examples/05`) — covers both `run_subtasks(parallel=True)` *and* `Agent.as_tool()`, with `TaskStarted` / `TaskCompleted` lifecycle events.
- Full reference: `website/docs/beta/task_delegation.mdx`.
- `tasks=` constructor knob (with `KnowledgeConfig`, etc.): `website/docs/beta/agent_harness.mdx`.

## Common pitfalls

- **Forgetting to opt in** — `tasks=False` is the default. No `TaskConfig`, no `run_subtask` tools.
- **Expecting sub-tasks to recurse with `run_subtask`** — they can't. Sub-tasks themselves have `tasks=False`. If you need deeper trees, use `Agent.as_tool()` self-delegation with a manual depth-counter middleware (see "Recursion safety" above).
- **Sharing mutable variables expecting them to merge** — concurrent sub-tasks each copy variables; sibling mutations don't propagate. Each sub-task's variable mutations stay local until sync-back on success.
- **Treating `dependencies` as scoped per sub-task** — only the top-level dict is copied. Mutable values inside it are still shared by reference. Treat dependencies as read-only inside sub-tasks.
- **No `description=` on `as_tool()`** — the LLM doesn't know when to call it. Required parameter.
- **`run_subtasks(parallel=False)` when work is concurrent** — defaults to `True` for a reason; only set `False` when later tasks depend on earlier results.
- **Confusing `task_{agent.name}` collisions** — pass `name=` to override if you want shorter names or distinct delegates of the same agent.
