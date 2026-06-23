---
status: accepted
date: 2026-06-16
---

# Tools are one `Tool` abstraction with three seams; `Toolkit` is itself a `Tool`

## Context

The beta agent loop has to expose several very different kinds of "tool" to the
same execution machinery:

- **`FunctionTool`** — wraps a Python callable; the agent executes it locally.
- **`ClientTool`** — schema only; execution is handed back to the *client*
  (e.g. a browser, an A2A peer, the AG-UI front-end).
- **builtin / provider-native tools** (`ShellTool`, `CodeExecutionTool`,
  `ImageGenerationTool`, `MCPServerTool`, `WebSearchTool`, `MemoryTool`, …) —
  the *provider* executes them server-side; the agent only declares the
  capability.
- **`_MCPProxyTool`** — function-tool-shaped, but forwards each call to a remote
  MCP server.
- **`Toolkit`** — a *collection* of tools (skills, Google Drive, an MCP server's
  whole tool list).

A naïve design would give each kind its own container type and its own
registration path, and would make `Toolkit` a separate "bag of tools" that the
agent has to special-case. The legacy `autogen/tools/Toolkit` does exactly that
— it is a plain container that is **not** a `Tool`.

## Decision

There is **one** abstraction: `Tool` (ABC, `autogen/beta/tools/tool.py`), with
exactly three seams that every kind implements:

- **`schemas(context) -> Iterable[ToolSchema]`** — what capability(ies) this
  contributes to the model request. Function-shaped tools emit
  `FunctionToolSchema`; provider-native tools emit their own `*ToolSchema`
  (a provider-neutral capability flag the mappers translate).
- **`set_provider(provider)`** — bind the `fast_depends` DI provider. Only the
  function family needs it; the base and the builtins no-op.
- **`register(stack, context, *, middleware)`** — subscribe this tool's
  execution behaviour onto the event stream for the duration of a run.

`Toolkit(Tool)` is the **composite**: it holds `dict[str, Tool]` and implements
all three seams by fanning out to its children. Because a toolkit *is a* `Tool`,
the agent treats a single tool and a toolkit identically, and toolkits nest
(`MCPToolkit(Toolkit)`, `SkillsToolkit(Toolkit)`, …). `ToolExecutor.register`
just calls `tool.register(...)` over a flat `Iterable[Tool]`, plus a fallback
`ToolNotFound` subscriber — it never inspects tool kind.

## Consequences / things that look wrong but are deliberate

- **`register()` carries *different* execution semantics per subclass — by
  design; do not hoist execution into the base.** `FunctionTool.register`
  wraps middleware, runs the callable, and sends a `ToolResultEvent`.
  `ClientTool.register` emits a `ClientToolCallEvent` and **never executes** —
  the result comes back from the client. `_MCPProxyTool.register` forwards to
  the remote server. The seam is `register`, not an `execute()` method,
  precisely *because* these paths have nothing in common except "subscribe
  something to the stream."

- **Builtin tools implement `register()` with a no-op subscriber, and that is
  not dead code.** `ShellTool` / `MCPServerTool` / `ImageGenerationTool` /
  etc. subscribe an empty `execute` on `BuiltinToolCallEvent`. The *provider*
  runs these server-side, so there is nothing for the agent to execute; the
  no-op subscription exists so the call event has a consumer. Their real
  contribution is `schemas()` — the `*ToolSchema` capability flag. Removing the
  `register` override would be wrong even though the body is `pass`.

- **`Toolkit` *is a* `Tool` (composite), unlike the legacy
  `autogen/tools/Toolkit`.** Do not "align" the two. The beta toolkit being a
  `Tool` is what lets an agent accept a toolkit anywhere it accepts a tool, lets
  toolkits nest, and lets `MCPToolkit` lazily turn a server into a toolkit of
  proxies without the agent knowing.

- **`MCPToolkit.schemas()` mutates `self._tools` on first call (lazy
  discovery).** This looks like a side effect in a getter, but it is the
  *correct* place per the project rule "no side effects in init": the MCP
  handshake (network I/O) must happen at runtime, not construction. The double-
  checked `_discover_lock` guards concurrent first calls.

- **`set_provider` is a no-op on most tools.** Only `FunctionTool` (and via the
  composite, `Toolkit`) needs the DI provider, because only locally-executed
  Python callables resolve `Context` / `Inject` / `Variable` dependencies.
  Client, builtin, and MCP-proxy tools ignore it. The base `Tool.set_provider`
  being a no-op rather than abstract is what keeps those subclasses free of
  boilerplate.

To add a new kind of tool: subclass `Tool`, implement `schemas()` (always), and
implement `register()` to express how it runs (locally, client-side, provider-
side no-op, or remote-forward). Implement `set_provider()` only if it executes
local Python. To group tools, build on `Toolkit` and override `schemas()` if
discovery is lazy — never special-case it in the agent or `ToolExecutor`.
