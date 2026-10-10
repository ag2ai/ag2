---
status: accepted
date: 2026-09-28
---

# 0020. One tool per name in a turn, resolved by precedence

## Context

The model calls a tool by name, and a `ToolCallEvent` reaches every tool subscribed to
that name on the stream. Per-tool state keyed by name, such as `approval_required`'s
"always" answer, assumes one tool behind each name. Tools reach a turn from several
sources: `Agent(tools=...)`, `ask(tools=...)`, toolkits, plugins, sub-task and knowledge
tools, `ToolSearchTool`'s deferred tools, names an `MCPToolkit` discovers at runtime, and
tools a remote client sends over AG-UI, A2A or NLIP (`ClientTool`).
When two of them shared a name, both ran on one call, and a human's decision about one did
not bind the other. An MCP server could therefore put an ungated tool next to a local
tool gated by `approval_required`.

Replacing a tool by registering another with the same name is a real use case, for
example overriding one member of a toolkit.

## Decision

A turn exposes one schema per name and registers one tool per name.
`resolve_tools` (`ag2/tools/precedence.py`) resolves the turn's tools member by member,
after toolkits discover theirs, and applies this precedence:

- **Tools declared in code override each other by order.** The later one wins, like a
  dict update, so `Agent(tools=[toolkit, my_deploy])` replaces the toolkit's `deploy`.
  The override is logged as a warning naming both sources, because the replaced tool may
  have carried an approval gate the winner does not. Built-in tools count as
  declared in code: a function tool and a built-in tool with the same name resolve by
  order too. Built-in schemas of one type may still repeat (several `MCPServerTool`
  servers), because the provider tells those apart.
- **Tools a remote peer provides rank below tools declared in code**, wherever they sit
  in the list: tools an MCP server reports and tools a client sends. A colliding one is
  dropped with a warning naming both sources.
- **Between remote peers, the first one's tool wins.** The others are dropped with a
  warning that suggests `tool_name_prefix` for MCP servers.

A dropped tool is removed from its toolkit's copy for the turn, so it is never
subscribed and never receives a call. `Agent` and `LiveAgent` both assemble their tools
through `resolve_tools`. `Tool.declared_in_code` marks the rank; MCP proxies and
`ClientTool` set it to `False`.

Composite membership is snapshotted immediately after discovery, before awaiting
another tool's schemas. Selection, pruning and registration use that same tree:
another turn refreshing a shared `MCPToolkit` must not change the members of a
turn that is still being resolved. This applies recursively to nested toolkits
and deferred tools in `ToolSearchTool`.
`Toolkit._snapshot()` lets a lazy composite retain the state it just discovered:
`MCPToolkit` binds the snapshot's discovery config to its resolved values, so
generating final schemas cannot refresh the snapshot and reintroduce a dropped tool.

Precedence only holds within a turn, and the tool behind a name can change between turns
(`ask(tools=...)`, an MCP server's changing tool list). So `approval_required` grants an
"always" answer to the implementation behind the call, not to its name: a tool sets
`ToolCallEvent.source` on its copy of the call before its middleware runs (a function
tool from its function's module and qualname, an MCP proxy from its server endpoint and
remote tool name), and the grant is keyed by it. The key is deterministic, so it
survives a restart with the conversation's variables, and one hook shared by several
tools does not carry a grant from one of them to another.

This is the one place that looks inside composites (`Toolkit`, `ToolSearchTool`), which
ADR 0002 otherwise keeps opaque to the agent. Precedence has to act on individual tools,
and a composite registers all its members at once.

### Rejected alternatives

- **Raising on a duplicate name.** It forbids a legitimate override, and it would let an
  MCP server break an agent at runtime by adding a tool.
- **Automatically renaming duplicates** (e.g. a forced source prefix). It silently
  renames tools that prompts and users refer to. `tool_name_prefix` is the explicit form.
- **Dispatching by a per-registration id.** The model and the human approving a call
  still see only the name, so two tools with the same name stay indistinguishable to them.
- **Letting MCP tools override by order.** A remote server's tool list changes without
  any change to the agent's code. Ranking it below code-declared tools means a server can
  never shadow a local, possibly approval-gated, tool. There is no opt-in to reverse this:
  to use the server's tool, remove the local one or set a `tool_name_prefix`.

## Consequences

- `known_tools` can rely on a name identifying one tool within a turn. The approval bypass
  does not rely on names alone, because it outlives the turn.
- An MCP server that starts exposing a name already in use loses that tool, with a
  warning, instead of doubling calls.
- A tool that exposes several names is kept or dropped as a whole. When it loses one of
  its names, it loses all of them.
