# AG2 Beta Development Guidelines

## Package Structure

`autogen/beta/` is a protocol-driven async agent framework. Key modules:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `agent.py` | Core agent loop and reply handling | `Agent`, `AgentReply` |
| `annotations.py` | Type annotations for dependency injection | `Context`, `Inject`, `Variable` |
| `context.py` | Runtime context (stream, dependencies, variables, prompt) | `Context` dataclass, `Stream` protocol |
| `stream.py` | In-memory event pub/sub | `MemoryStream`, `SubStream` |
| `events/` | Event types for the agent loop | `BaseEvent`, `ModelRequest`, `ModelResponse`, `ToolCallEvent`, `ToolResultEvent`, `Usage`, … |
| `config/` | LLM provider clients (see [below](#llm-provider-clients)) | `ModelConfig`, `LLMClient`, `AnthropicConfig`, `OpenAIConfig`, `GeminiConfig`, … |
| `tools/` | Tool system — builtin + user-defined | `tool`, `Toolkit`, `ToolResult`, `CodeExecutionTool`, `ShellTool`, `WebSearchTool`, … |
| `tools/subagents/` | Agent-to-agent delegation | `subagent_tool`, `run_task`, `depth_limiter`, `persistent_stream`, `StreamFactory` |
| `middleware/` | Request/response interception | `BaseMiddleware`, `Middleware`, `LoggingMiddleware`, `RetryMiddleware`, `TokenLimiter`, `HistoryLimiter`, … |
| `response/` | Structured output validation | `ResponseSchema`, `PromptedSchema`, `ResponseProto`, `response_schema` |
| `history.py` | Conversation history storage | `History`, `Storage`, `MemoryStorage` |
| `hitl.py` | Human-in-the-loop hooks | — |
| `streams/` | Persistent stream backends (e.g. Redis) | — |

### Public API (`autogen.beta`)

Top-level modules:
- `autogen.beta` - top-level module with most basic functionality
- `autogen.beta.types` - Type aliases and constants
- `autogen.beta.config` - LLM provider clients (see [below](#llm-provider-clients))
- `autogen.beta.tools` - Tool system — builtin + user-defined (see [below](#builtin-tools))
- `autogen.beta.tools.subagents` - Agent-to-agent delegation (see [below](#subagent-delegation))
- `autogen.beta.testing` - Testing utilities
- `autogen.beta.middleware` - Request/response interception (see [below](#middleware))

Advanced modules:
- `autogen.beta.events` - Event types for the agent loop
- `autogen.beta.streams` - Persistent stream backends (e.g. Redis)

### Re-export rules

All implementations must be re-exported from their public module's `__init__.py` and listed in `__all__`. If an implementation requires optional dependencies, wrap the import in a `try/except ImportError` block and register a `_missing_optional_dependency_config` fallback (see `autogen/beta/config/__init__.py`, `autogen/beta/middleware/builtin/__init__.py` as the reference pattern). This ensures users get a clear install hint instead of an unexplained `ImportError`.

### Design principles

- **Protocols over inheritance**: `LLMClient`, `ModelConfig`, `Stream`, `Storage`, `Tool` are all `Protocol` classes — implementations satisfy them structurally.
- **Async throughout**: all major operations (`ask`, tool execution, LLM calls) are async. Sync tool functions run via `sync_to_thread`.
- **Event-driven**: all agent-loop communication flows through the `Stream` as typed events.
- **Dependency injection**: all user-provided functions (tools, prompt hooks, HITL, etc.) use `Context`, `Inject`, and `Variable` annotations; resolution is handled by `fast_depends`.

## Builtin Tools

Builtin tools live in `autogen/beta/tools/builtin/`. Each tool has:
- A `ToolSchema` dataclass (provider-neutral capability flag)
- A `Tool` class (constructs the schema, resolves Variables)

### API Design

- Use `version` as the public parameter name on Tool constructors for provider-versioned tools (e.g., `WebSearchTool(version="web_search_20260209")`). The schema field may use a more specific name internally (e.g., `web_search_version`) — the Tool maps between them.
- Tool constructor parameters that accept runtime values must also accept `Variable` for deferred context resolution (e.g., `max_uses: int | Variable | None`).
- Tools with no configurable parameters (e.g., `MemoryTool`, `CodeExecutionTool`) should still accept a `version` keyword argument to allow version pinning.
- Provider mappers in `autogen/beta/config/{provider}/mappers.py` convert `ToolSchema` instances to provider-specific API dicts. Use `t.version` instead of hardcoding version strings.

### Adding a New Builtin Tool

1. Create `autogen/beta/tools/builtin/{tool_name}.py` with a `ToolSchema` dataclass and `Tool` class.
2. Add mapper handling in every provider's mapper:
   - Supported: add an `elif isinstance(t, YourToolSchema)` branch returning the provider-specific dict.
   - Unsupported: the existing fallback `raise UnsupportedToolError(t.type, "provider")` handles it.
3. Add tests for every provider (see test guidelines below).
4. If the tool accepts `Variable` parameters, add 2 tests to `test/beta/tools/test_resolve.py`: one resolving from context, one raising `KeyError` on missing.

## Subagent Delegation

Subagent tools live in `autogen/beta/tools/subagents/` and are imported from `autogen.beta.tools.subagents` (not re-exported from `autogen.beta.tools`).

| File | Purpose |
|------|---------|
| `run_task.py` | `run_task()`, `TaskResult` — execute an agent as a sub-task |
| `subagent_tool.py` | `subagent_tool()`, `StreamFactory` — wrap an agent as a callable tool |
| `depth_limiter.py` | `depth_limiter()` — `ToolMiddleware` to cap nesting depth |
| `persistent_stream.py` | `persistent_stream()` — `StreamFactory` that reuses a stream across calls |

### Agent.as_tool()

`Agent.as_tool(description, name?, stream?, middleware?)` is a convenience method that delegates to `subagent_tool()`. It creates a tool named `task_{agent.name}` with parameters `objective` (required) and `context` (optional).

### depth_limiter

`depth_limiter(max_depth=3)` returns a `ToolMiddleware` that prevents unbounded recursive delegation (A → B → A, or deep chains).

- **Concurrent-safe**: the middleware is read-only — it checks `context.variables["ag:task_depth"]` without mutation. Depth is incremented by `run_task()` in a per-call variables copy, so concurrent sibling calls (dispatched via `asyncio.gather`) get independent counters.
- Attach to `subagent_tool(middleware=[depth_limiter()])` or `agent.as_tool(middleware=[depth_limiter()])`.

### persistent_stream

`persistent_stream()` returns a `StreamFactory` that gives the same agent a consistent stream across multiple invocations within a context. It stores the stream ID in `context.dependencies` keyed by `ag:{agent.name}:stream`, and reuses the parent stream's storage backend.

Use it when sub-task history should accumulate across calls rather than starting fresh each time:

```python
agent.as_tool(description="...", stream=persistent_stream())
```

### Context flow in run_task

| What | Behavior | Why |
|------|----------|-----|
| Dependencies | Copied (`dict.copy()`) | Isolated; child mutations don't affect parent |
| Variables | Copied (new dict); synced back on success (excluding `_DEPTH_KEY`) | Concurrent-safe; user variable mutations propagate back |
| History | Fresh stream per call | Clean context; LLM passes relevant info via `context` parameter |
| Depth counter | Incremented in child copy; excluded from sync-back | Internal bookkeeping; never leaks to parent |

## LLM Provider Clients

Provider clients live in `autogen/beta/config/{provider}/`. Each provider has at least three files:
- `config.py` — a `@dataclass(slots=True)` implementing the `ModelConfig` protocol
- `{provider}_client.py` — a concrete class satisfying the `LLMClient` protocol (async `__call__`)
- `mappers.py` — pure functions for converting messages, tools, response schemas, and usage between internal and provider-specific formats

### Client conventions

- The constructor takes connection params (api_key, base_url, timeout, …) plus a `CreateOptions` TypedDict for generation params. It wraps the provider's async SDK client.
- `__call__` converts messages/tools via mappers, calls the provider API, normalises the response into `ModelResponse` with `Usage`.
- Streaming: emit `ModelMessageChunk` / `ModelReasoning` events via `context.send()` while accumulating the full response.
- Non-streaming: build the complete response directly.

### Mapper conventions

- `convert_messages(messages) -> provider format` — converts `Sequence[BaseEvent]` to the provider's message list.
- `tool_to_api(tool) -> dict` — converts a `ToolSchema` to the provider's tool definition. Use `isinstance()` checks; unsupported tools fall through to `raise UnsupportedToolError(t.type, "provider")`.
- `response_proto_to_*(schema)` — converts `ResponseProto` to the provider's structured-output format. Use `_ensure_additional_properties_false()` where the provider requires it.
- `normalize_usage(usage) -> Usage` — maps provider-specific usage keys to the normalised `Usage` dataclass.

### Adding a new provider

1. Create `autogen/beta/config/{provider}/` with `config.py`, `{provider}_client.py`, and `mappers.py`.
2. Register the config in `autogen/beta/config/__init__.py`: import inside a `try/except ImportError` block and add a `_missing_optional_dependency_config` fallback.
3. Add the config to `__all__`.
4. Add mapper tests under `test/beta/config/{provider}/`

## Testing Conventions

Use `just test-beta` as alias for `pytest` execution to run beta tests.

### Assertion style

Avoid chained field-access assertions like `result[0]["tool_calls"][0]["function"]["arguments"] == {...}`. Instead, compare the whole object directly (`assert msg == {...}`) or use **dirty-equals** `IsPartialDict` when only some fields matter:

```python
# Bad
assert result[0]["role"] == "assistant"
assert result[0]["tool_calls"][0]["function"]["arguments"] == {}

# Good — full comparison
assert result[0] == {"role": "assistant", "tool_calls": [...]}

# Good — partial match with dirty-equals (always use dict syntax, not kwargs)
from dirty_equals import IsPartialDict
assert result[0] == IsPartialDict({"role": "assistant"})  # Good
assert result[0] == IsPartialDict(role="assistant")        # Bad — use dict syntax
```

### Function vs class-based tests

Use **plain functions** for standalone tests. Use **classes** to group multiple related tests that share a logical subject (e.g., `TestImageInput`, `TestBinaryInput`). Do not wrap a single test method in a class — keep it a plain function instead.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (90-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk vitest run          # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->
