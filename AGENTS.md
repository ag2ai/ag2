# AG2 Development Guidelines

## AI-assisted contribution policy

Before opening a PR, read and follow `.github/AI_POLICY.md`.

- Do not open PRs with unverified AI-generated code or text.
- Ensure the PR description explains the real problem or use case and accurately reflects the diff.
- Include validation and testing information in the PR body.
- Be prepared to explain and revise the contribution in response to reviewer questions.
- Write the PR description using `.github/PULL_REQUEST_TEMPLATE.md`. Keep its section headings (`## Why are these changes needed?`, `## Related issue number`, `## Checks`, `## AI assistance`), fill each one in, and only check a checklist box once it is actually true.

## Architecture Decision Records (ADR)

Cross-cutting and hard-to-reverse design decisions are recorded in `docs/adr/`, sequentially numbered (`0001-*.md`, `0002-*.md`, …) with `status` / `date` frontmatter and a short Context / Decision / Consequences body.

- **Consult them before changing established public API or architecture.** They explain *why* something is the way it is — e.g. `0003-eval-run-api-takes-agent-instances.md` records that the eval `run_*` API takes prebuilt `Agent` instances (not factories) and explicitly-built `Suite`s. If a change contradicts an ADR, supersede it rather than silently reverting the code.
- **Add one when a decision qualifies**: it is hard to reverse, surprising without context (a reader would assume the opposite), and the result of a real trade-off. Scan `docs/adr/` for the highest number and increment. Keep it short — recording *that* a decision was made and *why* is the value.

## Code Style Guidelines

- Do not use `from __future__ import annotations`.
- Do not use global variables or top-level side-effect function calls unless the user explicitly allows it.
- For filesystem paths, use `pathlib.Path` internally. Public signatures should accept `str | os.PathLike[str]`.
- Top-level imports from `ag2.*` are for common APIs that are broadly reusable across scenarios and core agent flows.
  Good: `ag2.[Input]` — common structures usable in `await agent.ask(Input())` and as tool results.
  Bad: `ag2.middleware.BaseMiddleware` — this is advanced/specialized and should be imported only when implementing custom middleware.
- Do not use function-level imports unless the user explicitly allows it.
  ```python
  # === BAD - import inside function ===
  def execute_tool():
      from .tool import Tool

      ...


  # === GOOD - top-level import ===
  from .tool import Tool


  def execute_tool(): ...
  ```
- Do not create nested functions inside runtime execution paths.
  ```python
  # === BAD - function will be created each call ===
  def execute_tool():
      def _inner_function():
          pass

      _inner_function()


  # === GOOD - function created once, executed each call ===
  def execute_tool():
      _inner_function()


  def _inner_function():
      pass


  # === GOOD - decorator executed import time, so we can use closure functions here ===
  def decorator(func):
      def wrapper():
          return func()

      return wrapper
  ```
- Do not perform side effects in initialization methods. Apply side effects only at runtime.
  ```python
  # === BAD - create directory in initial method ===
  class KnowledgeStore:
      def __init__(self, path: str | os.PathLike[str]) -> None:
          self.path = Path(path)
          # side effect - directory creation
          self.path.parent.mkdir(parents=True, exist_ok=True)

      def run(self) -> None: ...


  # === GOOD - create directory in runtime method ===
  class KnowledgeStore:
      def __init__(self, path: str | os.PathLike[str]) -> None:
          self.path = Path(path)

      def run(self) -> None:
          self.path.parent.mkdir(parents=True, exist_ok=True)
          ...
  ```

## Package Structure

`ag2/` is a protocol-driven async agent framework. Key modules:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `agent.py` | Core agent loop and reply handling | `Agent`, `AgentReply` |
| `annotations.py` | Type annotations for dependency injection | `Context`, `Inject`, `Variable` |
| `context.py` | Runtime context (stream, dependencies, variables, prompt) | `Context` dataclass, `Stream` protocol |
| `stream.py` | In-memory event pub/sub | `MemoryStream`, `SubStream` |
| `events/` | Event types for the agent loop | `BaseEvent`, `ModelRequest`, `ModelResponse`, `ToolCallEvent`, `ToolResultEvent`, `Usage`, … |
| `config/` | LLM provider clients (see [below](#llm-provider-clients)) | `ModelConfig`, `LLMClient`, `AnthropicConfig`, `OpenAIConfig`, `GeminiConfig`, … |
| `tools/` | Tool system — builtin + user-defined | `tool`, `Toolkit`, `ToolResult`, `CodeExecutionTool`, `ShellTool`, `WebSearchTool`, … |
| `tools/subagents/` | Agent-to-agent delegation | `subagent_tool`, `run_task`, `persistent_stream`, `StreamFactory` |
| `eval/` | Offline evaluation framework | `run`, `scorer`, `EvalTarget`, `Suite`, `Task`, `Trace`, `RunResult`, `Feedback`, `BudgetThresholds`, plus prebuilts under `eval.scorers` |
| `middleware/` | Request/response interception | `BaseMiddleware`, `Middleware`, `LoggingMiddleware`, `RetryMiddleware`, `TokenLimiter`, `HistoryLimiter`, … |
| `response/` | Structured output validation | `ResponseSchema`, `PromptedSchema`, `ResponseProto`, `response_schema` |
| `history.py` | Conversation history storage | `History`, `Storage`, `MemoryStorage` |
| `hitl.py` | Human-in-the-loop hooks | — |
| `streams/` | Persistent stream backends (e.g. Redis) | — |

### Public API (`ag2`)

Top-level modules:
- `ag2` - top-level module with most basic functionality
- `ag2.types` - Type aliases and constants
- `ag2.config` - LLM provider clients (see [below](#llm-provider-clients))
- `ag2.tools` - Tool system — builtin + user-defined (see [below](#builtin-tools))
- `ag2.tools.subagents` - Agent-to-agent delegation (see [below](#subagent-delegation))
- `ag2.testing` - Testing utilities
- `ag2.middleware` - Request/response interception (see [below](#middleware))
- `ag2.observer` - Reusable observer implementations
- `ag2.eval` - Offline evaluation framework (datasets, scorers, runner, persistence)

Advanced modules:
- `ag2.events` - Event types for the agent loop
- `ag2.streams` - Persistent stream backends (e.g. Redis)
- `ag2.watch` - Watch system for triggering observers
- `ag2.knowledge` - Knowledge management
- `ag2.plugin` - Plugin system

### Re-export rules

All implementations must be re-exported from their public module's `__init__.py` and listed in `__all__`. If an implementation requires third-party dependencies, wrap the import in a `try/except ImportError` block and register a missing-dependency fallback so users get a clear install hint instead of an unexplained `ImportError` (see `ag2/config/__init__.py`, `ag2/middleware/builtin/__init__.py` as the reference pattern). Two fallbacks exist:

- **Core modules** use **optional dependencies** shipped as pyproject extras — fall back via `missing_optional_dependency`, which hints `pip install "ag2[<extra>]"`.
- **Extensions** (`ag2/extensions/`) are **not** shipped as extras — declare their third-party packages as **additional dependencies** and fall back via `missing_additional_dependency`, which hints the upstream package directly (e.g. `pip install "daytona>=0.171.0,<1"`).

### Design principles

- **Protocols over inheritance**: `LLMClient`, `ModelConfig`, `Stream`, `Storage`, `Tool` are all `Protocol` classes — implementations satisfy them structurally.
- **Async throughout**: all major operations (`ask`, tool execution, LLM calls) are async. Sync tool functions run via `sync_to_thread`.
- **Event-driven**: all agent-loop communication flows through the `Stream` as typed events.
- **Dependency injection**: all user-provided functions (tools, prompt hooks, HITL, etc.) use `Context`, `Inject`, and `Variable` annotations; resolution is handled by `fast_depends`.

## Builtin Tools

Builtin tools live in `ag2/tools/builtin/`. Each tool has:
- A `ToolSchema` dataclass (provider-neutral capability flag)
- A `Tool` class (constructs the schema, resolves Variables)

### API Design

- Use `version` as the public parameter name on Tool constructors for provider-versioned tools (e.g., `WebSearchTool(version="web_search_20260209")`). The schema field may use a more specific name internally (e.g., `web_search_version`) — the Tool maps between them.
- Tool constructor parameters that accept runtime values must also accept `Variable` for deferred context resolution (e.g., `max_uses: int | Variable | None`).
- Tools with no configurable parameters (e.g., `MemoryTool`, `CodeExecutionTool`) should still accept a `version` keyword argument to allow version pinning.
- Provider mappers in `ag2/config/{provider}/mappers.py` convert `ToolSchema` instances to provider-specific API dicts. Use `t.version` instead of hardcoding version strings.

### Adding a New Builtin Tool

1. Create `ag2/tools/builtin/{tool_name}.py` with a `ToolSchema` dataclass and `Tool` class.
2. Add mapper handling in every provider's mapper:
   - Supported: add an `elif isinstance(t, YourToolSchema)` branch returning the provider-specific dict.
   - Unsupported: the existing fallback `raise UnsupportedToolError(t.type, "provider")` handles it.
3. Add tests for every provider (see test guidelines below).
4. If the tool accepts `Variable` parameters, add 2 tests to `test/tools/test_resolve.py`: one resolving from context, one raising `KeyError` on missing.

## Subagent Delegation

Subagent tools live in `ag2/tools/subagents/` and are imported from `ag2.tools.subagents` (not re-exported from `ag2.tools`).

| File | Purpose |
|------|---------|
| `run_task.py` | `run_task()`, `TaskResult` — execute an agent as a sub-task |
| `subagent_tool.py` | `subagent_tool()`, `StreamFactory` — wrap an agent as a callable tool |
| `persistent_stream.py` | `persistent_stream()` — `StreamFactory` that reuses a stream across calls |

### Agent.as_tool()

`Agent.as_tool(description, name?, stream?, middleware?)` is a convenience method that delegates to `subagent_tool()`. It creates a tool named `task_{agent.name}` with parameters `objective` (required) and `context` (optional).

### Auto-injected `run_subtask` / `run_subtasks`

Sub-task delegation is **off by default** (`tasks=False`). Pass `tasks=TaskConfig(...)` to opt in, and the `Agent` gains a `run_subtask(task)` and a `run_subtasks(tasks=[...], parallel=True)` tool. Each call spawns a **subtask Agent** that:

- Inherits the parent's user-supplied tools by default (filterable via `TaskConfig.include_tools` / `exclude_tools`, extendable via `extra_tools`).
- Is itself constructed with `tasks=False` (the default), so the subtask has **no** `run_subtask` tools — recursive delegation is structurally impossible. No depth limiting required.
- Runs on its own `MemoryStream`; child events do not leak into the parent's stream beyond `TaskStarted` / `TaskCompleted` / `TaskFailed` lifecycle events.

The LLM is told (via the tool description) that `run_subtask` may be invoked multiple times in parallel within a single response, encouraging the parallel-tool-use pattern Anthropic recommends.

### persistent_stream

`persistent_stream()` returns a `StreamFactory` that gives the same agent a consistent stream across multiple invocations within a context. It stores the stream ID in `context.dependencies` keyed by `ag:{agent.name}:stream`, and reuses the parent stream's storage backend.

Use it when sub-task history should accumulate across calls rather than starting fresh each time:

```python
agent.as_tool(description="...", stream=persistent_stream())
```

### Context flow in run_task

| What | Behavior | Why |
|------|----------|-----|
| Dependencies | Shallow-copied (`dict.copy()`) | Isolated at the top level; mutable values are still shared by reference. Treat dependencies as read-only inside subtasks. |
| Variables | Copied (new dict); **not** synced back | Concurrent siblings via `asyncio.gather` would race-clobber a shared dict — last-writer-wins is silent data loss. Each subtask's mutations stay scoped to it. |
| History | Fresh stream per call | Clean context; LLM passes relevant info via `context` parameter |
| Tool inheritance | Parent's user-supplied tools (filtered by `TaskConfig`) | Subtasks need real capabilities to do work; child has no `run_subtask` tools so recursion is impossible |

## LLM Provider Clients

Provider clients live in `ag2/config/{provider}/`. Each provider has at least three files:
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

1. Create `ag2/config/{provider}/` with `config.py`, `{provider}_client.py`, and `mappers.py`.
2. Register the config in `ag2/config/__init__.py`: import inside a `try/except ImportError` block and add a `_missing_optional_dependency_config` fallback.
3. Add the config to `__all__`.
4. Add mapper tests under `test/config/{provider}/`
