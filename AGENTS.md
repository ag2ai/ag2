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
| `middleware/` | Request/response interception | `BaseMiddleware`, `Middleware`, `LoggingMiddleware`, `RetryMiddleware`, `TokenLimiter`, `HistoryLimiter`, … |
| `response/` | Structured output validation | `ResponseSchema`, `PromptedSchema`, `ResponseProto`, `response_schema` |
| `history.py` | Conversation history storage | `History`, `Storage`, `MemoryStorage` |
| `hitl.py` | Human-in-the-loop hooks | — |
| `streams/` | Persistent stream backends (e.g. Redis) | — |

### Public API (`autogen.beta`)

Top-level modules:
- `autogen.beta` - top-level module with most basic functionality
- `autogen.beta.config` - LLM provider clients (see [below](#llm-provider-clients))
- `autogen.beta.tools` - Tool system — builtin + user-defined (see [below](#builtin-tools))
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
