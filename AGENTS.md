# AG2 Beta Development Guidelines

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
