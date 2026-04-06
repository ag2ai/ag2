# test/beta/ Guidelines

## Builtin Tools Testing

### Structure

Provider-specific tool tests live in `test/beta/config/{provider}/tools/`:
- `test_{tool}.py` — e2e tests for supported tools (one file per tool)
- `test_unsupported.py` — all unsupported tools for the provider in one file
- `test_tool_to_api.py` — generic function tool mapping (not builtin-specific)

Variable resolution tests live in `test/beta/tools/test_resolve.py`.

### Test Pattern

Tests must be e2e: instantiate the **Tool** class, call `schemas()`, pass through the provider mapper:

```python
@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "web_search_20250305", "name": "web_search"}
```

Do **not** instantiate schema classes directly in provider tests — always go through the Tool.

### Fixtures

Use the shared `context` pytest fixture from `test/beta/config/conftest.py` (no need to import — pytest discovers it automatically):

```python
async def test_defaults(context: Context) -> None: ...
```

### Coverage Requirements

Every builtin tool must be tested in **every** provider:
- **Supported**: test the happy-path mapping in `test_{tool}.py`
- **Unsupported**: test `UnsupportedToolError` is raised in `test_unsupported.py`

For OpenAI, test both `tool_to_api` (completions) and `tool_to_responses_api` (responses) paths. Group unsupported tests under `TestCompletionsApi` / `TestResponsesApi` classes.

### Variable Resolution

Each tool that accepts `Variable` parameters needs exactly 2 tests in `test/beta/tools/test_resolve.py`:
1. Value resolved from context
2. Missing key raises `KeyError`
