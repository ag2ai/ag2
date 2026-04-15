# test/beta/ Guidelines

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

### Imports

All imports must be at the top of the test file. Never place imports inside individual test functions until user asks for it.

### Function vs class-based tests

Use **plain functions** for standalone tests. Use **classes** to group multiple related tests that share a logical subject (e.g., `TestImageUrlInput`, `TestBinaryInput`). Do not wrap a single test method in a class — keep it a plain function instead.

If you need to apply markers to each test in class, apply them to the class itself.

```python
# Bad - markers are applied to each test individually
class TestAgent:
    @pytest.mark.asyncio
    async def test_defaults(self, context: Context) -> None: ...

    @pytest.mark.asyncio
    async def test_defaults(self, context: Context) -> None: ...

# Good - markers are applied to the class itself
@pytest.mark.asyncio
class TestAgent:
    async def test_defaults(self, context: Context) -> None: ...

    async def test_defaults(self, context: Context) -> None: ...
```

### Section comments

Do not use banner-style section dividers (e.g. `# ---\n# Section\n# ---`). Class names and test names are sufficient structure.

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
