---
name: test-scaffold
description: AG2 test patterns — pytest fixtures, async tests, toolkit schema tests, and end-to-end tool call verification.
version: 1.0.0
license: Apache-2.0
---

# Test Scaffold

Patterns and conventions for writing tests in `test/beta/`.

## File structure

```
test/beta/
├── conftest.py        # Shared fixtures
├── test_<module>.py   # One file per source module
└── tools/
    └── test_<toolkit>.py
```

Every test file must start with the AG2 copyright header:

```python
# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
```

## Shared fixtures

Available everywhere via `conftest.py`:

| Fixture | Type | Description |
| :--- | :--- | :--- |
| `context` | `Context` | A minimal `Context` for schema / tool registration tests |
| `async_mock` | factory | Returns a coroutine-returning mock — use for patching async callables |

Import them by name in your test function signature — pytest injects them automatically.

## Async tests

Mark async tests with `@pytest.mark.asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_feature(context) -> None:
    result = await some_async_call()
    assert result == expected
```

## Testing a toolkit's schemas

Call `await toolkit.schemas(context)` to get the tool schemas and assert their structure:

```python
@pytest.mark.asyncio
async def test_toolkit_exposes_expected_tools(context) -> None:
    toolkit = MyToolkit()

    schemas = await toolkit.schemas(context)

    names = {s.function.name for s in schemas}
    assert names == {"tool_a", "tool_b"}
```

For detailed schema assertions use `dirty_equals.IsPartialDict`:

```python
from dirty_equals import IsPartialDict

@pytest.mark.asyncio
async def test_tool_schema_shape(context) -> None:
    [schema] = await my_tool.schemas(context)

    assert asdict(schema) == {
        "type": "function",
        "function": IsPartialDict({
            "name": "my_tool",
            "parameters": IsPartialDict({
                "properties": IsPartialDict({
                    "query": IsPartialDict({"type": "string"}),
                }),
                "required": ["query"],
            }),
        }),
    }
```

## End-to-end tool call tests

Use `TrackingConfig` and `ToolCallEvent` to verify the agent actually invokes a tool:

```python
from autogen.beta.config import TrackingConfig
from autogen.beta.events import ToolCallEvent

@pytest.mark.asyncio
async def test_agent_calls_my_tool() -> None:
    config = TrackingConfig()
    agent = Agent("assistant", config=config, tools=[my_toolkit])

    await agent.ask("Do the thing.")

    tool_calls = [e for e in config.events if isinstance(e, ToolCallEvent)]
    assert any(e.name == "my_tool" for e in tool_calls)
```

`TrackingConfig` records every event without making real LLM calls — it replies with a scripted tool call + result sequence.

## Testing storage-backed classes

Test persistence classes (`_SQLiteStore`, etc.) directly without routing through the full agent:

```python
import tempfile
from pathlib import Path

def test_sqlite_store_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _SQLiteStore(Path(tmpdir) / "test.db")

        store.store("key", "value")
        content, _ = store.retrieve("key")

        assert content == "value"
```

Always use `tmp_path` (pytest fixture) or `tempfile.TemporaryDirectory` for files — never hardcode `/tmp`.

## Parametrized tests

Parametrize when the same assertion applies across several inputs:

```python
@pytest.mark.parametrize("bad_name", ["", "Has Spaces", "CamelCase", "has/slash"])
def test_reject_bad_names(bad_name) -> None:
    with pytest.raises((ValueError, InvalidSkillNameError)):
        validate_name(bad_name)
```

## Generating a test scaffold

Use the `scaffold` script to generate a starter test file for a module:

```bash
python skills/test-scaffold/scripts/scaffold.py autogen.beta.tools.toolkits.memory
```

This reads the module's public symbols and emits a `test_memory.py` template with one empty test per public class/function.
