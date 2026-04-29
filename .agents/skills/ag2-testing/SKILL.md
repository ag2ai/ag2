---
name: ag2-testing
description: Test AG2 beta agents and tools without hitting a real LLM provider. Pass `TestConfig(...)` from `autogen.beta.testing` as the agent's config (or per-`ask`) to mock LLM responses, inject `ToolCallEvent`s to simulate tool execution, and assert success / error paths. Use when the user is writing pytest tests for an Agent or Tool.
license: Apache-2.0
---

# Testing agents and tools

## When to use

Writing tests for code that builds AG2 beta `Agent`s, custom `@tool` functions, middleware, or response schemas — anywhere you don't want to make real LLM API calls.

## 60-second recipe — mock an LLM response

```python
import pytest
from autogen.beta import Agent
from autogen.beta.testing import TestConfig

@pytest.mark.asyncio
async def test_mocked_response():
    agent = Agent("test_agent")
    reply = await agent.ask("Hi!", config=TestConfig("This is a mocked response."))
    assert reply.body == "This is a mocked response."
```

`TestConfig(*responses)` replaces the model client. Each positional arg is the mocked response for the next sequential turn — strings for text replies, `ToolCallEvent` for tool dispatches.

## Simulate a successful tool call

Pass a `ToolCallEvent` first (the model "decides" to call the tool), then the final answer:

```python
import pytest
from autogen.beta import Agent
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig

@pytest.mark.asyncio
async def test_tool_success():
    def my_tool() -> str:
        return "tool execution result"

    agent = Agent("test_agent", tools=[my_tool])
    config = TestConfig(
        ToolCallEvent(name="my_tool"),
        "final result",
    )
    reply = await agent.ask("Please use my_tool", config=config)
    assert reply.body == "final result"
```

## Test tool error paths

If a tool raises, the exception propagates to `ask()`:

```python
@pytest.mark.asyncio
async def test_tool_raises():
    def failing_tool() -> str:
        raise ValueError("Something went wrong")

    config = TestConfig(
        ToolCallEvent(name="failing_tool"),
        "result",
    )
    agent = Agent("test_agent", config=config, tools=[failing_tool])

    with pytest.raises(ValueError, match="Something went wrong"):
        await agent.ask("Hi!")
```

## Tool not found

If the LLM calls a tool the agent doesn't have, the framework raises `ToolNotFoundError`:

```python
from autogen.beta.exceptions import ToolNotFoundError

@pytest.mark.asyncio
async def test_tool_not_found():
    config = TestConfig(ToolCallEvent(name="unregistered_tool"))
    agent = Agent("test_agent", config=config)
    with pytest.raises(ToolNotFoundError, match="Tool `unregistered_tool` not found"):
        await agent.ask("Hi!")
```

## Useful test patterns

### Override `Depends` dependencies

```python
def get_production_db():
    raise Exception("Do not call in tests!")

@tool
def read_data(db: Annotated[object, Depends(get_production_db)]) -> str:
    return "Data"

agent = Agent("test", tools=[read_data])
agent.dependency_provider.override(get_production_db, lambda: "mock_db")
```

### Override `Inject` dependencies

Just pass `dependencies={...}` to `agent.ask(...)`:

```python
await agent.ask("Read", dependencies={"database_pool": fake_pool})
```

### Capture stream events

```python
from autogen.beta import MemoryStream
from autogen.beta.events import ToolCallEvent

stream = MemoryStream()
collected: list[ToolCallEvent] = []
stream.where(ToolCallEvent).subscribe(lambda e: collected.append(e))

await agent.ask("Test", stream=stream)
assert collected[0].name == "expected_tool"
```

### Multi-turn mock

Each positional arg in `TestConfig(...)` corresponds to one model response. For a multi-turn test, supply enough responses for each turn the test exercises.

## Going deeper

- Source doc: `website/docs/beta/testing.mdx`.
- Test markers / async config — repo `pyproject.toml`. Use `@pytest.mark.asyncio` (the project uses pytest-asyncio).
- Streams (for asserting events): `website/docs/beta/advanced/stream.mdx`.

## Common pitfalls

- **Forgetting `@pytest.mark.asyncio`** — the test will skip or fail oddly.
- **Mismatched response count** — `TestConfig` runs out of responses if the agent makes more LLM calls than you expect (e.g. tool error → another LLM call). Add more positional args or assert that the call sequence is what you intended.
- **Mocking the LLM but not the tool** — your tool function still runs (and may hit real APIs / disk). Mock the *tool* if you're isolating LLM behaviour, or override its `Depends` to inject test doubles.
- **Asserting on `reply.body` when you set a `response_schema`** — `body` is the raw text. Use `await reply.content()` for the validated value.
- **Sharing `Agent` instances across async tests** — agents carry mutable state (variables, dependencies). Construct fresh agents per test for isolation.
- **Using real provider clients in CI** — wrap the provider config with `TestConfig` per-test or via a fixture; never rely on `OPENAI_API_KEY` etc. being available in test environments.
