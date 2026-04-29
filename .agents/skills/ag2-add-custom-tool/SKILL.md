---
name: ag2-add-custom-tool
description: Add a custom Python tool to an AG2 beta `Agent` using the `@tool` decorator. Use when the user wants to give an Agent a new capability backed by Python code (API calls, DB queries, computations, file ops). Covers sync and async tools, parameter typing, Pydantic schema customisation, returning typed `Input` / `ToolResult` (text / data / images / binary), `final=True` early-exit, and dependency injection via `Context` / `Inject` / `Variable` / `Depends`.
license: Apache-2.0
---

# Add a custom Python tool

## When to use

The user wants their `Agent` to take a real-world action: hit an API, query a database, compute something, return an image. If they want shipped tools (web search, code exec, shell), see `ag2-use-builtin-tools` and `ag2-shell-tool` instead.

## 60-second recipe

```python
from autogen.beta import Agent, tool
from autogen.beta.config import OpenAIConfig

@tool
def calculate_shipping_cost(destination: str, weight_kg: float) -> str:
    """Calculates shipping cost for a package to a destination."""
    return "$15.00"

agent = Agent(
    "shipping",
    prompt="Use tools when helpful.",
    config=OpenAIConfig(model="gpt-4o-mini"),
    tools=[calculate_shipping_cost],
)
```

The `@tool` decorator generates the LLM-facing schema from the function signature, type hints, and docstring. **The docstring is the description the LLM sees** — write it for an LLM reader, not just a human.

You can also pass plain undecorated functions in `tools=[...]` and AG2 wraps them automatically:

```python
def get_weather(location: str) -> str:
    """Returns the current weather for a given location."""
    return "Sunny, 22°C"

agent = Agent("weather", tools=[get_weather])
```

Or attach a tool to an existing agent with `@agent.tool`:

```python
agent = Agent("calc")

@agent.tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b
```

## Sync vs async

Both `def` and `async def` are supported. **Synchronous tools run in a thread by default** so blocking I/O does not freeze the event loop. For ultra-fast pure-Python tools, opt out:

```python
@tool(sync_to_thread=False)
def format_name(first: str, last: str) -> str:
    """Formats a full name."""
    return f"{last.upper()}, {first.capitalize()}"
```

Native async tools run in the main event loop directly:

```python
import aiohttp

@tool
async def fetch(url: str) -> str:
    """Fetches a URL with aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            return await r.text()
```

## Validating inputs with Pydantic `Field`

Use `Annotated[T, Field(...)]` to give the LLM strict bounds. The framework forwards these into the JSON Schema:

```python
from typing import Annotated
from pydantic import Field
from autogen.beta import tool

@tool
def set_temperature(
    temp: Annotated[int, Field(description="Target temperature.", ge=10, le=30)],
    mode: Annotated[str, Field(description="Mode.", pattern="^(heat|cool|auto)$")],
) -> str:
    """Sets the thermostat."""
    return f"Set to {temp}°C in {mode} mode."
```

You can also override the tool name and description on the decorator:

```python
@tool(name="custom_math_tool", description="Performs advanced math.")
def math_op(a: int, b: int) -> int:
    return a + b
```

## Returning typed `Input` / `ToolResult`

A plain `str` return is wrapped in `TextInput` automatically. For richer payloads, return an `Input` subtype or compose with `ToolResult`:

```python
from autogen.beta import DataInput, ImageInput, TextInput, ToolResult, tool

@tool
def get_status(task_id: str) -> TextInput:
    return TextInput(f"Task {task_id} is in progress.")

@tool
def get_user_profile(user_id: str) -> DataInput:
    return DataInput({"id": user_id, "name": "Alice", "role": "admin"})

@tool
def fetch_chart(chart_id: str) -> ImageInput:
    return ImageInput(f"https://charts.example.com/{chart_id}.png")

@tool
def analyze_product(product_id: str) -> ToolResult:
    """Returns image + structured metadata in one tool call."""
    return ToolResult(
        ImageInput(f"https://cdn.example.com/products/{product_id}.jpg"),
        {"id": product_id, "name": "Widget Pro", "stock": 42},
    )
```

For raw bytes of arbitrary format, use `BinaryInput(data=..., media_type="application/pdf")`.

## End the turn early with `final=True`

When the tool already knows the exact final answer, skip the extra LLM round-trip:

```python
from autogen.beta import ToolResult, tool

@tool
def handoff_to_human(ticket_id: str) -> ToolResult:
    """Escalates and returns the final user-facing message verbatim."""
    return ToolResult(f"Ticket {ticket_id} was escalated.", final=True)
```

A `final=True` `ToolResult` must contain exactly one part (`TextInput` or `DataInput`).

## Dependency injection (Context / Inject / Variable / Depends)

Tools can pull execution-time values without exposing them to the LLM. See `references/dependency_injection.md` for the full table; the basics:

```python
from typing import Annotated
from autogen.beta import Context, Inject, Variable, tool

@tool
def query_db(query: str, ctx: Context) -> str:
    """Runs a SQL query."""
    db = ctx.dependencies["db"]
    return db.execute(query)

@tool
def fetch(url: str, http: Annotated[object, Inject("http_session")]) -> str:
    """Fetches with a shared HTTP session."""
    return http.get(url).text

@tool
def send(text: str, api_key: Annotated[str, Variable()]) -> str:
    """Sends a message via the configured channel."""
    ...
```

`Inject` annotations are stripped from the LLM-facing schema — they're an internal injection mechanism.

## Going deeper

- `references/dependency_injection.md` — `Context` vs `Inject` vs `Variable` vs `Depends`, defaults, factories, mutability, overrides.
- `website/docs/beta/tools/tools.mdx` — full `@tool` reference, including the synthesized JSON Schema.
- `website/docs/beta/depends.mdx` — `Depends` lifecycle, yield-based teardown, caching, test overrides.
- `website/docs/beta/inputs/inputs.mdx` — the `Input` factory hierarchy and provider support matrix.
- `website/docs/beta/tools/toolkits.mdx` — bundle related tools into a reusable `Toolkit`.
- `website/docs/beta/tools/tool_middleware.mdx` — async hooks around a single tool (validation, redaction, approval — see also `ag2-hitl`).

## Common pitfalls

- **Vague docstring** — the LLM uses it to decide *when* to call the tool. "Calculates shipping cost based on destination and weight" is much better than "Shipping calc".
- **No type hints** — without them the framework can't generate a useful JSON Schema; the LLM may not call your tool at all.
- **Blocking the event loop** — if you write `def` (sync) tool with heavy CPU or network and pass `sync_to_thread=False`, the loop blocks. Default behaviour (run in a thread) is safe; only opt out for cheap pure-Python work.
- **Function-level imports inside tools** — repo convention disallows them. Hoist `import` to module top.
- **Nested function definitions inside the tool body** — also disallowed (recreates the function on every call).
- **Returning `dict` directly when you wanted structured data** — wrap it in `DataInput(...)` so the framework treats it as structured rather than coercing to text.
- **Forgetting `final=True` requires exactly one part** — combining multiple `Input`s with `final=True` will raise.
