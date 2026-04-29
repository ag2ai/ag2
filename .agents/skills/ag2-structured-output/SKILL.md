---
name: ag2-structured-output
description: Get a typed Python value back from an AG2 beta `Agent` instead of free text. Pass `response_schema=` (a Pydantic model, dataclass, primitive, union, `ResponseSchema`, or `@response_schema` validator) and read the parsed result via `await reply.content()`. Use when the user wants validated structured output, classification, extraction, or scoring. Covers `ResponseSchema`, `@response_schema`, `PromptedSchema` (for providers without native structured output), per-turn override, validation retries, and primitive embedding.
license: Apache-2.0
---

# Structured output

## When to use

- The user wants a Pydantic model, dataclass, dict, primitive, or union back — not a string.
- They're doing classification, extraction, scoring, normalisation, or anything where downstream code parses the reply.
- They want automatic retry on validation failure.

## 60-second recipe

```python
from pydantic import BaseModel, Field
from typing import Annotated

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig

class TicketTriage(BaseModel):
    category: Annotated[str, Field(description="e.g. billing, bug, account_access")]
    urgency: Annotated[str, Field(description="low, medium, or high")]
    summary_one_line: Annotated[str, Field(description="Max 120 characters", max_length=120)]

agent = Agent(
    "triage",
    prompt="You triage support messages. Be conservative with urgency.",
    config=OpenAIConfig(model="gpt-4o-mini"),
    response_schema=TicketTriage,
)

reply = await agent.ask("I was charged twice and can't export reports. Quarter close blocked.")
triage = await reply.content()      # → typed TicketTriage
print(triage.category, triage.urgency)
```

`reply.body` is still the raw model text; `await reply.content()` runs validation and returns the parsed value. If validation fails, `content()` raises (e.g. `pydantic.ValidationError`).

## Schema types you can pass

| Type | What you get |
|---|---|
| Primitive (`int`, `float`, `bool`) | Bare value, framework wraps in `{"data": ...}` for the API |
| `dataclass` | Instance of the dataclass |
| Pydantic `BaseModel` | Instance of the model |
| Union (`int \| str`, `(int, str)`) | One of the alternatives |
| `dict[K, V]`, `TypedDict` | Validated dict |
| `ResponseSchema(...)` | Same as above, with explicit `name` / `description` for the provider |
| `@response_schema` callable | Custom validation/parsing logic |
| `PromptedSchema(inner)` | Schema injected into the system prompt for providers without native structured output |

## `ResponseSchema` — name your payload

Helps the provider treat the structured output as a named contract:

```python
from autogen.beta import Agent, ResponseSchema

schema = ResponseSchema(int | str, name="ByteWidth", description="Number of bits in one byte.")
agent = Agent("assistant", config=config, response_schema=schema)
```

## `@response_schema` — custom validation

For clamping, regex cleanup, decoding wrapped JSON, or combining fields:

```python
from autogen.beta import Agent, response_schema

@response_schema
def parse_rating(content: str) -> int:
    """Parse a rating and clamp to 1–5."""
    return max(1, min(5, int(content)))

agent = Agent("assistant", config=config, response_schema=parse_rating)
```

Multi-parameter form synthesises a JSON object schema from the parameter names:

```python
from typing import Annotated
from pydantic import Field
from autogen.beta import response_schema

@response_schema
def extract_listing(
    title: Annotated[str, Field(description="Product name")],
    price_usd: Annotated[float, Field(description="Price in USD", ge=0)],
    in_stock: Annotated[bool, Field(description="True if it ships now")],
) -> dict:
    return {"title": title, "price_usd": price_usd, "in_stock": in_stock}
```

The function also participates in dependency injection — `Context`, `Variable`, `Inject`, `Depends` work the same way as in tools (and don't appear in the JSON schema).

Async validators are supported:

```python
import json

@response_schema
async def fetch_and_validate(content: str) -> dict:
    data = json.loads(content)
    data["validated"] = True
    return data
```

## `PromptedSchema` — for providers without native structured output

Injects the JSON schema into the system prompt instead of using `response_format`:

```python
from autogen.beta import Agent, PromptedSchema

agent = Agent("assistant", config=config, response_schema=PromptedSchema(int))
```

Wraps any inner schema (type, `ResponseSchema`, `@response_schema` callable). The validation logic stays the same; only the wire format changes.

Custom prompt template:

```python
PromptedSchema(int, prompt_template="Reply with JSON matching this schema:\n{schema}")
```

## Per-turn override

```python
agent = Agent("assistant", config=config)

turn = await agent.ask("How many seconds in a minute?", response_schema=int)
print(await turn.content())   # 60

turn2 = await turn.ask("Say hello.")     # back to default (no schema)
```

Pass `response_schema=None` to drop a schema set on the agent for one call.

## Retries

When validation fails, automatically re-ask the model:

```python
result = await reply.content(retries=3)   # initial + up to 3 re-asks
result = await reply.content(retries=math.inf)   # interactive only — could loop forever
```

The validation error is sent back to the model as a follow-up so it can correct itself.

## Primitive embedding (`embed`)

Bare primitives (`int`, `float`, `bool`, `list[T]`, primitive unions) get wrapped in `{"data": ...}` by default — most structured-output APIs handle objects more reliably than bare values. `content()` transparently unwraps. Opt out:

```python
ResponseSchema(int, name="RawInt", embed=False)              # model must produce a bare 42
@response_schema(embed=False)
def parse_rating(value: int) -> int: ...
```

## Going deeper

- Working starter: `assets/recipe_builder.py` (mirrors `code_examples/02`) — Pydantic model + `@tool` + `response_schema=`.
- Full reference: `website/docs/beta/structured_output.mdx` — covers every schema type, multi-param `@response_schema`, `Field` constraints, `PromptedSchema`, retries, embedding semantics.

## Common pitfalls

- **Reading `reply.body` when you wanted typed output** — `reply.body` is the raw text. `await reply.content()` does the parsing.
- **Forgetting `await` on `content()`** — it's async; you'll get a coroutine, not the value.
- **No `description` in the Pydantic field** — the LLM may guess what to put in each field. Add a `Field(description=...)` for every non-obvious key.
- **Provider doesn't support native structured output** — wrap with `PromptedSchema(...)` rather than fighting the API.
- **`retries=math.inf` in production** — will loop forever on a model that can't comply. Use a finite count.
- **Per-turn override is single-turn** — passing `response_schema=int` to one `ask()` doesn't change the agent's default. The next turn returns to whatever was set on the constructor.
