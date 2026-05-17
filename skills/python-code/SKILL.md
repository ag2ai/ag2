---
name: python-code
description: AG2 Python coding conventions — type annotations, async patterns, tool authoring, and style rules for the autogen.beta package.
version: 1.0.0
license: Apache-2.0
---

# Python Code

Guidelines for writing Python code in the AG2 `autogen.beta` package. Follow every rule here unless the user explicitly overrides one.

## Type annotations

Always annotate every function signature — parameters and return type:

```python
def resolve(name: str, default: int = 0) -> int | None:
    ...
```

Use `from __future__ import annotations` at the top of every module to enable forward references without quoting them:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
```

Prefer `X | Y` unions (PEP 604) over `Optional[X]` or `Union[X, Y]`. Use `X | None` for optional values.

## Async

Use `async def` / `await` for all I/O-bound operations. Never block the event loop with synchronous I/O inside an `async` context.

When bridging sync and async, use `asyncio.iscoroutine()` to detect coroutines at runtime rather than assuming:

```python
result = fn(**kwargs)
if asyncio.iscoroutine(result):
    result = await result
```

## Pydantic

Use `pydantic` for data models (not `dataclasses`) when the model needs JSON serialization, validation, or schema export.

Use `model_config = ConfigDict(frozen=True)` for value objects that must be immutable.

Never call `.model_dump()` on a plain `dict` — check first or use `isinstance`.

## Tool authoring

Use the `@tool` decorator from `autogen.beta.tools.final`:

```python
from typing import Annotated
from pydantic import Field
from autogen.beta.tools.final import tool

@tool(
    name="find_files",
    description="Search for files matching a glob pattern.",
)
def find_files(
    pattern: Annotated[str, Field(description="Glob pattern, e.g. '**/*.py'.")],
    path: Annotated[str, Field(description="Root directory to search.")] = ".",
) -> list[str]:
    ...
```

Rules:
- Every parameter must be `Annotated[<type>, Field(description="...")]`.
- The description on `@tool` must be a full English sentence ending in a period. It appears verbatim in the LLM's tool schema.
- Prefer returning `str` or a `pydantic` model — LLMs handle these best.
- Sync functions are automatically wrapped for async use; `async def` is fine too.

## Toolkit authoring

Extend `Toolkit` from `autogen.beta.tools.final`:

```python
from autogen.beta.tools.final import Toolkit

class MyToolkit(Toolkit):
    __slots__ = ("_state",)

    def __init__(self, *, middleware=()) -> None:
        self._state = ...
        super().__init__(
            self._tool_a(),
            self._tool_b(),
            name="my_toolkit",
            middleware=(),  # always pass empty tuple; per-tool middleware goes on tools
        )

    def _tool_a(self) -> FunctionTool:
        state = self._state

        @tool(name="tool_a", description="Does A.")
        def _impl(...) -> str:
            ...
        return _impl
```

Use `__slots__` on every toolkit class to avoid accidental attribute leakage.

## Style

- Run `ruff check --fix` and `ruff format` before committing.
- Run `mypy autogen/beta/` to catch type errors.
- Line length: 120 characters (`ruff` default in this repo).
- No `Any` unless unavoidable — prefer `object` or a proper union.
- No bare `except:` — always catch a specific exception type.
- No commented-out code; no `TODO` comments in production paths.
- Use `__all__` to declare public API in every `__init__.py`.

## Imports

Order: stdlib → third-party → local (each separated by a blank line). `ruff` enforces this automatically.

Relative imports inside `autogen.beta` are fine for intra-package references.

## Error handling

Raise descriptive exceptions. The repo defines custom exception types in `autogen.beta.exceptions` — use them instead of generic `ValueError` / `RuntimeError` when a matching one exists.

## Copyright header

Every new `.py` file must start with:

```python
# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
```

## Tests

Every new public symbol must have tests. See the `test-scaffold` skill for test patterns.
