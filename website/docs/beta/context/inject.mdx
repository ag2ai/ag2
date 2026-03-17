---
title: Dependency Injection
sidebarTitle: Inject
---

# Dependency Injection

## What is Dependency Injection?

Dependency Injection (DI) is a design pattern used to pass complex objects or services into your tools at runtime, rather than hardcoding them or re-instantiating them repeatedly. This keeps your tools pure, testable, and completely decoupled from external resource management.

## Dependencies

The key difference between [Variables](../variables){.internal-link} and Dependencies is their intended use case. Variables are designed to pass lightweight, serializable state (like strings, flags, or configuration IDs) between the LLM, tools, and agents. Dependencies, on the other hand, are meant for complex objects that have specific behaviors and lifecycle management, such as database connections, HTTP sessions, or clients for external APIs.

### Agent Dependencies

If a dependency is required across all conversations for a specific agent, you can provide it directly when initializing the agent.

```python linenums="1"
from autogen.beta import Agent
import aiohttp

# The session object is available to all tool calls made by this agent
agent = Agent(
    name="WebScraper",
    dependencies={"http_session": aiohttp.ClientSession()}
)
```

### Conversation Dependencies

If a dependency is only relevant for a single conversation, you can inject it dynamically when calling the `ask` method.

```python linenums="1"
await agent.ask(
    "Query the user database",
    # The db_connection is injected only for this specific conversation
    dependencies={"db": create_db_connection()}
)
```

### Mixed Dependencies

When you provide both agent-level and conversation-level dependencies, the framework automatically merges them. If there is a key collision, the dependencies provided during the `ask` call take precedence and override the agent's default dependencies.

```python linenums="1" hl_lines="3 9"
agent = Agent(
    name="DataAgent",
    dependencies={"default_db": db_1, "active_db": db_1}
)

# During this call, "active_db" is overridden by db_2
await agent.ask(
    "Check the backup database",
    dependencies={"active_db": db_2}
)
```

## Context Dependency Access

The simplest way to access your dependencies inside a tool is through the `Context` object. By adding an argument annotated with `Context`, the framework injects the current execution context, which includes the `.dependencies` dictionary.

```python linenums="1" hl_lines="1 4 6"
from autogen.beta import Context, tool

@tool
def query_database(query: str, context: Context) -> str:
    # Access the complex dependency directly from the context
    db = context.dependencies.get("db")

    result = db.execute(query)
    return f"Result: {result}"
```

## Accessing Dependencies with Inject

Instead of passing the entire context object, you can explicitly request specific dependencies directly in your tool's function signature. This approach clarifies your tool's requirements and automatically handles validation.

Use the `Inject` annotation to pull a dependency from the context dictionary by its key. By default, `Inject` looks for a key that matches the argument's name.

```python linenums="1" hl_lines="8"
from typing import Annotated
from autogen.beta import Inject, tool

@tool
def fetch_data(
    url: str,
    # Automatically looks for "http_session" in the dependencies dictionary
    http_session: Annotated[object, Inject()]
) -> str:
    pass
```

If your argument name differs from the dependency key, you can provide the exact key explicitly:

```python linenums="1" hl_lines="5"
@tool
def fetch_data(
    url: str,
    # Looks for "http_session", but assigns it to the "session" argument
    session: Annotated[object, Inject("http_session")]
) -> str:
    pass
```

!!! note "LLM Tool Schema"
    Dependency injection annotations (like `Inject` and `Depends`) do not affect the tool schema provided to the LLM. They are purely an internal framework mechanism for injecting dependencies into your functions.

## Dependencies with Default Values

Sometimes a dependency might not be provided by the user. You can define fallback behaviors directly within the `Inject` annotation using either `default` or `default_factory`. Without a default, a missing dependency will raise a `ValidationError`.

### Static Defaults

Use `default` for simple, immutable fallback values.

```python linenums="1" hl_lines="4"
@tool
def process_data(
    # If "client" is not provided in dependencies, it defaults to None
    client: Annotated[object | None, Inject(default=None)]
) -> str:
    if client is None:
        return "No client provided."
    return "Processing..."
```

### Dynamic Defaults

For mutable objects or dependencies that need to be instantiated at runtime, use `default_factory`. The framework ensures that the factory function is called when the dependency is missing.

```python linenums="1" hl_lines="7"
def create_default_client() -> object:
    return DefaultDatabaseClient()

@tool
def update_record(
    # Uses the dynamically created client if "db" wasn't provided
    db: Annotated[object, Inject(default_factory=create_default_client)]
) -> str:
    db.save()
    return "Record updated"
```
