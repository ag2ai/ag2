---
description: Patterns for registering tools and functions with AG2 agents
globs: "**/*.py"
alwaysApply: false
---

# AG2 Tool Registration

## Six Approaches

### 1. Decorator Pattern (Most Common)

Register a function for LLM calling and execution on separate agents:

```python
from autogen import ConversableAgent, UserProxyAgent

@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Search the web for information")
def web_search(query: Annotated[str, "The search query"]) -> str:
    return search_api(query)
```

**Important:** `@register_for_llm()` must be the inner decorator (closest to `def`), and `@register_for_execution()` the outer one. The LLM agent decides WHEN to call the tool; the execution agent actually RUNS it.

### 2. `@tool` Decorator (Standalone)

Create a `Tool` object without binding to an agent:

```python
from autogen.tools import tool

@tool(description="Search the web for information")
def web_search(query: Annotated[str, "The search query"]) -> str:
    return search_api(query)

# web_search is now a Tool instance — register later:
web_search.register_for_llm(assistant)
web_search.register_for_execution(user_proxy)
```

Both `name` and `description` are optional — defaults to the function name and docstring.

### 3. Tool Class

```python
from autogen.tools import Tool

tool = Tool(
    name="web_search",
    description="Search the web for information",
    func_or_tool=web_search_function,
)

# Register for both LLM and execution on the same agent:
tool.register_tool(agent)

# Or register separately:
tool.register_for_llm(assistant)
tool.register_for_execution(user_proxy)
```

### 4. Toolkit (Multiple Tools)

```python
from autogen.tools import Toolkit

toolkit = Toolkit([tool1, tool2, tool3])
toolkit.register_for_llm(assistant)
toolkit.register_for_execution(user_proxy)
```

### 5. Module-level Helper

```python
from autogen import register_function

register_function(
    my_func,
    caller=assistant,
    executor=user_proxy,
    description="What my_func does",
)
```

### 6. Functions Parameter (Simplest)

```python
agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    functions=[web_search, calculator],
)
```

Registers functions for LLM only (not execution). Best for single-agent setups.

## Pre-Built Experimental Tools

AG2 includes 20+ ready-to-use tools:

```python
from autogen.tools.experimental import DuckDuckGoSearchTool, QuickResearchTool

# No-config tools work immediately
search = DuckDuckGoSearchTool()
search.register_for_llm(assistant)
search.register_for_execution(user_proxy)

# Some tools need an LLM config
research = QuickResearchTool(llm_config=llm_config)
research.register_for_llm(assistant)
research.register_for_execution(user_proxy)
```

See the imports rule for the full list of available experimental tools.

## Parameter Descriptions

Use `Annotated` with string descriptions for tool parameters:

```python
from typing import Annotated

def calculate(
    expression: Annotated[str, "A mathematical expression to evaluate"],
    precision: Annotated[int, "Number of decimal places"] = 2,
) -> str:
    """Calculate the result of a mathematical expression."""
    return str(round(eval(expression), precision))
```

The function docstring becomes the tool description. Parameter types and annotations are used to generate the JSON schema sent to the LLM.

## Dependency Injection

Use `ChatContext` to access conversation state inside tools:

```python
from autogen.tools import ChatContext, Depends

def context_aware_tool(
    query: str,
    context: ChatContext = Depends(ChatContext),
) -> str:
    messages = context.chat_messages
    last_msg = context.last_message
    return process(query, messages)
```

## Common Mistakes

- Do NOT forget to register for BOTH llm AND execution — the LLM needs the schema, the executor needs the function
- Do NOT use `register_for_llm` and `register_for_execution` on the same agent in a two-agent setup — typically the assistant calls and the user_proxy executes
- Do NOT omit type annotations — they are required for JSON schema generation
- The decorators return `Tool` objects, not the original function. If you need the original function later, keep a reference before decorating.
