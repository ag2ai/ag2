---
name: add-tool
description: Add a properly typed and registered tool function to an existing AG2 agent setup. Generates the function with schema, type annotations, and correct registration on both caller and executor agents.
---

# Add Tool to AG2 Agent

You are an expert at adding tools to AG2 agents. When the user asks to add a tool/function:

## 1. Understand the Requirements

- What should the tool do?
- Which agent should CALL it (register_for_llm)?
- Which agent should EXECUTE it (register_for_execution)?
- What parameters does it need?
- What does it return?

## 2. Generate the Tool

Follow this pattern exactly:

```python
from typing import Annotated

@executor_agent.register_for_execution()
@caller_agent.register_for_llm(description="Clear description of what this tool does")
def tool_name(
    param1: Annotated[str, "Description of param1"],
    param2: Annotated[int, "Description of param2"] = default_value,
) -> str:
    """Detailed description used as the tool's description for the LLM."""
    # Implementation
    return result
```

## 3. Rules

- Always use `Annotated[type, "description"]` for every parameter — this generates the JSON schema
- The function docstring becomes the tool description if no `description=` is passed to `register_for_llm()`
- Return type should be `str` for simple tools — LLMs work best with string outputs
- `@register_for_llm()` must be the INNER decorator (closest to `def`)
- `@register_for_execution()` must be the OUTER decorator
- For complex return types, serialize to JSON string
- If the tool needs conversation context, use dependency injection:

```python
from autogen.tools import ChatContext, Depends

@executor.register_for_execution()
@caller.register_for_llm(description="Context-aware tool")
def my_tool(
    query: Annotated[str, "The query"],
    context: ChatContext = Depends(ChatContext),
) -> str:
    messages = context.chat_messages
    return process(query, messages)
```

## 4. Alternative: Tool Class

If the user prefers explicit Tool objects:

```python
from autogen.tools import Tool

tool = Tool(
    name="tool_name",
    description="What it does",
    func_or_tool=my_function,
)
tool.register_for_llm(caller_agent)
tool.register_for_execution(executor_agent)
# Or: tool.register_tool(agent)  # registers both on same agent
```

## 5. Verify

After generating, check:
- [ ] Type annotations on all parameters
- [ ] Descriptive `Annotated` strings for each parameter
- [ ] Decorator order is correct (execution outer, llm inner)
- [ ] Return type is specified
- [ ] Description clearly tells the LLM WHEN to use this tool
