---
description: Coding patterns for creating and configuring AG2 agents
globs: "**/*.py"
alwaysApply: false
---

# AG2 Agent Patterns

## Agent Classes

AG2 provides two main agent classes, imported from `autogen`:

```python
from autogen import ConversableAgent, UserProxyAgent
```

- **ConversableAgent** — The primary agent class. Use for all LLM-powered agents with full control over configuration.
- **UserProxyAgent** — Pre-configured as human proxy (human-in-the-loop). Defaults: `human_input_mode="ALWAYS"`, `code_execution_config={}`, `llm_config=False`.

## Creating Agents

Always provide a `name` (no whitespace allowed) and explicit `llm_config`:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]})

agent = ConversableAgent(
    name="researcher",
    system_message="You are a research assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

## Key Parameters

- `human_input_mode`: `"ALWAYS"` | `"TERMINATE"` | `"NEVER"` — controls when the agent asks for human input
- `code_execution_config`: `False` to disable, or dict like `{"use_docker": True}` to enable
- `llm_config`: `LLMConfig(...)` to enable LLM, `False` to disable
- `is_termination_msg`: Callable that returns True when conversation should end
- `max_consecutive_auto_reply`: Limit auto-replies before stopping
- `description`: Used by GroupChat for speaker selection — always set this for group chat agents
- `functions`: Pass callables directly to auto-register as tools

## Common Mistakes

- Do NOT use whitespace in agent names — it raises ValueError when llm_config is set
- Do NOT pass `llm_config=True` — use `LLMConfig(...)` or `False`
- Do NOT forget `description` for agents used in GroupChat — the speaker selector uses it
- Do NOT set `code_execution_config=True` — use a dict like `{}` or `{"use_docker": True}`
## Starting a Chat

```python
# Two-agent chat (async)
chat_result = await user_proxy.a_run(
    assistant,
    message="Help me analyze this data.",
)

# Access results (async properties — must await)
await chat_result.process()
print(await chat_result.summary)
```

## Dynamic System Messages

Use `UpdateSystemMessage` to change the system message based on context:

```python
from autogen import ConversableAgent, UpdateSystemMessage

agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    update_agent_state_before_reply=[
        UpdateSystemMessage("Current context: {context_var}")
    ],
)
```

`UpdateSystemMessage` accepts a format string with context variable keys, or a `Callable[[ConversableAgent, list[dict]], str]`.
