---
description: Patterns for AG2 sequential chats and pipelines using a_sequential_run
globs: "**/*.py"
alwaysApply: false
---

# AG2 Sequential Chats

## Overview

Sequential chats run a series of two-agent conversations in order, automatically carrying over summaries from previous chats. Use for pipelines where each stage processes the output of the previous one.

## Basic Pipeline

```python
import asyncio
import os
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

user = ConversableAgent(name="user", llm_config=False, human_input_mode="NEVER")
researcher = ConversableAgent(name="researcher", llm_config=llm_config, human_input_mode="NEVER")
analyst = ConversableAgent(name="analyst", llm_config=llm_config, human_input_mode="NEVER")
writer = ConversableAgent(name="writer", llm_config=llm_config, human_input_mode="NEVER")

async def main():
    results = await user.a_sequential_run(
        [
            {
                "recipient": researcher,
                "message": "Research the latest trends in renewable energy.",
                "max_turns": 3,
                "summary_method": "reflection_with_llm",
            },
            {
                "recipient": analyst,
                "message": "Analyze the key findings and identify top 3 trends.",
                "max_turns": 3,
                "summary_method": "reflection_with_llm",
            },
            {
                "recipient": writer,
                "message": "Write a concise executive summary.",
                "max_turns": 2,
                "summary_method": "last_msg",
            },
        ]
    )

    # Each result can be processed (async properties — must await)
    for i, result in enumerate(results):
        await result.process()
        print(await result.summary)

asyncio.run(main())
```

## Chat Queue Dict Keys

Each dict in the chat queue supports:

```python
{
    "recipient": agent,            # The responding agent
    "message": str | callable,     # Initial message (or callable returning str)
    "max_turns": int,              # Maximum turns for this chat
    "summary_method": str,         # "last_msg" or "reflection_with_llm"
    "summary_args": dict,          # Args for summary method
    "clear_history": bool,         # Clear history before this chat (default True)
    "silent": bool,                # Suppress output (default False)
    "carryover": str | list,       # Additional context to carry over
    "finished_chat_indexes_to_exclude_from_carryover": list[int],  # Skip summaries from specific prior chats
}
```

## Carryover

By default, summaries from all previous chats are automatically carried over as context. You can control this:

```python
results = await user.a_sequential_run(
    [
        {"recipient": researcher, "message": "Research X", "summary_method": "reflection_with_llm"},
        {"recipient": analyst, "message": "Analyze the findings"},  # Gets researcher's summary
        {
            "recipient": writer,
            "message": "Write the report",
            "finished_chat_indexes_to_exclude_from_carryover": [0],  # Skip researcher summary, keep analyst only
        },
    ]
)
```

## Summary Methods

| Method | Behavior |
|--------|----------|
| `"last_msg"` | Uses the last message as the summary |
| `"reflection_with_llm"` | LLM generates a summary of the conversation |

`"reflection_with_llm"` gives better summaries but costs an extra LLM call per stage.

## When to Use Sequential vs Group Chat

| Scenario | Use |
|----------|-----|
| Fixed pipeline, each stage has one specialist | `a_sequential_run` |
| Dynamic collaboration, agents decide flow | `run_group_chat` with `AutoPattern` |
| Explicit routing, state-machine style | `run_group_chat` with `DefaultPattern` + handoffs |

## Common Mistakes

- Do NOT forget `summary_method` — the default may not carry useful context to the next stage
- Do NOT use `max_turns=1` if you expect the agent to use tools — tool calls need at least 2 turns (call + response)
- Do NOT assume all chats share history — each is independent, connected only by carryover summaries
- Use `a_sequential_run` (instance method on the sender agent), not the deprecated `initiate_chats` module-level function
