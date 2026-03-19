---
description: Hook system and custom reply functions in AG2
globs: "**/*.py"
alwaysApply: false
---

# AG2 Hooks and Custom Reply Functions

## Hook System

AG2 agents support three hookable methods for modifying behavior:

### register_hook

```python
agent.register_hook(
    hookable_method="update_agent_state",
    hook=my_state_hook,
)
```

### Available Hook Points

1. **`update_agent_state`** — Called before reply generation to update agent state
   ```python
   def my_state_hook(agent: ConversableAgent, messages: list[dict]) -> None:
       # Modify agent state, update system message, etc.
       agent.update_system_message("Updated context...")
   ```

2. **`process_all_messages_before_reply`** — Filter/transform all messages before reply
   ```python
   def filter_messages(messages: list[dict]) -> list[dict]:
       # Return filtered list of messages
       return [m for m in messages if m.get("role") != "system"]
   ```

3. **`process_last_received_message`** — Transform the last received message
   ```python
   def enrich_message(messages: list[dict]) -> list[dict]:
       # Modify the last message (e.g., add context)
       messages[-1]["content"] += "\n\nAdditional context: ..."
       return messages
   ```

## Custom Reply Functions

Override how agents respond to specific senders:

```python
agent.register_reply(
    trigger=ConversableAgent,  # or agent name string, or callable
    reply_func=my_reply_handler,
    position=0,  # Priority (lower = earlier)
)
```

### Reply Function Signature

```python
def my_reply_handler(
    recipient: ConversableAgent,
    messages: list[dict] | None,
    sender: Agent | None,
    config: Any | None,
) -> tuple[bool, str | dict | None]:
    """
    Returns:
        (True, reply) — Use this reply, stop checking other handlers
        (False, None) — Skip this handler, try the next one
    """
    if should_handle(messages):
        return True, "My custom response"
    return False, None
```

### Trigger Types

- `Agent` class — matches any instance of that class
- `Agent` instance — matches that specific agent
- `str` — matches agent by name
- `Callable[[Agent], bool]` — custom predicate
- `list` — matches any trigger in the list (OR logic)

## Nested Chats

Register sub-conversations triggered by specific agents:

```python
agent.register_nested_chats(
    trigger=analyst_agent,
    chat_queue=[
        {
            "recipient": fact_checker,
            "message": lambda recipient, messages, sender, config: messages[-1]["content"],
            "summary_method": "last_msg",
        },
        {
            "recipient": editor,
            "message": "Polish the verified content:",
            "summary_method": "last_msg",
        },
    ],
)
```

## Common Patterns

### Early Termination

```python
def terminate_on_keyword(
    recipient, messages, sender, config
) -> tuple[bool, str | None]:
    last = messages[-1].get("content", "") if messages else ""
    if "TASK_COMPLETE" in last:
        return True, "Task completed successfully."
    return False, None

agent.register_reply(trigger=ConversableAgent, reply_func=terminate_on_keyword, position=0)
```

### Message Logging Hook

```python
def log_messages(messages: list[dict]) -> list[dict]:
    for msg in messages:
        logger.info(f"[{msg.get('role')}] {msg.get('content', '')[:100]}")
    return messages

agent.register_hook("process_all_messages_before_reply", log_messages)
```
