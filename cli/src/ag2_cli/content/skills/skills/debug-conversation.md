---
name: debug-conversation
description: Analyze AG2 agent conversation logs and code to diagnose issues like wrong speaker selection, tool call failures, infinite loops, or unexpected termination.
---

# Debug AG2 Conversation

You are an expert at debugging AG2 multi-agent conversations. When the user shares conversation output or code that isn't working:

## 1. Common Issues Checklist

### Tool Calls Not Happening
- Is the tool registered for LLM on the correct agent? (`register_for_llm`)
- Is the tool registered for execution on the correct agent? (`register_for_execution`)
- Is the tool description clear enough for the LLM to know WHEN to call it?
- Are parameter types properly annotated?
- Check: does the agent's `llm_config` include the tools? (auto-added by registration)

### Wrong Speaker Selected (Group Chat)
- Does each agent have a meaningful `description`?
- If using `AutoPattern`, is `llm_config` set in `group_manager_args`?
- If using legacy `GroupChat` (deprecated), consider migrating to `run_group_chat` with patterns
- Are handoffs configured correctly (`OnCondition` / `OnContextCondition`)?
- Is `send_introductions=True` set (legacy) or are descriptions clear enough (modern)?

### Infinite Loops
- Is `max_consecutive_auto_reply` set?
- Is `max_rounds` set on `run_group_chat`?
- Is `is_termination_msg` defined?
- Are agents bouncing messages without making progress? Check system prompts.

### Unexpected Termination
- Check `human_input_mode` — `"TERMINATE"` prompts at termination, `"NEVER"` auto-terminates
- Check `is_termination_msg` — is it matching too aggressively?
- Check `max_consecutive_auto_reply` — is it too low?
- Is an agent returning empty/None responses?

### LLM Errors
- Is `llm_config` set to `False` when it should have a config?
- Are API keys set correctly in environment?
- Is the model name correct?
- Check rate limits — add `timeout` to LLMConfig

### Code Execution Failures
- Is `code_execution_config` set on the executing agent?
- Is Docker required but not running?
- Check `work_dir` permissions
- Is `use_docker` set correctly for the environment?

## 2. Debugging Techniques

### Enable Verbose Logging
```python
import autogen
import logging

# Enable AG2 runtime logging (writes to SQLite or file)
autogen.runtime_logging.start(logger_type="sqlite", config={"dbname": "ag2_logs.db"})

# Or use standard Python logging for console output
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Message History
```python
# After chat completes
for msg in chat_result.chat_history:
    print(f"[{msg['role']}] {msg.get('name', 'unknown')}: {msg['content'][:200]}")
```

### Check Tool Registration
```python
# Verify tools are registered
print("LLM tools:", list(agent.llm_config.tools))
print("Function map:", list(agent.function_map.keys()))
```

### Check Reply Functions
```python
# List registered reply functions
for trigger, func, config in agent._reply_func_list:
    print(f"Trigger: {trigger}, Func: {func.__name__}")
```

## 3. Provide Fix

After identifying the issue:
1. Explain what went wrong and why
2. Show the specific code change needed
3. Suggest preventive patterns (termination conditions, max rounds, etc.)
