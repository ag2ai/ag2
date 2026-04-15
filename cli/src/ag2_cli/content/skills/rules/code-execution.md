---
description: Patterns for AG2 code execution with LocalCommandLineCodeExecutor and Docker
globs: "**/*.py"
alwaysApply: false
---

# AG2 Code Execution

## Overview

AG2 agents can generate and execute code. The recommended pattern separates the LLM agent (which writes code) from the executor agent (which runs it safely).

## Basic Setup

```python
import os
from autogen import ConversableAgent, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

# Agent that writes code
coder = ConversableAgent(
    name="coder",
    system_message="You are a Python developer. Write code to solve the user's task. Put all code in ```python blocks.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Agent that executes code
executor = ConversableAgent(
    name="executor",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(
            work_dir="./coding_output",
            timeout=60,
        ),
    },
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

result = await executor.a_run(
    coder,
    message="Write a Python script to analyze a CSV file and plot the results.",
)
await result.process()
```

## Code Executors

| Executor | Use When |
|----------|----------|
| `LocalCommandLineCodeExecutor` | Development, trusted code |
| `DockerCommandLineCodeExecutor` | Production, untrusted code (sandboxed) |

### LocalCommandLineCodeExecutor

```python
from autogen.coding import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor(
    work_dir="./output",     # Where code files are saved and run
    timeout=60,              # Max execution time in seconds
)
```

### DockerCommandLineCodeExecutor

```python
from autogen.coding import DockerCommandLineCodeExecutor

executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",   # Docker image
    work_dir="./output",
    timeout=120,
)
```

Requires Docker to be running. Strongly recommended for production.

## PythonCodeExecutionTool (Experimental)

A tool-based approach — the agent calls code execution as a tool:

```python
from autogen.tools.experimental import PythonCodeExecutionTool

code_tool = PythonCodeExecutionTool(work_dir="./output")
code_tool.register_for_llm(assistant)
code_tool.register_for_execution(user_proxy)
```

## Safety Best Practices

- **Always use Docker for untrusted code** — `DockerCommandLineCodeExecutor` sandboxes execution
- Set `timeout` to prevent infinite loops
- Set `max_consecutive_auto_reply` to limit code-retry cycles
- Use `work_dir` to isolate file output
- Never run untrusted code with `LocalCommandLineCodeExecutor` in production

## Common Mistakes

- Do NOT use `code_execution_config=True` — use a dict with an executor: `{"executor": LocalCommandLineCodeExecutor(...)}`
- Do NOT give the coder agent code execution AND LLM — separate the roles (coder writes, executor runs)
- Do NOT forget `work_dir` — without it, files are created in the current directory
- Do NOT confuse `LocalCommandlineCodeExecutor` (lowercase L) with `LocalCommandLineCodeExecutor` (capital L) — the correct one has a capital L in "Line"
- Do NOT set `timeout=0` — it means no timeout, which is dangerous
