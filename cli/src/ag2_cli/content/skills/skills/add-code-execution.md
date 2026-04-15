---
name: add-code-execution
description: Add code execution capability to AG2 agents using LocalCommandLineCodeExecutor or Docker. Use when the user wants agents that can write and run Python code.
---

# Add Code Execution to AG2 Agents

You are an expert at setting up code execution in AG2 agent workflows. When the user wants agents that can write and run code:

## 1. Understand the Requirements

Ask the user:
- What language should the code be in? (Python is best supported)
- Should execution be sandboxed? (Docker for production, local for development)
- What packages/libraries does the code need access to?
- Should there be a human approval step before execution?

## 2. Two-Agent Code Execution (Recommended)

```python
import os
from autogen import ConversableAgent, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

# Coder writes code
coder = ConversableAgent(
    name="coder",
    system_message="""You are an expert Python developer.
Write code to solve the user's task. Always put code in ```python blocks.
When the task is complete, reply with TERMINATE.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Executor runs code and returns output
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
    max_consecutive_auto_reply=10,
)

result = await executor.a_run(
    coder,
    message="Create a script that fetches the top 10 Python packages from PyPI and saves them to a CSV.",
)
await result.process()
```

## 3. Docker Execution (Production)

```python
from autogen.coding import DockerCommandLineCodeExecutor

executor = ConversableAgent(
    name="executor",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "executor": DockerCommandLineCodeExecutor(
            image="python:3.11-slim",
            work_dir="./coding_output",
            timeout=120,
        ),
    },
)
```

## 4. Human-Approved Execution

For workflows where a human should review code before running:

```python
executor = ConversableAgent(
    name="executor",
    llm_config=False,
    human_input_mode="ALWAYS",  # Asks for approval before each execution
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="./output"),
    },
)
```

## 5. Code Execution in Group Chat

```python
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern

planner = ConversableAgent(
    name="planner",
    system_message="You break tasks into coding steps.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Plans the coding approach.",
)

coder = ConversableAgent(
    name="coder",
    system_message="You write Python code based on the plan. Put code in ```python blocks.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Writes Python code.",
)

executor = ConversableAgent(
    name="executor",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="./output", timeout=60),
    },
    description="Runs Python code and returns results.",
)

user = ConversableAgent(name="user", llm_config=False, human_input_mode="NEVER")

result = run_group_chat(
    pattern=RoundRobinPattern(
        initial_agent=planner,
        agents=[planner, coder, executor],
        user_agent=user,
    ),
    messages="Analyze the iris dataset and create a classification model.",
    max_rounds=12,
)
```

## 6. Rules

- Separate coder (LLM) from executor (no LLM) — don't give one agent both roles
- Use `DockerCommandLineCodeExecutor` for untrusted code or production
- Always set `timeout` to prevent infinite execution
- Always set `max_consecutive_auto_reply` to limit retry loops
- Set `work_dir` to isolate output files
- Use `human_input_mode="ALWAYS"` during development for safety
- The executor extracts code from ```python blocks automatically
- Capital L in `LocalCommandLineCodeExecutor` (not `Commandline`)
