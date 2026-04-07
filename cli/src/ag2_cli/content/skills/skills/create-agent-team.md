---
name: create-agent-team
description: Scaffold a complete AG2 multi-agent team with agents, tools, group chat configuration, and an entry point. Use when the user wants to create a new multi-agent workflow from scratch.
---

# Create AG2 Agent Team

You are an expert AG2 framework developer. The user wants to scaffold a multi-agent team. Follow these steps:

## 1. Understand the Task

Ask the user (if not already clear):
- What is the overall goal of the agent team?
- What distinct roles are needed? (e.g., researcher, coder, reviewer)
- Should agents communicate dynamically (AutoPattern) or in sequence (RoundRobinPattern)?
- What tools/capabilities do the agents need?

## 2. Generate the Code

Create a Python file with this structure:

```python
import os
from typing import Annotated
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

# 1. LLM Configuration
llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
)

# 2. Define Agents
# - Give each agent a clear name (no whitespace), system_message, and description
# - description is used by AutoPattern's LLM to select the next speaker
# - Set human_input_mode appropriately

user = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
)

# ... define role-specific agents ...

# 3. Register Tools (if needed)
# - Use @agent.register_for_llm() and @agent.register_for_execution() decorators
# - Or use Tool/Toolkit classes
# - For pre-built tools: from autogen.tools.experimental import DuckDuckGoSearchTool

# 4. Orchestration — choose pattern based on workflow:
#    - AutoPattern: LLM picks next speaker (best for dynamic collaboration)
#    - RoundRobinPattern: sequential pipeline (each agent adds to output)
#    - DefaultPattern: pure handoff-driven (you define all transitions)

result = run_group_chat(
    pattern=AutoPattern(
        initial_agent=first_agent,
        agents=[agent1, agent2, agent3],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    ),
    messages="Describe the task here.",
    max_rounds=15,
)

# 5. Results
result.process()
print(result.summary)
```

## 3. For Handoff-Driven Workflows

When agents should explicitly route to each other:

```python
from autogen.agentchat.group import (
    OnCondition, AgentTarget, TerminateTarget, StringLLMCondition,
)
from autogen.agentchat.group.patterns import DefaultPattern

# Define handoffs
researcher.handoffs.add_llm_condition(
    OnCondition(
        target=AgentTarget(analyst),
        condition=StringLLMCondition(prompt="Research is complete and ready for analysis"),
    )
)
analyst.handoffs.set_after_work(TerminateTarget())

result = run_group_chat(
    pattern=DefaultPattern(
        initial_agent=researcher,
        agents=[researcher, analyst],
        user_agent=user,
    ),
    messages="Start researching...",
    max_rounds=20,
)
```

## 4. Rules to Follow

- Import from `autogen`, not `ag2`
- Always use `LLMConfig()` class, never raw dicts for llm_config
- Every agent must have a unique name and a meaningful `description`
- Pair every `register_for_llm()` with a `register_for_execution()`
- Set `human_input_mode="NEVER"` for fully autonomous agents
- Use type annotations with `Annotated[type, "description"]` for all tool parameters
- Do NOT add tools to the group manager's llm_config
- Always set `max_rounds` to prevent infinite loops
- When using `AutoPattern`, always provide `llm_config` in `group_manager_args`
