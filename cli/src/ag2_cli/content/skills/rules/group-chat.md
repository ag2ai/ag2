---
description: Patterns for AG2 multi-agent orchestration with run_group_chat, patterns, handoffs, and guardrails
globs: "**/*.py"
alwaysApply: false
---

# AG2 Group Chat

## Modern API: `run_group_chat`

The recommended way to orchestrate multiple agents is `run_group_chat` with a pattern:

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

llm_config = LLMConfig({"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]})

researcher = ConversableAgent(name="researcher", llm_config=llm_config, description="Finds information")
analyst = ConversableAgent(name="analyst", llm_config=llm_config, description="Analyzes data")
writer = ConversableAgent(name="writer", llm_config=llm_config, description="Writes final output")
user = ConversableAgent(name="user", human_input_mode="NEVER", llm_config=False)

result = run_group_chat(
    pattern=AutoPattern(
        initial_agent=researcher,
        agents=[researcher, analyst, writer],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    ),
    messages="Research and write a report on renewable energy.",
    max_rounds=15,
)
result.process()
print(result.summary)
```

Use `await a_run_group_chat(...)` for async. Note: `run_group_chat` returns sync properties (e.g. `result.summary`), while `a_run_group_chat` returns async properties that must be awaited (e.g. `await result.summary`).

## Patterns

Choose a pattern based on how you want speakers selected:

| Pattern | Selection Method | Use When |
|---------|-----------------|----------|
| `AutoPattern` | LLM selects next speaker | Dynamic collaboration, agents should decide flow |
| `RoundRobinPattern` | Fixed rotation | Sequential pipeline (each agent adds to output) |
| `ManualPattern` | Human selects | Interactive, human-guided workflows |
| `RandomPattern` | Random | Testing, brainstorming |
| `DefaultPattern` | Handoffs only | Pure handoff-driven flow, no automatic selection |

### AutoPattern (LLM-Selected)

```python
from autogen.agentchat.group.patterns import AutoPattern

pattern = AutoPattern(
    initial_agent=researcher,
    agents=[researcher, analyst, writer],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)
```

Requires `llm_config` in `group_manager_args` for the LLM to select speakers. Each agent's `description` is critical — it's how the LLM decides who speaks next.

### RoundRobinPattern

```python
from autogen.agentchat.group.patterns import RoundRobinPattern

pattern = RoundRobinPattern(
    initial_agent=researcher,
    agents=[researcher, analyst, writer],
    user_agent=user,
)
```

### DefaultPattern (Handoff-Driven)

For customer service routing, state-machine workflows, or explicit agent-to-agent transitions:

```python
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import OnCondition, AgentTarget, TerminateTarget, StringLLMCondition

triage.handoffs.add_llm_conditions([
    OnCondition(target=AgentTarget(billing), condition=StringLLMCondition(prompt="Customer has a billing question")),
    OnCondition(target=AgentTarget(support), condition=StringLLMCondition(prompt="Customer has a technical issue")),
])
billing.handoffs.set_after_work(TerminateTarget())
support.handoffs.set_after_work(TerminateTarget())

pattern = DefaultPattern(
    initial_agent=triage,
    agents=[triage, billing, support],
    user_agent=user,
)
```

No `group_manager_args` needed — routing is entirely driven by agent handoffs.

### Common Pattern Parameters

All patterns accept:

```python
Pattern(
    initial_agent=agent,             # First agent to speak
    agents=[agent1, agent2, ...],    # All participating agents
    user_agent=user,                 # Optional user agent
    context_variables=ctx,           # Shared state (ContextVariables)
    group_after_work=TerminateTarget(),  # Default action when no handoff matches
    summary_method="last_msg",       # "last_msg" or "reflection_with_llm"
)
```

## Handoffs

Agents can hand off to other agents based on conditions. Handoffs are evaluated in order:

1. **OnContextCondition** — fast, checks context variables, no LLM cost
2. **OnCondition** — LLM evaluates a natural-language condition
3. **after_work** — fallback if no condition matched

```python
from autogen.agentchat.group import (
    OnCondition, OnContextCondition, ContextVariables,
    AgentTarget, TerminateTarget, RevertToUserTarget,
    StringLLMCondition,
)

# LLM-based handoff
researcher.handoffs.add_llm_condition(
    OnCondition(
        target=AgentTarget(analyst),
        condition=StringLLMCondition(prompt="Research is complete and data needs analysis"),
    )
)

# Context-variable-based handoff (no LLM cost)
analyst.handoffs.add_context_condition(
    OnContextCondition(
        target=AgentTarget(writer),
        condition=StringContextCondition(variable_name="analysis_done"),
    )
)

# Default fallback
writer.handoffs.set_after_work(TerminateTarget())
```

## Context Variables

Shared mutable state accessible by all agents in the group chat:

```python
from autogen.agentchat.group import ContextVariables

ctx = ContextVariables(data={"stage": "research", "findings": []})

pattern = AutoPattern(
    initial_agent=researcher,
    agents=[researcher, analyst],
    context_variables=ctx,
    ...
)
```

Access in tools via dependency injection:

```python
from autogen.tools import ChatContext, Depends

def save_finding(
    finding: Annotated[str, "A research finding"],
    context: ChatContext = Depends(ChatContext),
) -> str:
    # Context variables are available through the chat context
    return f"Saved: {finding}"
```

## Transition Targets

Control where handoffs go:

| Target | Behavior |
|--------|----------|
| `AgentTarget(agent)` | Hand off to a specific agent |
| `AgentNameTarget("name")` | Hand off by agent name (more serializable) |
| `TerminateTarget()` | End the conversation |
| `StayTarget()` | Current agent speaks again |
| `RevertToUserTarget()` | Return to the user agent |
| `AskUserTarget()` | Prompt the user for input |
| `NestedChatTarget(config)` | Start a sub-conversation |
| `GroupChatTarget(config)` | Start a nested group chat |
| `FunctionTarget(fn)` | Call a Python function to decide next step |

## Guardrails

Add safety constraints to agents:

```python
from autogen.agentchat.group.guardrails import LLMGuardrail, RegexGuardrail

# LLM-based guardrail
guardrail = LLMGuardrail(
    name="relevance_check",
    condition="The response is off-topic or not relevant to the task",
    target=AgentTarget(moderator),
    llm_config=llm_config,
)
agent.register_output_guardrail(guardrail)

# Regex-based guardrail (faster, no LLM cost)
pii_guardrail = RegexGuardrail(
    name="pii_filter",
    condition=r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
    target=AgentTarget(sanitizer),
)
agent.register_output_guardrail(pii_guardrail)
```

## Agent Descriptions Matter

In group chat, `description` is critical for speaker selection (especially with `AutoPattern`):

```python
researcher = ConversableAgent(
    name="researcher",
    description="Searches for information and gathers data. Call this agent when facts or data are needed.",
    llm_config=llm_config,
)
```

## Common Mistakes

- Do NOT use legacy `GroupChat + GroupChatManager` — use `run_group_chat` with patterns instead
- Do NOT use duplicate agent names in the same group chat
- Do NOT forget `description` for agents — the speaker selector relies on it
- Do NOT forget to set `max_rounds` to prevent infinite loops
- Do NOT use `run_swarm` — use `run_group_chat` with `DefaultPattern` + handoffs instead
- When using `AutoPattern`, always provide `llm_config` in `group_manager_args`
- `LLMGuardrail` requires a `llm_config` parameter — it won't work without one
