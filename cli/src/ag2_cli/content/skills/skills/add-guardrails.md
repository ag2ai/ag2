---
name: add-guardrails
description: Add safety guardrails to AG2 agents using LLMGuardrail and RegexGuardrail. Use when the user wants to enforce safety constraints, filter PII, or redirect off-topic responses.
---

# Add Guardrails to AG2 Agents

You are an expert at adding safety guardrails to AG2 multi-agent systems. When the user wants to add safety constraints:

## 1. Understand the Requirements

Ask the user:
- What should be blocked? (PII, off-topic responses, harmful content, specific patterns)
- What should happen when a guardrail triggers? (redirect to another agent, terminate, sanitize)
- Is pattern matching sufficient (RegexGuardrail) or does it need LLM judgment (LLMGuardrail)?

## 2. LLM-Based Guardrail

For complex, nuanced safety checks:

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import AgentTarget, TerminateTarget
from autogen.agentchat.group.guardrails import LLMGuardrail

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

# Create a moderator to handle violations
moderator = ConversableAgent(
    name="moderator",
    system_message="A guardrail was triggered. Politely explain that the response was filtered and ask the user to rephrase.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# LLM evaluates whether the response violates the condition
guardrail = LLMGuardrail(
    name="relevance_check",
    condition="The response is off-topic, contains speculation not based on data, or makes claims without evidence",
    target=AgentTarget(moderator),
    llm_config=llm_config,  # Required — the guardrail needs its own LLM
)

# Register on the agent whose output should be checked
assistant.register_output_guardrail(guardrail)
```

## 3. Regex-Based Guardrail

For fast, pattern-based checks (no LLM cost):

```python
from autogen.agentchat.group.guardrails import RegexGuardrail

# Block SSN patterns
pii_guardrail = RegexGuardrail(
    name="ssn_filter",
    condition=r"\b\d{3}-\d{2}-\d{4}\b",
    target=AgentTarget(sanitizer),
)
assistant.register_output_guardrail(pii_guardrail)

# Block email addresses
email_guardrail = RegexGuardrail(
    name="email_filter",
    condition=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    target=AgentTarget(sanitizer),
)
assistant.register_output_guardrail(email_guardrail)

# Block credit card numbers
cc_guardrail = RegexGuardrail(
    name="cc_filter",
    condition=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    target=TerminateTarget(),
)
assistant.register_output_guardrail(cc_guardrail)
```

## 4. Multiple Guardrails

Stack multiple guardrails on an agent — they are checked in registration order:

```python
# Check PII first (fast, no LLM cost), then content quality (LLM-based)
assistant.register_output_guardrail(pii_guardrail)     # Regex — fast
assistant.register_output_guardrail(email_guardrail)   # Regex — fast
assistant.register_output_guardrail(relevance_guard)   # LLM — slower but thorough
```

## 5. Guardrails with Group Chat

```python
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

# Register guardrails before creating the pattern
researcher.register_output_guardrail(pii_guardrail)
writer.register_output_guardrail(relevance_guard)

result = run_group_chat(
    pattern=AutoPattern(
        initial_agent=researcher,
        agents=[researcher, writer, moderator],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    ),
    messages="Research and write a report on...",
    max_rounds=15,
)
```

## 6. Rules

- `LLMGuardrail` requires its own `llm_config` parameter — it won't work without one
- `RegexGuardrail` uses Python regex syntax — test patterns before deploying
- Register regex guardrails before LLM guardrails for efficiency (fail fast on cheap checks)
- The `target` determines where execution goes when the guardrail triggers
- Use `TerminateTarget()` to stop the conversation on critical violations
- Use `AgentTarget(moderator)` to redirect to a handler agent
- Guardrails check the agent's OUTPUT, not input — they fire after the agent responds
