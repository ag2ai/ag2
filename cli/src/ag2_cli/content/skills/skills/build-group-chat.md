---
name: build-group-chat
description: Build an AG2 handoff-driven workflow with DefaultPattern, agent handoffs, context variables, and routing. Use when the user wants customer service routing, state-machine workflows, or explicit agent-to-agent transitions.
---

# Build AG2 Handoff-Driven Workflow

You are an expert at building AG2 handoff-driven workflows using `run_group_chat` with `DefaultPattern`. When the user wants to build a routing/handoff workflow:

## 1. Understand the Workflow

Ask the user:
- What agents are needed and what are their roles?
- What conditions determine routing between agents?
- Is there shared state that agents need to read/update?
- What happens when an agent finishes (terminate, escalate, loop)?

## 2. Generate the Code

```python
import os
from typing import Annotated
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group import (
    OnCondition, AgentTarget, TerminateTarget, ContextVariables, StringLLMCondition,
)
from autogen.agentchat.group.patterns import DefaultPattern

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

# Shared state across all agents
ctx = ContextVariables(data={"status": "new", "category": ""})

# 1. Define agents
triage = ConversableAgent(
    name="triage",
    system_message="You are a customer service triage agent. Determine the customer's issue category and route appropriately.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

billing = ConversableAgent(
    name="billing",
    system_message="You handle billing inquiries: refunds, charges, subscription changes.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

support = ConversableAgent(
    name="support",
    system_message="You handle technical support issues.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user = ConversableAgent(name="user", llm_config=False, human_input_mode="NEVER")

# 2. Define handoffs — OnCondition is evaluated by the LLM
triage.handoffs.add_llm_conditions([
    OnCondition(
        target=AgentTarget(billing),
        condition=StringLLMCondition(prompt="Customer has a billing or payment question"),
    ),
    OnCondition(
        target=AgentTarget(support),
        condition=StringLLMCondition(prompt="Customer has a technical issue or bug report"),
    ),
])

# 3. Define after-work behavior
billing.handoffs.set_after_work(TerminateTarget())
support.handoffs.set_after_work(TerminateTarget())

# 4. Register tools (optional)
@billing.register_for_execution()
@billing.register_for_llm(description="Look up a customer's recent charges")
def lookup_charges(
    customer_id: Annotated[str, "The customer ID to look up"],
) -> str:
    # Replace with actual DB lookup
    return "Last charge: $29.99 on 2025-03-01 (Monthly subscription)"

# 5. Run the group chat with DefaultPattern
result = run_group_chat(
    pattern=DefaultPattern(
        initial_agent=triage,
        agents=[triage, billing, support],
        user_agent=user,
        context_variables=ctx,
    ),
    messages="I was charged twice this month and need a refund",
    max_rounds=15,
)
result.process()
print(result.summary)
```

## 3. Advanced: Multi-Level Routing

For complex workflows with escalation:

```python
escalation = ConversableAgent(
    name="escalation",
    system_message="You handle escalated issues that other agents could not resolve.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Billing can escalate
billing.handoffs.add_llm_condition(
    OnCondition(
        target=AgentTarget(escalation),
        condition=StringLLMCondition(prompt="Issue requires manager approval or is beyond standard policy"),
    )
)

# Support can escalate
support.handoffs.add_llm_condition(
    OnCondition(
        target=AgentTarget(escalation),
        condition=StringLLMCondition(prompt="Issue is a critical bug or requires engineering team"),
    )
)

# After-work with RevertToUserTarget
from autogen.agentchat.group import RevertToUserTarget
escalation.handoffs.set_after_work(RevertToUserTarget())
```

## 4. Context-Aware Routing

Use `OnContextCondition` for deterministic routing based on shared state (faster, no LLM cost):

```python
from autogen.agentchat.group import (
    OnContextCondition, ExpressionContextCondition, ContextExpression, ReplyResult,
)

# Tool that updates context and hands off
@triage.register_for_execution()
@triage.register_for_llm(description="Classify the customer issue")
def classify_issue(
    issue_type: Annotated[str, "billing or support"],
    context_variables: ContextVariables,
) -> ReplyResult:
    context_variables["issue_type"] = issue_type
    target = AgentTarget(billing) if issue_type == "billing" else AgentTarget(support)
    return ReplyResult(
        message=f"Classified as {issue_type}",
        context_variables=context_variables,
        target=target,
    )
```

## 5. Rules

- Use `run_group_chat` with `DefaultPattern`, NOT deprecated `run_swarm`
- Use `ConversableAgent` for all agents
- Import handoff classes from `autogen.agentchat.group`
- Always set `max_rounds` to prevent infinite loops
- Use `TerminateTarget()` as the default after-work to avoid hanging conversations
- Write clear, specific condition strings — the LLM evaluates them to decide routing
- Keep the number of conditions per agent small (2-4) for reliable routing
- Prefer `OnContextCondition` over `OnCondition` for simple state checks (cheaper, deterministic)
