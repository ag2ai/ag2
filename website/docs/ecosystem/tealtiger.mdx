---
title: TealTiger
---

# TealTiger Governance Integration

![TealTiger](https://raw.githubusercontent.com/agentguard-ai/tealtiger/main/.github/logo/tealtiger-logo-256.png)

[TealTiger](https://github.com/agentguard-ai/tealtiger) provides deterministic governance for AG2 agents via the `register_reply` interceptor mechanism. No LLM in the governance path — all policy evaluation is deterministic with <5ms overhead.

## Installation

```bash
pip install "ag2-tealtiger>=0.1.0"
```

**Compatibility:** AG2 `>=0.6.0,<1.0`

## Quick Start

### Zero-Config Observe Mode

Track cost, PII, and tool usage across all agents with zero configuration:

```python
from ag2_tealtiger import TealTigerGuard

# Observe everything, block nothing
guard = TealTigerGuard()
guard.attach(my_agent)

# After some tool calls...
for entry in guard.audit_trail:
    print(f"{entry.agent_id}: {entry.tool_name} → {entry.action} (cost: ${entry.cost_tracked:.4f})")
```

### Policy Enforcement

```python
from ag2_tealtiger import TealTigerGuard, GovernanceMode
from tealtiger import TealEngine

engine = TealEngine(policies=[
    {"type": "cost_limit", "max_per_session": 5.00},
    {"type": "pii_block", "categories": ["ssn", "credit_card"]},
    {"type": "tool_allowlist", "allowed": ["web_search", "read_file"]},
])

guard = TealTigerGuard(engine=engine, mode=GovernanceMode.ENFORCE)
guard.attach(agent)

# Tool calls are now evaluated against policies
# DENY blocks with structured denial message visible in conversation
# ALLOW passes through transparently
```

### ConversableAgent Subclass

```python
from ag2_tealtiger import TealTigerAuditAgent, GovernanceMode

# Zero-config — governance built in
agent = TealTigerAuditAgent(name="coder")

# With policy engine and budget
agent = TealTigerAuditAgent(
    name="executor",
    engine=my_engine,
    mode=GovernanceMode.ENFORCE,
    budget_limit=5.0,
)

# Access governance state
print(agent.audit_trail)
print(agent.summary)
```

### Governed GroupChat

```python
from ag2_tealtiger import GovernedGroupChat, TealTigerGuard, GovernanceMode

guard = TealTigerGuard(engine=engine, mode=GovernanceMode.ENFORCE)

for agent in [coder, reviewer, executor]:
    guard.attach(agent)

group_chat = GovernedGroupChat(agents=[coder, reviewer, executor], guard=guard)
speaker = group_chat.select_speaker(last_speaker=coder)

# Frozen agents skipped (no engine call)
# Over-budget agents skipped (no engine call)
# Policy-denied agents skipped to next candidate
# ALL_SPEAKERS_DENIED terminates the round safely
```

## Per-Agent Kill Switch

```python
# Freeze mid-conversation — blocks ALL actions regardless of mode
guard.freeze("dangerous_agent")

# Other agents continue normally
speaker = group_chat.select_speaker(last_speaker=coder)

# Unfreeze to restore normal governance
guard.unfreeze("dangerous_agent")
```

## Budget Enforcement

```python
guard.set_budget("expensive_agent", limit=5.0)

# After cumulative cost exceeds limit:
# - ENFORCE: blocks with BUDGET_EXCEEDED
# - 80% warning emitted at threshold
# - Reset: guard.reset_budget("expensive_agent")
```

## REFER Escalation

For multi-agent scenarios where binary ALLOW/DENY isn't enough:

```python
# When TealEngine returns REFER, action is suspended (not terminated)
# GroupChat continues with remaining agents
# The escalation receipt contains full context for a human reviewer

# Resolve later:
guard.resolve_refer(decision_id, resolution="ALLOW", approval_id="reviewer-123")
```

## Inter-Agent Message Governance

```python
# Messages between agents are also governed
# Frozen sender → blocked regardless of mode
# Policy-denied message → blocked in ENFORCE, logged in MONITOR
# Each message gets its own decision_id (independent from tool governance)
```

## Audit Trail Export

```python
# Export full governance audit as JSONL
entries_written = guard.export_audit_trail("audit.jsonl")

# Each line contains: agent_id, tool_name, action, risk_score,
# reason_codes, evaluation_time_ms, TEEC correlation context
```

## Features

| Feature | Observe | Monitor | Enforce |
|---------|:-------:|:-------:|:-------:|
| Cost tracking per agent | ✅ | ✅ | ✅ |
| PII detection in tool args | ✅ | ✅ | ✅ |
| Structured audit entries (TEEC) | ✅ | ✅ | ✅ |
| Correlation IDs (UUID v4) | ✅ | ✅ | ✅ |
| Per-agent freeze/unfreeze | ✅ | ✅ | ✅ |
| Policy evaluation | — | ✅ (log) | ✅ (block) |
| Budget enforcement | — | ✅ (log) | ✅ (block) |
| REFER escalation | — | ✅ | ✅ |
| Inter-agent message governance | ✅ | ✅ | ✅ |
| Governed speaker selection | ✅ | ✅ | ✅ |
| Decision receipt expiry | — | ✅ | ✅ |
| JSONL audit export | ✅ | ✅ | ✅ |

## Governance Modes

- **OBSERVE** — Zero-config default. Track everything, block nothing. Build behavioral baseline.
- **MONITOR** — Evaluate policies, log decisions, allow all through. See what would be blocked.
- **ENFORCE** — Evaluate policies, block denied actions. Production mode.

## Error Handling

| Scenario | ENFORCE | MONITOR/OBSERVE |
|----------|---------|-----------------|
| Engine returns DENY | Block + denial message | Log + allow |
| Engine throws exception | Fail-closed (deny) | Fail-open (allow) |
| Agent frozen | Block (always) | Block (always) |
| Budget exceeded | Block | Log + allow |

## Links

- **PyPI:** [ag2-tealtiger](https://pypi.org/project/ag2-tealtiger/)
- **Source:** [github.com/agentguard-ai/tealtiger/packages/ag2-tealtiger](https://github.com/agentguard-ai/tealtiger/tree/main/packages/ag2-tealtiger)
- **Examples:** [Zero-config](https://github.com/agentguard-ai/tealtiger/blob/main/packages/ag2-tealtiger/examples/zero_config_observe.py) | [Policy enforcement](https://github.com/agentguard-ai/tealtiger/blob/main/packages/ag2-tealtiger/examples/policy_enforcement.py) | [Governed GroupChat](https://github.com/agentguard-ai/tealtiger/blob/main/packages/ag2-tealtiger/examples/governed_groupchat.py)
- **License:** Apache-2.0
- **Feature Request:** [#2942](https://github.com/ag2ai/ag2/issues/2942)
