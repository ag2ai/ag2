---
title: Context-Aware Routing
sidebarTitle: Context-Aware Routing
---

The Context-Aware Routing pattern uses a router agent to read the
user's request, classify it into a category, and dispatch to the
specialist whose domain matches.

**Classic primitives:** `#!python DefaultPattern`, `#!python StringLLMCondition`
(LLM-evaluated routing inside the framework), or
`#!python ExpressionContextCondition` over a router-tool-set domain field.

### Key Characteristics

* **Router agent thin.** The router's only job is to classify and
  call the matching `#!python classify_as_<category>` tool.
* **Dynamic `#!python Handoff`.** Each classify tool returns
  `#!python Handoff(target=<specialist>)` directly. The framework
  reads the `#!python target` from the tool result and routes the
  next turn to that specialist without any graph condition — no
  `#!python ContextEquals` rules needed.
* **Specialist's reply terminates.** A `#!python FromSpeaker(<specialist>) → TerminateTarget`
  rule closes the workflow after the specialist speaks.

### Routing Mechanics

Each `#!python classify_as_<category>` tool returns
`#!python Handoff(target=<specialist_name>)`. The workflow adapter
reads the `#!python Handoff.target` from the tool's
`#!python ToolResultEvent` and stamps it onto the outgoing packet as
`#!python routing.target`. When `#!python fold` processes that
packet, `#!python expected_next_speaker` is set directly from
`#!python routing.target`, bypassing the transition graph entirely.
No `#!python ContextEquals` state variable is needed.

!!! note "Why not ContextEquals?"
    The original idiom — tools call `#!python set_context` then return
    a string; the graph uses `#!python ContextEquals` to pick the
    next speaker — stalls in practice.

    `#!python set_context` emits a non-substantive `#!python EV_CONTEXT_SET`
    envelope, and `#!python ContextEquals` only evaluates when a
    *substantive* `#!python EV_PACKET` follows. But the router's classify
    tool also returns a plain string. If the router produces no text body
    (as a minimal implementation would), the round is silent:
    `#!python build_round_envelope` returns `#!python None`, no
    `#!python EV_PACKET` is posted, `#!python fold` is never called,
    and `#!python ContextEquals` never fires. The channel stalls.

    `#!python Handoff` sidesteps this entirely — the routing target is
    resolved from the tool result, not from a graph condition evaluated
    on a subsequent envelope.

## Agent Flow

```mermaid
sequenceDiagram
    participant User as user
    participant Router as router
    participant Billing as billing
    participant Technical as technical
    participant General as general

    User->>Router: question
    Router->>Router: classify_as_<category>(reason) → Handoff(target=<category>)
    Note over Router: Handoff.target routes directly to specialist
    alt Handoff(target="billing")
        Router->>Billing: AgentTarget(billing) via Handoff
        Billing->>User: answer; FromSpeaker(billing) → TerminateTarget("billing_resolved")
    else Handoff(target="technical")
        Router->>Technical: AgentTarget(technical) via Handoff
        Technical->>User: answer; FromSpeaker(technical) → TerminateTarget("technical_resolved")
    else Handoff(target="general")
        Router->>General: AgentTarget(general) via Handoff
        General->>User: answer; FromSpeaker(general) → TerminateTarget("general_resolved")
    end
```

## Migrating from Classic to Beta?

| Classic | Beta |
|---|---|
| `#!python StringLLMCondition` (framework asks LLM at the transition) | Router agent's LLM turn calls `#!python classify_as_<category>` tool; tool returns `#!python Handoff(target=specialist)` |
| `#!python ReplyResult(context_variables={"category": "billing"}, target=AgentTarget(billing))` | `#!python return Handoff(target="billing", reason=reason)` directly from the classify tool |
| `#!python ExpressionContextCondition(...)` per category | No graph condition needed — `#!python Handoff.target` is authoritative |

## Code

!!! tip
    The router uses real Sonnet (the classification is the
    LLM-driven part). Each specialist also uses real Sonnet to give
    a domain-flavoured reply.

```python linenums="1"
"""Cookbook 07 — Context-Aware Routing pattern.

A router agent reads the user's request, classifies it into a
category, and returns Handoff(target=specialist) from the classify
tool. The framework routes directly to the named specialist without
any ContextEquals graph conditions.
"""

import asyncio

from dotenv import load_dotenv

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_CHANNEL_CLOSED,
    EV_PACKET,
    EV_TEXT,
    WORKFLOW_TYPE,
    AgentTarget,
    FromSpeaker,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
    TerminateTarget,
    Transition,
    TransitionGraph,
)
from autogen.beta.testing import TestConfig

load_dotenv()

async def classify_as_billing(reason: str) -> Handoff:
    """Classify the request as billing and route to the billing specialist."""
    print(f"  [tool] classify_as_billing({reason!r})")
    return Handoff(target="billing", reason=reason)

async def classify_as_technical(reason: str) -> Handoff:
    """Classify as technical and route to the technical specialist."""
    print(f"  [tool] classify_as_technical({reason!r})")
    return Handoff(target="technical", reason=reason)

async def classify_as_general(reason: str) -> Handoff:
    """Classify as general support and route to the general specialist."""
    print(f"  [tool] classify_as_general({reason!r})")
    return Handoff(target="general", reason=reason)

async def main() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6")

    hub_obj = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    link = LocalLink(hub_obj)

    user_hc = HubClient(link, hub=hub_obj)
    router_hc = HubClient(link, hub=hub_obj)
    billing_hc = HubClient(link, hub=hub_obj)
    technical_hc = HubClient(link, hub=hub_obj)
    general_hc = HubClient(link, hub=hub_obj)

    user_agent = Agent("user", config=TestConfig())

    router_agent = Agent(
        "router",
        prompt=(
            "You are the routing agent. Classify the user's request "
            "into ONE of three categories and call the matching tool.\n"
            "\n"
            "Categories:\n"
            "* `classify_as_billing` — payment, refund, invoice, "
            "subscription tier, pricing.\n"
            "* `classify_as_technical` — bug, error, integration, "
            "API, setup, connectivity. Anything technical.\n"
            "* `classify_as_general` — account info, policy, FAQ, "
            "anything not billing or technical.\n"
            "\n"
            "Call exactly ONE tool with a short `reason` argument."
        ),
        config=config,
    )
    router_agent.tool(classify_as_billing)
    router_agent.tool(classify_as_technical)
    router_agent.tool(classify_as_general)

    billing_agent = Agent(
        "billing",
        prompt=(
            "You are the billing specialist. Reply in 1-2 sentences "
            "with concrete next steps for billing/payment/subscription "
            "issues. Don't escalate — just answer."
        ),
        config=config,
    )
    technical_agent = Agent(
        "technical",
        prompt=(
            "You are the technical specialist. Reply in 1-2 sentences "
            "with concrete diagnostic next steps for bugs, API errors, "
            "or integration problems. Don't escalate — just answer."
        ),
        config=config,
    )
    general_agent = Agent(
        "general",
        prompt=(
            "You are the general support specialist. Reply in 1-2 "
            "sentences answering account, policy, or FAQ questions. "
            "Don't escalate — just answer."
        ),
        config=config,
    )

    user = await user_hc.register(user_agent, Passport(name="user"), Resume())
    router = await router_hc.register(router_agent, Passport(name="router"), Resume())
    billing = await billing_hc.register(billing_agent, Passport(name="billing"), Resume())
    technical = await technical_hc.register(technical_agent, Passport(name="technical"), Resume())
    general = await general_hc.register(general_agent, Passport(name="general"), Resume())

    graph = TransitionGraph(
        initial_speaker=user.agent_id,
        transitions=[
            # Specialist's reply terminates.
            Transition(when=FromSpeaker(billing.agent_id),   then=TerminateTarget("billing_resolved")),
            Transition(when=FromSpeaker(technical.agent_id), then=TerminateTarget("technical_resolved")),
            Transition(when=FromSpeaker(general.agent_id),   then=TerminateTarget("general_resolved")),
            # User's question → router. Routing to the specialist is
            # via Handoff returns from classify tools — no ContextEquals
            # rules needed.
            Transition(when=FromSpeaker(user.agent_id), then=AgentTarget(router.agent_id)),
        ],
        default_target=TerminateTarget("fall_through"),
        max_turns=10,
    )

    channel = await user.open(
        type=WORKFLOW_TYPE,
        target=[router.agent_id, billing.agent_id, technical.agent_id, general.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    print(f"channel: {channel.channel_id}\n")

    name_by_id = {
        user.agent_id: "user",
        router.agent_id: "router",
        billing.agent_id: "billing",
        technical.agent_id: "technical",
        general.agent_id: "general",
    }

    await channel.send(
        "I tried to upgrade my subscription but the API is returning a "
        "500 error. The status page says everything is green. Help?"
    )

    close_env = await user.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
        timeout=120.0,
    )

    for env in await hub_obj.read_wal(channel.channel_id):
        speaker = name_by_id.get(env.sender_id, env.sender_id[:8])
        if env.event_type == EV_TEXT:
            print(f"{speaker:>14}: {env.event_data['text']}")
        elif env.event_type == EV_PACKET:
            routing = env.event_data.get("routing", {}) or {}
            if routing.get("kind") == "handoff":
                line = f"[Handed off via {routing.get('tool', '')}] {routing.get('reason', '')}"
                print(f"{speaker:>14}: {line.rstrip()}")
            body = env.event_data.get("body", "")
            if body:
                print(f"{speaker:>14}: {body}")

    print(f"\nclosed: reason={close_env.event_data.get('reason')!r}")

    await user_hc.close()
    await router_hc.close()
    await billing_hc.close()
    await technical_hc.close()
    await general_hc.close()
    await hub_obj.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Output

```console
channel: 8b4d...

           user: I tried to upgrade my subscription but the API is returning a 500 error. The status page says everything is green. Help?
  [tool] classify_as_technical('API returning 500 error during subscription upgrade — technical issue')
         router: [Handed off via classify_as_technical] API returning 500 error during subscription upgrade — technical issue
      technical: Capture the full request (endpoint, payload, response headers, and request-id) and re-try the upgrade — if the 500 persists, share the request-id so we can trace it server-side; check whether the failure is tied to a specific plan tier, payment method, or coupon, as those code paths are the most common 500 sources during upgrades.

closed: reason='technical_resolved'
```
