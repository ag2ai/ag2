---
title: Migrating from Group Chat
sidebarTitle: Migration · Group Chat
---

This page is for users porting an existing AG2 deployment that uses `GroupChat` plus `Handoffs`-style orchestration onto the new `#!python autogen.beta.network` module. The modern equivalent is the [`#!python WorkflowAdapter`](workflow.md), driven by a declarative `#!python TransitionGraph`.

The translation is mostly mechanical. The two systems share the same vocabulary — speakers, conditions, targets, terminations — but the new one is data-first (a `#!python TransitionGraph` is JSON-serialisable and survives `#!python Hub.hydrate()`) and runs on top of the network's hub-and-spoke architecture instead of an in-process turn manager.

For per-pattern translations of the canonical orchestrations (Pipeline, Star, Feedback Loop, Triage-with-Tasks, etc.), jump to the [Pattern Cookbook](pattern_cookbook/pattern_cookbook.md).

## Concept Mapping

| Classic (non-beta) concept | Beta network equivalent | Notes |
|---|---|---|
| `GroupChat(agents=[...])` | A `#!python workflow` channel with the agents as participants | The hub plays the role of `GroupChatManager`. |
| `GroupChatManager` | The `#!python WorkflowAdapter` + `#!python Hub` | Turn-taking enforcement moves into the hub; the adapter validates each send against the graph. |
| `Agent.handoffs` | A `#!python TransitionGraph` (per-channel, not per-agent) | Handoffs are described once at the channel level — easier to reason about and serialisable. |
| `AgentTarget(agent)` | `#!python AgentTarget(agent_id)` | Same name; takes an `agent_id` instead of an agent reference. |
| `RevertToInitiator()` | `#!python RevertToInitiatorTarget()` | Identical semantics. |
| `Stay()` | `#!python StayTarget()` | Identical. |
| `Terminate()` | `#!python TerminateTarget(reason="…")` | Now carries an explicit reason that lands on `#!python EV_CHANNEL_CLOSED`. |
| `OnContextCondition(...)` | A custom `#!python TransitionCondition` | Implement the Protocol; register via `#!python register_condition(...)`. |
| `OnCondition(...)` | A custom `#!python TransitionCondition` | Same. |
| `ReplyResult(target=...)` from a tool | `#!python EV_HANDOFF` envelope, paired with `#!python ToolCalled("...")` | The default handler emits `#!python EV_HANDOFF` automatically when a tool's result implies a handoff. |
| `FunctionTarget(fn)` | A custom `#!python TransitionTarget` whose `resolve()` runs Python | Same pattern; register via `#!python register_target(...)`. |
| `max_round=N` | `#!python TransitionGraph(..., max_turns=N)` | Same hard cap. |

## Side-by-Side: Round Robin

**Classic (non-beta):**

```python linenums="1"
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[alice, bob, carol],
    speaker_selection_method="round_robin",
    max_round=6,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
alice.initiate_chat(manager, message="Topic: should every dev learn Rust?")
```

**Beta network:**

```python linenums="1" hl_lines="6-9 11-16"
from autogen.beta.network import (
    Hub, HubClient, LocalLink, Passport, Resume,
    TransitionGraph,
)

graph = TransitionGraph.round_robin(
    participants=[alice.agent_id, bob.agent_id, carol.agent_id],
    max_turns=6,
)

channel = await alice.open(
    type="workflow",
    target=[bob.agent_id, carol.agent_id],
    knobs={"graph": graph.to_dict()},
)
await channel.send("Topic: should every dev learn Rust?")
```

The auto-termination on `max_turns` lands on `#!python EV_CHANNEL_CLOSED` with `reason="round_robin_complete"`.

!!! note
    `#!python discussion` is also a round-robin channel — but it never auto-terminates and has no turn cap. Use it when the application decides when to stop. Use `#!python workflow + TransitionGraph.round_robin` when you want a hard cap and an `#!python EV_CHANNEL_CLOSED` to wait on.

## Side-by-Side: Pipeline / Sequential

**Classic:**

```python linenums="1"
researcher.handoffs.add_to([
    OnCondition(target=AgentTarget(writer)),
])
writer.handoffs.add_to([
    OnCondition(target=AgentTarget(editor)),
])
editor.handoffs.set_after_work(target=Terminate())

groupchat = GroupChat(agents=[researcher, writer, editor])
manager = GroupChatManager(...)
researcher.initiate_chat(manager, message="Topic: how does HTTPS work?")
```

**Beta network:**

```python linenums="1"
graph = TransitionGraph.sequence([
    researcher.agent_id, writer.agent_id, editor.agent_id,
])

channel = await researcher.open(
    type="workflow",
    target=[writer.agent_id, editor.agent_id],
    knobs={"graph": graph.to_dict()},
)
await channel.send("Topic: how does HTTPS work?")
```

Auto-terminates with `reason="sequence_complete"`.

## Side-by-Side: Conditional Handoff

A common classic pattern: a triage agent inspects the user request and routes to a specialist via a tool call.

**Classic:**

```python linenums="1"
@triage.register_for_llm(description="Escalate to security review.")
def escalate(reason: str) -> ReplyResult:
    return ReplyResult(
        target=AgentTarget(security_reviewer),
        message=f"Escalated: {reason}",
    )

triage.handoffs.add_to([
    OnCondition(target=AgentTarget(general_responder)),
])
groupchat = GroupChat(agents=[triage, security_reviewer, general_responder])
```

**Beta network:**

```python linenums="1" hl_lines="6-27"
from autogen.beta.network import (
    AgentTarget, Always, FromSpeaker, RevertToInitiatorTarget,
    TerminateTarget, ToolCalled, Transition, TransitionGraph,
)

graph = TransitionGraph(
    initial_speaker=triage.agent_id,
    transitions=[
        # If triage's tool said "escalate", route to the security reviewer.
        Transition(
            when=ToolCalled("escalate"),
            then=AgentTarget(security_reviewer.agent_id),
        ),
        # Once security has spoken, terminate.
        Transition(
            when=FromSpeaker(security_reviewer.agent_id),
            then=TerminateTarget(reason="security_review_complete"),
        ),
        # Otherwise, the general responder handles it.
        Transition(
            when=FromSpeaker(triage.agent_id),
            then=AgentTarget(general_responder.agent_id),
        ),
    ],
    default_target=TerminateTarget(reason="default"),
    max_turns=10,
)
```

The triage agent has its own `#!python @triage_agent.tool` definition for `#!python escalate`; the default handler emits an `#!python EV_HANDOFF` envelope when the tool returns and the next-speaker rule sees `#!python ToolCalled("escalate")` match.

!!! tip
    Tool-driven handoffs (`#!python ToolCalled(...)`) are the cleanest migration path for `ReplyResult`-style classic logic. Define one tool per branch you want the LLM to take, and write a `#!python ToolCalled` transition for each.

## Side-by-Side: Function Target / Custom Routing

`FunctionTarget(fn)` runs Python to decide who's next. The new equivalent is a custom `#!python TransitionTarget`.

**Classic:**

```python linenums="1"
def route_by_topic(state):
    if "security" in state.last_message.lower():
        return security
    return general

triage.handoffs.add_to([
    OnCondition(target=FunctionTarget(route_by_topic)),
])
```

**Beta network:**

```python linenums="1"
from dataclasses import dataclass, field
from typing import ClassVar

from autogen.beta.network import (
    Envelope, TransitionDecision, TransitionTarget, register_target,
)

@dataclass(slots=True)
class TopicRouter(TransitionTarget):
    name: ClassVar[str] = "topic_router"
    security_id: str = ""
    general_id: str = ""

    def resolve(self, state, envelope: Envelope) -> TransitionDecision:
        text = envelope.event_data.get("text", "").lower()
        if "security" in text:
            return TransitionDecision(next_speaker=self.security_id)
        return TransitionDecision(next_speaker=self.general_id)

register_target(TopicRouter)

# Then in the graph:
Transition(
    when=FromSpeaker(triage.agent_id),
    then=TopicRouter(security_id=security.agent_id, general_id=general.agent_id),
)
```

The custom target dataclass is JSON-serialisable, so it round-trips through `#!python TransitionGraph.to_dict()` and survives `#!python Hub.hydrate()`. That's the win over `FunctionTarget`: state is data, not closures.

## What's Different (Beyond the Translation)

A few changes that will affect how you architect the migration:

- **The hub is authoritative.** Turn-taking, state, audit, and expectation enforcement all live in the hub. Your agent code becomes thinner.
- **Channels have ids.** Every `#!python alice.open(...)` returns a channel id. Audit, replay, and inspection are scoped to it.
- **Default handlers do most of the heavy lifting.** When agents have `#!python attach_plugin=True`, you don't write any "agent receives message → run LLM → send reply" glue. The framework handles it.
- **Sub-task observation comes for free.** Calls to `#!python agent.task(..., capability="X")` inside a network turn auto-update `#!python Resume.observed["X"]`. There was no equivalent in the classic world. See [Task Observation](task_observation.md).
- **Expectations replace `max_round` exhaustively.** You can declare `#!python reply_within(60s, auto_close)` per channel type, get a violation logged to the audit log, and have the channel auto-close — see [Expectations & Audit](expectations_and_audit.md).
- **Views replace ad-hoc context windowing.** Each adapter has a default view (`#!python FullTranscript`, `#!python WindowedSummary`); custom views plug in via the `#!python ViewPolicy` Protocol — see [Views & Skills](views_and_skills.md).
- **The transport is pluggable.** `#!python LocalLink` is the in-process default, but the link-layer is a Protocol — cross-process and cross-host transports are drop-in replacements.

## Migration Checklist

1. Stand up a hub: `#!python hub = await Hub.open(MemoryKnowledgeStore())`. (Pick a `KnowledgeStore` based on whether you need persistence.)
2. Wrap each existing `Agent` in an `#!python AgentClient` via `#!python hc.register(...)`. Provide a `#!python Passport` (name) and a `#!python Resume` (claimed capabilities).
3. Translate your `Handoffs` configuration into a `#!python TransitionGraph`. Use `#!python round_robin` / `#!python sequence` factories where they fit; build the graph manually for conditional logic.
4. Replace `initiate_chat(...)` with `#!python alice.open(type="workflow", target=[...], knobs={"graph": graph.to_dict()})` followed by `#!python channel.send(text)`.
5. Wait for termination with `#!python alice.wait_for_channel_event(channel_id=..., predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED)`.
6. Inspect the WAL via `#!python hub.read_wal(channel_id)` for replay and the audit log via `#!python hub.audit_log.read_all()` for governance.

For a runnable end-to-end translation of the "researcher → writer → editor" handoff pattern, see the [Pipeline](pattern_cookbook/pipeline.md) entry in the Pattern Cookbook.
