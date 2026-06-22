---
title: Pattern Cookbook
sidebarTitle: Overview
---

Side-by-side translations of the canonical multi-agent orchestration
patterns from classic (non-beta) AG2 (`#!python autogen.agentchat.group`) onto the
beta `#!python WorkflowAdapter`. Each pattern lists the classic primitives
it used, a beta translation, the full runnable source, and any gaps
that don't yet have a clean equivalent.

For the classic (non-beta) versions and their conceptual background, see the
[Pattern Cookbook in the user guide](../../../user-guide/advanced-concepts/pattern-cookbook/overview.md).

## Status Legend

| Status | Meaning |
|---|---|
| **Clean** | Maps one-to-one onto existing beta primitives. Code below works as written. |
| **Partial** | Translates with one or more named gaps. Workaround included; gap tracked on roadmap. |

## Quick Reference

| # | Pattern | Status | Beta primitives | Notable gap |
|---|---|---|---|---|
| 1 | [Pipeline](pipeline.md) | Clean | `#!python TransitionGraph.sequence` or `#!python FromSpeaker → AgentTarget` chain | — |
| 2 | [Hierarchical](hierarchical.md) | Partial | `#!python Handoff` returns from delegate tools + `#!python ContextEquals` for terminate | No `#!python NestedChatTarget`; sub-flows via separate channels |
| 3 | [Star](star.md) | Clean | `#!python ToolCalled` graph rule for spoke selection + WAL-reading synthesis tool | — |
| 4 | [Coordinator](coordinator.md) | Clean | One generic `#!python handoff(to)` tool returning `#!python Handoff` + `#!python finish` tool returning `#!python Finish`; empty transitions list + `#!python RevertToInitiatorTarget` default | — |
| 5 | [Escalation](escalation.md) | Clean | `#!python Handoff` returns from escalate tools + `#!python ToolCalled("resolve")` for terminate | — |
| 6 | [Redundant](redundant.md) | Partial | Sequential fan-out via `#!python TransitionGraph.sequence` | No parallel dispatch; one specialist at a time |
| 7 | [Feedback Loop](feedback_loop.md) | Clean | `#!python ContextEquals` on a `done` flag with `#!python max_turns` cap | — |
| 8 | [Context-Aware Routing](context_aware_routing.md) | Partial | Router agent's tool sets `#!python category` → `#!python ContextEquals` per branch | No `#!python LLMCondition`; routing reasoning lives inside the coordinator's LLM, not the framework |
| 9 | [Triage with Tasks](triage_with_tasks.md) | Clean | `#!python TransitionGraph.sequence` over a triage-produced plan; `#!python knobs["context_vars"]` seeds the queue | — |

The "Organic" pattern from classic AG2 (LLM-driven `#!python AutoPattern` group-manager auto-selection) translates to the [Coordinator](coordinator.md) pattern: one generic parameterised handoff tool on the coordinator, plus a `#!python Finish` tool to terminate. No graph rewrite when specialists are added or removed. See [Roadmap](#roadmap) for the remaining gaps.

## Routing idioms

Every cookbook page below uses one of three composable routing idioms.
Picking the right one is the most common decision when porting a
classic pattern.

**1. Static routing, graph-centralised.** Plain `#!python @tool` returning
a string; the graph holds the routing edge.

```python linenums="1"
async def delegate_researcher(reason: str) -> str:
    """Send the work to the research specialist."""
    return f"routing: {reason}"

# in the graph:
Transition(when=ToolCalled("delegate_researcher"),
           then=AgentTarget(researcher.agent_id))
```

When to use: many routing edges across multiple agents and you want
one place (the graph) to read the topology. Used by `03_star.py`.

**2. Static routing, tool-local.** `#!python @tool` returning a fixed-target
`#!python Handoff`. No graph rule needed — the framework reads
`#!python Handoff.target` from the agent's local `#!python ToolResultEvent`
stream and stamps it onto the packet's routing field directly.

```python linenums="1"
async def delegate_researcher(reason: str) -> Handoff:
    """Send the work to the research specialist."""
    return Handoff(target="researcher", reason=reason)
```

When to use: simple workflows where the routing target is obvious
from the tool's name and you'd rather not maintain a parallel graph
rule. Used by `02_hierarchical.py` and `04_escalation.py`.

**3. Dynamic routing.** `#!python @tool` returning a *computed-target*
`#!python Handoff`. The case that wasn't cleanly expressible before
the typed return shape.

```python linenums="1"
async def smart_route(query: str, state: ChannelStateInject) -> Handoff:
    target = pick_best_specialist(query, state.context_vars)
    return Handoff(target=target, reason="routed by load")
```

When to use: target depends on runtime state, load balancing, or any
condition the graph can't express declaratively.

## State updates

State mutations are independent of routing and use the workflow-scoped
helpers `#!python set_context(channel, key, value)` and
`#!python delete_context(channel, key)`. They live in
`#!python autogen.beta.network.workflow_helpers` rather than the
top-level `#!python autogen.beta.network` namespace — they only operate
on `#!python WorkflowState.context_vars` and raise `#!python RuntimeError`
on a non-workflow channel, so the import path itself signals the
adapter scope.

```python linenums="1" hl_lines="1"
from autogen.beta.network.workflow_helpers import set_context

async def classify_as_billing(reason: str, channel: ChannelInject) -> str:
    """Classify the request as billing. Sets context_vars['category']."""
    await set_context(channel, "category", "billing")
    return f"classified as billing: {reason}"
```

## Knowing when the workflow is finished

Every cookbook page below uses the same close-detection primitive in
its demo `#!python main`:

```python linenums="1"
close_env = await intake.wait_for_channel_event(
    channel_id=channel.channel_id,
    predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
    timeout=180.0,
)
print(f"closed: reason={close_env.event_data.get('reason')!r}")
```

`#!python AgentClient.wait_for_channel_event` blocks on the per-channel
inbox until an envelope matching the predicate arrives — here, the
`#!python EV_CHANNEL_CLOSED` envelope every termination route emits.
The reason string distinguishes the close (e.g. `'sequence_complete'`,
`'approved'`, `'resolved'`, `'fall_through'`, `'max_iterations'`,
`'ttl_expired'`).

To print the transcript, dump the WAL once after close — no polling
helper needed (this is the pattern used in each cookbook page below).

For the full picture of the five close routes (application close,
agent-side tool, adapter sentinel, workflow `#!python TerminateTarget`,
TTL / expectations), see
[Closing Channels → Watching for Close](../termination.md#watching-for-close).

## Roadmap

The gaps surfaced in each pattern page are tracked. None block the
patterns; each either has a working workaround or is currently
deferred:

* **`#!python LLMCondition`** — declarative routing where the framework asks an LLM whether a transition fires. Unblocks the cleanest version of Context-Aware Routing.
* **`#!python NestedChatTarget`** — first-class sub-flow target. Unblocks Hierarchical and Redundant fully.
* **Parallel dispatch** — fan out to multiple speakers within a single workflow turn, gather their replies, advance once all are in. Unblocks parallel Redundant.
* **Auto-merge of tool-return fields into context** — the `#!python ReplyResult.context_variables` analogue. Cuts the boilerplate for Escalation and Triage-with-Tasks.
* **`#!python AllOf` / `#!python AnyOf` condition composers** — useful sugar across most patterns; today the conjunction is encoded via transition order.

For the broader scope of what the workflow adapter does and doesn't
yet ship versus classic, see [Migrating from Group Chat](../migration_from_group_chat.md).

## See Also

- [Workflow Adapter](../workflow.md) — the underlying graph machinery and built-in conditions.
- [Context Variables](../context_variables.md) — `#!python set_context` / `#!python delete_context`, `#!python ContextEquals`, custom conditions.
- [Closing Channels](../termination.md) — termination patterns; many cookbook entries lean on `#!python TerminateTarget` with informative reasons.
- [Migrating from Group Chat](../migration_from_group_chat.md) — concept-level translation table.
