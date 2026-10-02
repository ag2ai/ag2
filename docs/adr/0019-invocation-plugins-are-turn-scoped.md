---
status: accepted
date: 2026-10-02
---

# Invocation plugins contribute to one turn

## Context

A constructor plugin is applied once. In a long-lived agent, rebuilding a
`SkillPlugin` refreshes its catalog and tool schemas, but applying it to the
shared agent would accumulate contributions and affect concurrent calls.

## Decision

`Agent.ask`, `run`, `resume`, and `AgentReply.ask` / `run` accept `plugins=`.
Invocation plugins contribute tools, static and dynamic prompts, middleware,
observers, assembly policies, dependency and variable defaults, and a HITL hook
inside the existing turn scope. They never mutate the agent.

Plugin prompt fragments are appended to the resolved invocation prompt,
including an explicit `prompt=` override. The override still suppresses the
agent's base and dynamic prompts. Invocation dynamic hooks run once per call.
Plugin prompt fragments and newly supplied defaults are removed on scope exit,
including failure and cancellation; reply continuations do not inherit them.
Changes to other context variables remain available to continuations.

Later invocation plugins override earlier defaults and same-named tools.
Existing context values override plugin defaults; explicit invocation tools
override plugin tools. HITL precedence is explicit call hook, agent hook, then
the first invocation plugin hook. Policies compose after agent policies in one
assembler. The stream lock covers plugin binding through cleanup.

## Consequences

A caller can construct `SkillPlugin(runtime)` per message while retaining the
agent and its long-lived resources. Runtime discovery caches still need their
normal invalidation; reusing a plugin reuses its construction-time snapshot.
Constructor plugins retain their existing lifetime and behavior.
