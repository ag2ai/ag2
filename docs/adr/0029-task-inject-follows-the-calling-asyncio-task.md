---
status: accepted
date: 2026-11-01
---

# `TaskInject` resolves on the calling asyncio task, not the shared dependency slot

Follow-up to the `ag2.task` slot consistency fix (#3287) and issue #3565: while
sibling tool calls each open a task on one conversation, `TaskInject` inside an
earlier sibling resolved to a later one.

## Context

`Task.__aenter__` stamps itself into `context.dependencies["ag2.task"]` and
`TaskInject` (an `Inject` of that key) reads the slot back. The slot is one
value per conversation, but tasks can legitimately overlap on a conversation:
`ToolExecutor.execute_tools` gathers sibling tool calls concurrently on one
`Context`, and each may open `agent.task(...)`. With one slot, every injection
under every sibling read whichever task entered last.

#3287 made the slot *consistent* — never empty while a task runs, never an
exited task — but it cannot make it *correct*: a tool inside sibling A still
resolved sibling B whenever B entered later.

## Decision

**`TaskInject` resolves from a `ContextVar` chain scoped to the calling
asyncio task, validated against the conversation.**

- `Task.__aenter__` pushes an `_ActiveTaskNode(context, task, previous)` onto a
  module-level `ContextVar`; `__aexit__` resets it with the token. Each
  `asyncio.Task` — every gathered tool call, every `spawn_background` — copies
  the variable at creation, so a sibling that enters a block extends its own
  copy and resolves its own task.
- Resolution walks the chain to the first node naming the resolving
  conversation (`node.context is context`) whose task has not exited. The
  identity check keeps a task opened on one conversation invisible to
  `TaskInject` on another conversation started inside its block; the
  exited-node skip lets a background task spawned inside a nested block still
  reach the outer task while it runs.
- The `ag2.task` dependency slot keeps being written and restored exactly as
  before (#3287 bookkeeping included) for direct readers of
  `context.dependencies["ag2.task"]` and the `ag2.network.TASK_DEP` key.
  `TaskInject` no longer reads it. The slot names the most recently entered
  task still running on the conversation — best-effort under overlap, and now
  the only place that semantic exists.
- `ag2.network.client.inject.TaskInject` re-exports `ag2.task.TaskInject`
  instead of restating the annotation, so the two cannot drift.

Decision points #3565 raised, and their outcomes:

1. *Conversation boundary* — resolved by the `node.context is context` check
   at resolution time; nothing to key or clear on entry.
2. *The dependency slot* — kept, write-only as far as the framework's own
   injection is concerned; removing it would break code reading the key
   directly for no behavioural gain.
3. *Handlers in another asyncio task* — code that is not a descendant of the
   block (a subscriber whose subscription copied the context before entry,
   a background task spawned before the block) resolves `None`, not the wrong
   task. That is the honest answer: no task encloses that code.
4. *Subagents* — `run_task` / `as_tool` delegates run on their own
   conversation, so the identity check resolves `TaskInject` to `None` there;
   a delegation is its own task scope and already reports itself to the parent
   with `TaskStarted` / `TaskCompleted`.

## Consequences

- The inside-a-block question "which task am I in" now has one answer: the
  innermost enclosing block on this conversation, in this asyncio task —
  regardless of what sibling tool calls do or the order they exit in.
- Sequential nesting is unchanged: inner → inner; after inner exits → outer;
  after both → `None`. The token reset is what restores the outer node, so
  entry and exit must stay in the same asyncio task (`async with` already
  guarantees this); an exit that somehow ran elsewhere leaves the node, which
  resolution skips as exited.
- The `Inject` import path in `ag2.task` is replaced by a `CustomField`
  subclass; anyone mirroring the old `Annotated[..., Inject("ag2.task",
  default=None)]` shape in their own code should import `TaskInject` instead.
