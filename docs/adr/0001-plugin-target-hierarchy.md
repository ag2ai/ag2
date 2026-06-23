---
status: accepted
date: 2026-06-16
---

# Agent, LiveAgent, and Plugin share a three-layer mixin hierarchy

## Context

`Agent`, `LiveAgent`, and `Plugin` independently grew near-identical authoring
surfaces — `prompt()` / `observer()` / `tool()` decorators, middleware and
policy registration, the prompt-init loop, and the constructor that wires up
tools, observers, dependencies, a serializer, and a tool executor. The copies
drifted (e.g. an attribute named `tools` on one and `_tools` on another), and a
silent bug — `Agent.add_tool` delegating to an abstract hook — slipped in under
the duplication.

## Decision

Collapse the shared surface into two base classes in `autogen/beta/plugin.py`:

- **`PromptObserverMixin`** — the authoring/collection surface common to *all
  three*: `prompt`/`observer`/`tool` decorators, `add_middleware` /
  `insert_middleware` / `add_policy` / `add_observer`, `_init_prompts`, and an
  **abstract `add_tool`** that subclasses implement to choose storage.
- **`PluginTarget(PromptObserverMixin)`** — for the *runnable* targets (`Agent`,
  `LiveAgent`): the shared constructor body (`_init_target`), eager `add_tool`,
  the wrap-now `hitl_hook`, the `_tool_executor`, and `_apply_plugin`.

`Plugin` stays on the **bare mixin** — it is the *source* of contributions,
never a target. `Agent` and `LiveAgent` inherit `PluginTarget`; each constructor
is now a single `_init_target(...)` call plus its own type-specific state
(`Agent`: config / response_schema / tasks / knowledge / assembly / execution
loop; `LiveAgent`: `_config` / `_stream` / realtime `run()`).

## Consequences / things that look wrong but are deliberate

- **`_apply_plugin` lives on `PluginTarget`, not the mixin.** Applying a plugin
  mutates the agent-side surface (`add_middleware`, `_agent_dependencies`, …)
  that `Plugin` does not have. Putting it on the mixin would give `Plugin` a
  method it cannot satisfy.
- **`add_tool` is intentionally split.** `Plugin.add_tool` is *deferred* (raw
  append to `_tools`); `PluginTarget.add_tool` is *eager* (binds to
  `dependency_provider` immediately). This is the one genuine behavioural
  difference; the mixin's abstract `add_tool` is the seam. Likewise
  `Plugin.hitl_hook` stores the hook raw while `PluginTarget.hitl_hook` wraps it
  now — deferred wrapping happens in `_apply_plugin`.
- **There is no `Plugin.register` / `NetworkPlugin.register`.** Plugins are pure
  *collectors* — tools, middleware, observers, prompts, hooks, **and assembly
  policies** all accumulate on the `Plugin`, and the single `_apply_plugin`
  copies every category onto the target. Earlier a `register(agent)` hook
  existed so subclasses could add extra wiring; once policies became a collected
  contribution, the only remaining override (`NetworkPlugin` adding its context
  policy) reduced to `self.add_policy(...)` in `__init__`. Apply a plugin with
  `agent._apply_plugin(plugin)`, not `plugin.register(agent)`.
- **`LiveAgent` mirrors `Agent`'s public attribute names** (`tools`,
  `dependency_provider`) so the eager `add_tool` and `_init_target` are truly
  shared rather than near-copies.

To add a new collectible contribution: declare it on `PromptObserverMixin`,
initialise it in both `Plugin.__init__` and `PluginTarget._init_target`, and
copy it in `_apply_plugin`. To add a new runnable agent type: subclass
`PluginTarget`, call `_init_target` from its constructor, and add only the
type-specific state.
