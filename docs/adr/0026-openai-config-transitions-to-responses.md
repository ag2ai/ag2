---
status: accepted
date: 2026-10-27
---

# 0026. OpenAIConfig transitions to the Responses API in 1.4.0

## Context

`OpenAIConfig` currently selects Chat Completions. Responses is the intended
OpenAI default, but immediately rebinding the name would change endpoints and
reject Chat Completions parameters in existing applications.

## Decision

Expose the existing implementation as `OpenAICompletionsConfig`. Until 1.4.0,
`OpenAIConfig` remains a compatibility subclass with the same constructor,
copy behavior and client. Mark the class deprecated for type checkers and emit
a `DeprecationWarning` from `create()`, following ADR 0025's rule that
construction and copying stay free of side effects.

In 1.4.0, remove the deprecated Chat Completions binding and bind
`OpenAIConfig` to `OpenAIResponsesConfig`. Recommend the explicit classes
during the transition: `OpenAICompletionsConfig` preserves Chat Completions;
`OpenAIResponsesConfig` opts into Responses now.

## Consequences

Existing code keeps its endpoint during the deprecation period. The explicit
Chat Completions class remains supported after 1.4.0, including for compatible
endpoints. Applications migrating to Responses must translate API-specific
options; changing the name alone does not translate those options.
