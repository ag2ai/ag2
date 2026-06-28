# Context Map

Bounded-context glossaries for the codebase. Each entry is a tight,
implementation-free list of the domain terms that context uses, kept here
(out of the source tree) so the language stays discoverable and consistent.

## Contexts

- [Agent invocation](./beta-agent-invocation.md) — how callers invoke an
  `ag2` agent and consume its result (`ask`, `run`, replies, streams).
- [Skills](./skills.md) — `agentskills.io` progressive-disclosure: discover skill
  folders, surface them to the model, load / read / execute on demand.

System-wide architectural decisions live in [`docs/adr/`](../adr/).
