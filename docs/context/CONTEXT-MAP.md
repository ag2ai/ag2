# Context Map

This repository's domain glossaries, one file per bounded context. Each file is a
glossary only — canonical terms and their meanings, devoid of implementation detail.

| Context | File | Scope |
|---------|------|-------|
| Agent invocation | [beta-agent-invocation.md](./beta-agent-invocation.md) | How callers invoke an `autogen.beta` agent and consume its result (`ask`, `run`, replies, streams). |

System-wide architectural decisions live in [`docs/adr/`](../adr/).
