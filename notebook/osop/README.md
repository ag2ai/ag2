# OSOP Workflow Examples for AutoGen

[OSOP](https://github.com/Archie0125/osop-spec) (Open Standard for Orchestration Protocols) is a portable YAML format for describing multi-agent workflows — think OpenAPI, but for agent orchestration.

## What's Here

| File | Description |
|------|-------------|
| `multi-agent-debate.osop.yaml` | A structured debate between pro/con agents with moderator synthesis and human judgment |

## Why OSOP?

AutoGen excels at multi-agent conversation patterns. OSOP provides a **framework-agnostic way to describe** those patterns so they can be:

- **Documented** — readable YAML that non-developers can understand
- **Validated** — check workflow structure before running it
- **Ported** — same workflow definition works across AutoGen, CrewAI, LangGraph, etc.
- **Visualized** — render the workflow as a graph in the [OSOP Editor](https://github.com/Archie0125/osop-editor)

## Quick Start

```bash
# Validate the workflow
pip install osop
osop validate multi-agent-debate.osop.yaml

# Or just read the YAML — it's self-documenting
cat multi-agent-debate.osop.yaml
```

## How It Maps to AutoGen

| OSOP Concept | AutoGen Equivalent |
|---|---|
| `node` with `type: agent` | `ConversableAgent` / `AssistantAgent` |
| `node` with `type: human` | `UserProxyAgent` |
| `edge` with `mode: parallel` | Parallel agent execution |
| `edge` with `mode: sequential` | `initiate_chat()` chain |
| `config.tools` | Agent tool registration |

## Learn More

- [OSOP Spec](https://github.com/Archie0125/osop-spec) — full specification
- [OSOP Examples](https://github.com/Archie0125/osop-examples) — 30+ workflow templates
- [OSOP Editor](https://github.com/Archie0125/osop-editor) — visual workflow editor
