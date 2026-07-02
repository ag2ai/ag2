---
title: Release History
sidebarTitle: Release History
---

# Release History — AG2 Classic and the Path to v1.0

AG2 Classic is the original AG2 framework — `autogen.agentchat` with `ConversableAgent`, `GroupChat`, `LLMConfig`, and the surrounding agents, tools, and capabilities. It is published on PyPI as the `autogen` package and lives in the [ag2classic](https://github.com/ag2ai/ag2classic) repository.

This page records how AG2 Classic evolved and the path that led to AG2 v1.0.

!!! note "Where things stand today"

    - **AG2 Classic** (the `autogen` package) is in **maintenance mode** — it receives critical and security fixes only.
    - **AG2 v1.0+** (the `ag2` package) is the actively developed line. Its documentation lives at [docs.ag2.ai](https://docs.ag2.ai).

## Background

For most of its life, AG2 shipped a single framework built around `ConversableAgent` and `GroupChat`. As the project matured, two goals emerged:

- **Tidy up the original framework** through a round of deprecations, then move it to long-term maintenance.
- **Build a new, redesigned framework** — developed under `autogen.beta` and centred on a new `Agent` — that would become the official AG2 at v1.0.

The releases below trace that transition from v0.12 through to v1.0.

## v0.12 — Deprecation notices

This release opened the deprecation and feedback period that prepared the original framework for maintenance mode.

- **Deprecation announcements.** The features listed below were marked as deprecated and scheduled for removal in v0.14.
- **Community feedback window opened.** Users were invited to flag any deprecation-bound feature that was critical to their workflow before v0.14.
- Beta development continued in parallel.

### Deprecated agents

| Agent | Alternative |
|-------|-------------|
| `GPTAssistantAgent` | `ConversableAgent` |
| `LLaVAAgent` | AG2 v1.0 provides native multimodal support on `Agent` |
| `WebSurferAgent` (contrib) | `autogen.agents.experimental.WebSurferAgent` |
| `TextAnalyzerAgent` | `ConversableAgent` with an appropriate system message |
| `MathUserProxyAgent` | `ConversableAgent` with tool calling |
| `SocietyOfMindAgent` | GroupChat patterns |
| `AgentOptimizer` | `ConversableAgent` with tool calling |
| `RetrieveAssistantAgent` | `AssistantAgent` |
| `QdrantRetrieveUserProxyAgent` | `RetrieveUserProxyAgent` with `vector_db='qdrant'` |
| `SwarmAgent` | `ConversableAgent` (already incorporated) |
| `RealtimeAgent` | Relied on deprecated realtime API endpoints |

### Deprecated swarm functions and the swarm module

The entire `autogen.agentchat.contrib.swarm_agent` module was removed in v0.14. All swarm orchestration is now provided by group chat.

| Feature | Alternative |
|---------|-------------|
| `initiate_swarm_chat()`, `run_swarm()` (and async variants) | `run_group_chat` |
| `AFTER_WORK`, `AfterWork`, `AfterWorkOption` | Group chat handoffs (`Handoffs`, transition targets) |
| `ON_CONDITION`, `OnCondition` | `autogen.agentchat.group.OnCondition` / `OnContextCondition` |

### Deprecated capabilities and modules

| Feature | Reason |
|---------|--------|
| `ImageGeneration` capability | Depended on the deprecated `TextAnalyzerAgent` |
| `agent_eval` module (`generate_criteria`, `quantify_criteria`, `CriticAgent`, `QuantifierAgent`, `SubCriticAgent`) | Experimental evaluation framework |

### Deprecated interoperability

| Feature | Alternative |
|---------|-------------|
| `CrewAIInteroperability` | LangChain or PydanticAI interop |

### Deprecated experimental tools

| Tool | Alternative |
|------|-------------|
| `PythonCodeExecutionTool` | AG2 v1.0 `CodeExecutionTool` |
| `FirecrawlTool` | `Crawl4AITool` or `BrowserUseTool` |
| `SearxngSearchTool` | `DuckDuckGoSearchTool` or `TavilySearchTool` |
| `WebSearchPreviewTool` | AG2 v1.0 `WebSearchTool` |

### What was *not* deprecated

The core framework and actively maintained features carried through unchanged, and they remain supported in AG2 Classic today:

- **Core agents**: `ConversableAgent`, `AssistantAgent`, `UserProxyAgent`
- **Group chat**: `GroupChat`, `GroupChatManager`, `run_group_chat`, and all orchestration patterns including handoffs (`Handoffs`, `OnCondition`, `OnContextCondition`, and transition targets)
- **RAG**: `RetrieveUserProxyAgent`, vector database integrations (ChromaDB, Qdrant, MongoDB, Couchbase, PGVector), graph RAG (Neo4j, FalkorDB)
- **Capabilities**: `Teachability`, `VisionCapability`, `TransformMessages`, `ToolsCapability`
- **Experimental agents**: `WebSurferAgent` (experimental), `DeepResearchAgent`, `ReasoningAgent`, `DocAgent`, `CaptainAgent`, `A2UIAgent`, and platform agents (Discord, Slack, Telegram)

### Notebooks and documentation removed in v0.14

A number of notebooks that referenced unavailable or outdated models (e.g. `gpt-3.5-turbo`, `gpt-4-vision-preview`, `gpt-4-turbo-preview`) were removed in v0.14, along with the associated blog posts and user guide pages for the deprecated features above.

## v0.13 — Transition period

- **Community feedback incorporated.** Deprecation decisions were revisited based on feedback gathered during the v0.12 window.
- **Beta API refinements.** Development and stabilisation of the beta API continued, and its orchestration capabilities were introduced.
- This was the last release in which the deprecated features were still available.

## v0.14 — Final Classic release

- **Deprecated features removed.** Everything marked deprecated in v0.12 (and not rescued by community feedback) was removed. See the lists above.
- **Beta moved to Release Candidate (RC).** The beta API became feature-complete and entered final testing.
- This was the last feature release of the original AG2 codebase.

## v1.0 — The new AG2

- **The redesigned framework became stable** and took over as **AG2 v1.0**, published as the `ag2` package.
- **The original framework became AG2 Classic** — it continues as the `autogen` package in maintenance mode, developed in the [ag2classic](https://github.com/ag2ai/ag2classic) repository.
- **Documentation** — Primary documentation now covers AG2 v1.0 at [docs.ag2.ai](https://docs.ag2.ai); this AG2 Classic documentation remains available at [classic.docs.ag2.ai](https://classic.docs.ag2.ai).

## Migrating to AG2 v1.0

AG2 Classic will be supported with critical and security fixes. When you're ready to move to the new framework — with its redesigned `Agent`, tools, and multi-agent capabilities — head to [docs.ag2.ai](https://docs.ag2.ai) for the v1.0 guides and migration notes.

## Community feedback

As an open-source project, we welcome feedback from the AG2 Classic community. If you rely on Classic and have questions, run into issues, or want to share what you need, please open an Issue or reach out on our [Discord](https://discord.com/invite/pAbnFJrkgZ) server.
