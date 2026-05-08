---
sidebarTitle: AG2 Beta
title: AG2 Beta
---

## Why did we create AG2 Beta?

The original **AutoGen** project released with its first public preview in September 2023, and **AG2** later diverged from that codebase in November 2024 to continue building on its core ideas.

**AutoGen** was one of the earliest frameworks for building AI agents and orchestrating agent-to-agent collaboration. That early vision proved valuable: it enabled real-world systems, informed the design of many tools, and helped shape the agent ecosystem.

Since then, the agent landscape has changed significantly. Over time, the community has established better practices, common protocols, and new interoperability standards. Capabilities that were once experimental are now becoming part of the expected foundation for agent platforms.

Examples include:

- [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol), introduced in November 2024
- [Agent2Agent (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/), introduced in April 2025
- [AG-UI](https://docs.ag-ui.com/introduction), introduced in May 2025

We have increasingly found that the original architecture inherited from **AutoGen** challenged the adoption of new ideas. Shipping modern capabilities inside the original design often requires introducing complexity, unnecessary migration effort, or compatibility compromises.

Not every part of the ecosystem is standardized yet, but the direction is clear. AI agents are no longer an experiment; they are standard application infrastructure.

**AG2 Beta** is our way to move forward with a future-focused foundation while applying the lessons we learned from building and operating hundreds of agent systems based on **AG2**.

## What is AG2 Beta?

**AG2 Beta** is a new development track inside **AG2** where we build capabilities that would be difficult or impractical to introduce on the original framework architecture.

We expect it to become the primary foundation for future AG2 agent development and production-ready multi-agent systems. Therefore, it will become V1.0 of AG2.

## Why use AG2 Beta?

**AG2 Beta** is built around a small, predictable, core and a set of opt-in primitives you compose to fit your application. Here is what you get out of the box.

### 1. A clean, async-first agent API

Two methods cover the conversational surface — `agent.ask(...)` to start a turn and `reply.ask(...)` to continue one. The agent loop, tool execution, and LLM calls are async throughout, with streaming enabled by default on supported providers.

### 2. A composable harness for capable, long-running, agents

AG2's harness layers on powerful primitives to the base Agent — **assembly policies** for context shaping, a **knowledge store** for persistent memory, **sub-task delegation** with isolated streams, and **middleware** for retries, logging, token limits, and history management. Build up the agent you need with the harness doing the heavy lifting for you.

The Agent's ability to fan out work, in parallel, to a team of specialist agents (seen as tools), or as subtasks, provides natural orchestration within each agent.

### 3. Production and Scalability

**Human-in-the-loop** hooks, **structured output** (static, callable, prompted, and transformable), **OpenTelemetry** tracing, **persistent backends** for history and streams (e.g. Redis), and a **testing utility** that mocks LLMs and tool calls without hitting the network — the primitives you need to take an Agent from prototype to production.

The runtime is async end-to-end, so a single process can drive many concurrent agents, tool calls, and provider streams without blocking, and sub-tasks fan out in parallel via `asyncio.gather`. State is externalised behind protocols — `History`, `Storage`, and `Stream` can be backed by Redis, a database, or anything you build — so agents stay effectively stateless and horizontal scaling is straightforward. Cross-cutting concerns like retries, rate limits, token budgets, and history compaction are middleware you compose onto an Agent.

### 4. UI and external integration

Every event in the agent loop — model requests and responses, tool calls and results, human-input requests, observer alerts — flows through an event stream. Streams, with natural filtering capabilities, can power UIs, logging, metrics, or approvals without touching the agent itself.

The stream is bidirectional: **AG-UI** renders model output and tool calls in real time while user responses come back as `HumanMessage` events, and persistent backends like Redis let separate processes — a web frontend and a background worker — share the same live conversation.

### 5. Tools, toolkits, and built-in tools

Define tools with a `@tool` decorator on plain functions. Use **type hints**, **dependency injection** (`Context`, `Inject`, `Variable`), and **toolkits** to organize related capabilities. Wire in **built-in tools** (web search, code execution, shell, memory) or expose any agent as a tool with `Agent.as_tool(...)`.

### 6. One configuration model across providers

A single, type-safe, interface spans **OpenAI**, **OpenAI Responses**, **Anthropic**, **Gemini**, **Vertex AI**, **Ollama**, and **DashScope**. Switching providers is a config change, not a rewrite, and structured output, multimodality (images, audio, video), and built-in tools work consistently across them.

## Compatibility with AG2

We value all the users and contributors who have made AG2 what it is today and want to bring you along on this journey.

Where it is practical and beneficial, we aim to preserve compatibility with existing AG2 workflows.

From day one, **AG2 Beta** agents can participate in established **AG2** multi-agent interaction patterns. This makes it possible to adopt **Beta** agents gradually within existing systems instead of rewriting everything at once.

## How do I try it out?

The **AG2 Beta** currently resides in the AG2 repository alongside the current AG2 code.

When you `pip install ag2`, this will include the current **AG2 Beta** release under the `autogen.beta` module.

If you want the most up-to-date version, use the `main` branch. To see current work in progress, view the [GitHub repository](https://github.com/ag2ai/ag2/issues?q=state%3Aopen%20label%3Abeta) for PRs with the `beta` label.

See the following pages for walkthroughs of the new **AG2 Beta** API.

## Current Focus Areas

**AG2 Beta** is actively focused on:

- improving the single-agent developer experience
- providing stronger context and memory management primitives
- simplifying integration with real applications, including Text UI, web, ambient, and background runtimes
- enabling new multi-agent coordination patterns that are not feasible in the current AG2 architecture
- supporting emerging standards and protocols across the AI agent ecosystem

We are building **AG2 Beta** to make agent development simpler, more modern, and easier to integrate into production-grade applications. We would love your feedback as the API evolves ([Discord](https://discord.com/invite/pAbnFJrkgZ)).
