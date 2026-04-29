---
name: ag2-overview
description: Map of AG2 beta capabilities and which sibling skill to reach for. Load first when the user mentions building with AG2 beta (autogen.beta) but the specific feature isn't yet clear — agents, tools, model config, delegation, memory, observers, structured output, HITL, AG-UI, telemetry, or testing.
license: Apache-2.0
---

# AG2 Beta — capability map

AG2 beta (`autogen.beta`) is an async, protocol-driven agent framework. The full reference docs live under `website/docs/beta/`. This skill is the index of sibling skills that cover the common build paths.

## When to use

Read this file first when a request mentions "AG2 beta", "autogen.beta", or building agents in this repo and you don't yet know which feature is needed. Use the table below to pick the right specialised skill, then load that skill's `SKILL.md` for the recipe.

## Pick the right skill

| User intent | Skill | What it covers |
|---|---|---|
| Build an `Agent` from scratch, pick a model | `ag2-quickstart` | `Agent`, `ModelConfig`, `ask()` / `reply.ask()` chaining, providers, env vars |
| Give the Agent a custom Python tool | `ag2-add-custom-tool` | `@tool`, sync/async, `ToolResult`, `Context`, `Inject`, `Variable`, `Depends` |
| Use shipped tools (web search, code exec, MCP, etc.) | `ag2-use-builtin-tools` | `WebSearchTool`, `WebFetchTool`, `CodeExecutionTool`, `MCPServerTool`, `ImageGenerationTool`, `MemoryTool`, `FilesystemToolkit`, `DuckDuckSearchTool`, `ExaToolkit`, `TavilySearchTool` |
| Run shell commands from an agent | `ag2-shell-tool` | `LocalShellTool` (any provider), provider-side `ShellTool`, sandboxing (`allowed`/`blocked`/`ignore`/`readonly`) |
| Get typed Pydantic / dataclass output | `ag2-structured-output` | `response_schema=`, `ResponseSchema`, `@response_schema`, `PromptedSchema`, `reply.content()`, retries |
| Multi-agent: parallel subtasks or named delegates | `ag2-subagent-delegation` | `tasks=TaskConfig()`, `run_subtasks(parallel=True)`, `Agent.as_tool()`, `persistent_stream` |
| Pause for human input or gate a tool with approval | `ag2-hitl` | `context.input()`, `hitl_hook`, `approval_required()` middleware |
| Logging, retry, history-trim, custom interception | `ag2-middleware` | `BaseMiddleware`, `LoggingMiddleware`, `RetryMiddleware`, `HistoryLimiter`, `TokenLimiter`, tool middleware |
| Test agents and tools | `ag2-testing` | `TestConfig`, mocking LLM responses, simulating `ToolCallEvent` |
| Persistent memory across runs, history compaction, assembly | `ag2-knowledge-and-memory` | `KnowledgeStore`, `KnowledgeConfig`, `WorkingMemoryAggregate`, `AssemblyPolicy`, `SlidingWindowPolicy`, `TokenBudgetPolicy`, `TailWindowCompact`, `SummarizeCompact` |
| Observability, alerts, halts | `ag2-observers-and-alerts` | `BaseObserver`, `TokenMonitor`, `LoopDetector`, `EventWatch`, `CadenceWatch`, `AlertPolicy`, `HaltEvent` |
| Send images / audio / video / PDFs in | `ag2-multimodal-input` | `ImageInput`, `AudioInput`, `VideoInput`, `DocumentInput`, `FilesAPI` |
| Web frontend via the AG-UI protocol | `ag2-ag-ui` | `AGUIStream`, FastAPI mount, CopilotKit |
| OpenTelemetry traces / metrics | `ag2-telemetry` | `TelemetryMiddleware`, GenAI semconv attributes, content capture |

## Project conventions for any skill that writes code into the AG2 repo

These are repo-wide rules from `CLAUDE.md`. Apply them whenever generating code that lands in `autogen/beta/`:

- Do **not** use `from __future__ import annotations`.
- All top-level imports — no function-level imports unless explicitly allowed.
- No nested functions in runtime execution paths (decorator factories are fine).
- No side effects in `__init__` — apply them at runtime.
- Internal filesystem paths use `pathlib.Path`; public signatures accept `str | os.PathLike[str]`.
- Common reusable APIs come from the `autogen.beta` top-level (e.g. `from autogen.beta import Agent, tool, Context`); advanced/specialised APIs come from sub-modules (`autogen.beta.middleware`, `autogen.beta.config`, etc.).

## Beta-doc cross-reference

If a skill's recipe is incomplete for the case at hand, the source docs are at `website/docs/beta/`. Each sibling skill points to its primary `.mdx` files in its own "Going deeper" section.
