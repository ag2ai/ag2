---
name: ag2-quickstart
description: Build a minimal AG2 beta `Agent` end to end — pick a model provider, set a prompt, call `agent.ask()`, then continue the conversation with `reply.ask()` (multi-turn). Use when the user is starting a new AG2 beta project, has no working `Agent` yet, or needs the multi-turn chaining pattern. Covers `OpenAIConfig`, `AnthropicConfig`, `GeminiConfig`, `OllamaConfig` etc., and env-var fallback for API keys.
license: Apache-2.0
---

# Quickstart: build your first AG2 beta Agent

## When to use

- The user is starting from a blank file and wants a working AG2 beta agent.
- The user is unsure which provider config to use.
- The user wants to chain follow-up turns without losing conversation context.
- A larger task needs the basic Agent setup as its skeleton — start here, then layer the relevant feature skill on top.

## 60-second recipe

```python
import asyncio
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig

async def main() -> None:
    agent = Agent(
        "assistant",
        prompt="You are a helpful assistant. Reply in one sentence.",
        config=OpenAIConfig(model="gpt-4o-mini"),
    )

    # First turn
    reply = await agent.ask("What is the capital of France?")
    print(reply.body)

    # Continue the same conversation — context is preserved
    reply = await reply.ask("And of Germany?")
    print(reply.body)

asyncio.run(main())
```

`Agent.ask(...)` starts a new turn and returns an `AgentReply`. `AgentReply.ask(...)` continues the same conversation, preserving its context and history. The reply text is in `reply.body`; for typed output see the `ag2-structured-output` skill (`reply.content()`).

## Picking a provider

Each provider has its own config class in `autogen.beta.config`. All accept `model=`, optional `api_key=`, and (where supported) `streaming=True`. **Streaming is recommended** — AG2 beta is async- and streaming-first.

```python
from autogen.beta.config import OpenAIConfig          # gpt-4o, gpt-5-*, o-series, etc.
from autogen.beta.config import OpenAIResponsesConfig # OpenAI Responses API (image gen, file_id support)
from autogen.beta.config import AnthropicConfig       # claude-sonnet-4-6, claude-opus-4-7, etc.
from autogen.beta.config import GeminiConfig          # Gemini Developer API (api_key)
from autogen.beta.config import VertexAIConfig        # Gemini on Google Vertex AI (project + location)
from autogen.beta.config import OllamaConfig          # local Ollama
from autogen.beta.config import DashScopeConfig       # Alibaba Qwen

config = AnthropicConfig(model="claude-sonnet-4-6", streaming=True)
```

If `api_key=` is omitted, the config reads the standard env var — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), etc.

For **OpenAI-compatible endpoints** (vLLM, LM Studio, Together, NVIDIA NIM, etc.) use `OpenAIConfig` with `base_url=` set:

```python
config = OpenAIConfig(
    model="qwen-3",
    base_url="http://localhost:8000/v1",
    api_key="NotRequired",
)
```

## Multi-turn — chain `reply.ask()`

```python
agent = Agent("planner", prompt="...", config=config)
reply = await agent.ask("Plan a 5-day Japan trip in late April.")
reply = await reply.ask("Budget is $2500 per person, two travellers.")
reply = await reply.ask("Prefer trains. Day-by-day itinerary.")
print(reply.body)
```

`reply.ask()` keeps the prior turns in scope so the LLM remembers the constraints. Calling `agent.ask(...)` again instead would start a fresh conversation. See `assets/multi_turn.py` for the full travel-planner example.

## Reusing model configs

Configs are immutable. Use `.copy(...)` to fork one with overrides:

```python
base = OpenAIConfig(model="gpt-5")
hot = base.copy(temperature=0.8)
cheap = base.copy(model="gpt-5-mini")
```

You can also override the model **per ask** — useful when the user brings their own API key per request:

```python
agent = Agent("assistant", prompt="Help.")
reply = await agent.ask("Hello!", config=OpenAIConfig(model="gpt-5", api_key="sk-..."))
```

The per-ask config completely replaces the agent's config for that turn.

## Going deeper

- Working starter (single-turn): `assets/hello_agent.py` (mirrors `code_examples/01`).
- Multi-turn starter: `assets/multi_turn.py` (mirrors `code_examples/03`).
- Full provider reference, including `VertexAIConfig` auth, `extra_body`, custom `httpx` client, env-var fallback table: `website/docs/beta/model_configuration.mdx`.
- Agent communication API surface (events, observing, HITL): `website/docs/beta/agents.mdx`.
- Static, dynamic, per-turn prompts: `website/docs/beta/system_prompts.mdx`.

## Common pitfalls

- **Forgetting to `await`** — every method on `Agent` / `AgentReply` is async. Wrap in `asyncio.run(main())` for scripts.
- **Calling `agent.ask()` twice expecting context to carry** — it doesn't; use `reply.ask()` instead.
- **Hardcoding API keys** — prefer env-var fallback (`OPENAI_API_KEY`, etc.) so configs commit cleanly.
- **Skipping `streaming=True`** — AG2 beta is streaming-first; you'll get a worse user experience without it on supported providers.
- **Per-ask `config=` is total override**, not a partial merge — be deliberate about which knobs you set.
