---
description: LLMConfig creation and management patterns for AG2
globs: "**/*.py"
alwaysApply: false
---

# AG2 LLMConfig

## Creating LLMConfig

Always use the `LLMConfig` class, not raw dicts:

```python
from autogen import LLMConfig

# Single model
llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    temperature=0.7,
)

# Multiple models (fallback chain)
llm_config = LLMConfig(
    {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    routing_method="fixed_order",  # or "round_robin"
)

# From environment / JSON file
llm_config = LLMConfig.from_json(env="OAI_CONFIG_LIST")
llm_config = LLMConfig.from_json(path="config.json")
```

## Supported Parameters

```python
LLMConfig(
    *configs,                    # One or more config dicts or LLMConfigEntry objects
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    timeout: int = None,
    seed: int = None,
    cache_seed: int = None,
    response_format: str | dict | BaseModel = None,
    parallel_tool_calls: bool = None,
    routing_method: "fixed_order" | "round_robin" = None,
)
```

## Config Dict Keys

Each config dict supports:

```python
{
    "model": "gpt-4o-mini",          # Required
    "api_key": "sk-...",             # Required (or via env)  # pragma: allowlist secret
    "base_url": "https://...",       # Optional, for custom endpoints
    "api_type": "openai",            # Optional, inferred from model
    "tags": ["gpt4", "production"],  # Optional, for filtering
}
```

## Provider-Specific Models

AG2 supports many providers. Common patterns:

```python
# OpenAI
{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}

# Azure OpenAI
{"model": "gpt-4", "api_type": "azure", "api_key": "...", "base_url": "https://your-resource.openai.azure.com", "api_version": "2024-02-01"}

# Anthropic
{"model": "claude-sonnet-4-20250514", "api_type": "anthropic", "api_key": os.environ["ANTHROPIC_API_KEY"]}

# Google Gemini
{"model": "gemini-2.0-flash", "api_type": "google", "api_key": os.environ["GOOGLE_API_KEY"]}

# Mistral
{"model": "mistral-large-latest", "api_type": "mistral", "api_key": os.environ["MISTRAL_API_KEY"]}
```

## Structured Outputs

Use `response_format` with a Pydantic model to get typed, structured responses:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float
    key_points: list[str]

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    response_format=Analysis,
)

agent = ConversableAgent(
    name="analyst",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

Supported on OpenAI, Anthropic, Google Gemini, and AWS Bedrock providers.

## Disabling LLM

Set `llm_config=False` to create agents that don't use an LLM (e.g., pure execution agents):

```python
executor = ConversableAgent(
    name="executor",
    llm_config=False,
    human_input_mode="NEVER",
)
```

## Common Mistakes

- Do NOT pass raw dicts as `llm_config` to agents — wrap in `LLMConfig()`
- Do NOT hardcode API keys — use `os.environ` or `.env` files
- Do NOT confuse `llm_config=None` (uses DEFAULT_CONFIG which is False) with `llm_config=False` (explicitly no LLM) — prefer explicit `False`
- Do NOT set `cache_seed` in production without understanding caching — it caches LLM responses by default
