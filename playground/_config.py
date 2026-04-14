"""Shared config loader for the playground examples.

Loads credentials from ``.env`` at the repo root and returns a cheap model
config. All playground scripts import ``default_config()`` from this module
so you can point every example at a different provider by editing one place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


def default_config():
    """Return the default playground config.

    Tries Gemini first (cheapest), then Anthropic, then OpenAI. Exits with
    a friendly hint if none of them are configured.
    """
    if key := os.getenv("GEMINI_API_KEY"):
        from autogen.beta.config import GeminiConfig

        return GeminiConfig(model="gemini-3-flash-preview", api_key=key, temperature=0)

    if key := os.getenv("ANTHROPIC_API_KEY"):
        from autogen.beta.config import AnthropicConfig

        return AnthropicConfig(model="claude-haiku-4-5", api_key=key, temperature=0)

    if key := os.getenv("OPENAI_API_KEY"):
        from autogen.beta.config import OpenAIConfig

        return OpenAIConfig(model="gpt-5.4-nano", api_key=key, temperature=0)

    sys.stderr.write(
        "No LLM credentials found. Set one of GEMINI_API_KEY, ANTHROPIC_API_KEY, "
        "or OPENAI_API_KEY in .env at the repo root.\n"
    )
    raise SystemExit(1)


def section(title: str) -> None:
    """Print a visible separator so playground output is easy to follow."""
    bar = "─" * max(4, 60 - len(title) - 2)
    print(f"\n── {title} {bar}")
