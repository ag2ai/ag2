# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What the live API does with the cache key and the cache options ag2 sends.

Prompt caching is best effort on OpenAI's side, and measurably so: the same prefix
sent twice under one key was seen reading back 1792 tokens on one run and nothing on
the next. So a hit is asserted over several attempts rather than one, and the two
facts that *are* guaranteed — `explicit` mode never caching, and the option being
refused below `gpt-5.6` — are asserted exactly.

Nothing caches below OpenAI's prefix minimum, so every call carries a long shared
instruction block, and a per-test marker keeps one run from reading the cache an
earlier one wrote.

A key is not asserted to isolate: it joins reliably and splits unreliably — a second
key was measured reading a prefix the first one wrote, and on a re-run it was not.

Loads ``.env`` from the repo root; skips if ``OPENAI_API_KEY`` is absent.
"""

import os
import uuid
from pathlib import Path

import pytest
from openai import BadRequestError, omit
from openai.types.chat.completion_create_params import PromptCacheOptions

from ag2 import Agent, MemoryStream
from ag2.config import OpenAIConfig
from ag2.events import ModelResponse

try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

MODEL = "gpt-5.4-nano"

# `prompt_cache_options` is gated on the model, not the SDK: anything earlier than this
# answers the whole object with `400 prompt_cache_options is not supported on this model`.
OPTIONS_MODEL = "gpt-5.6-luna"

# A cache hit is an optimisation the provider may decline, so a single miss is not a
# failure. Three tries is enough to tell "declined this time" from "never caches".
ATTEMPTS = 3

INSTRUCTION = "Answer every question in a single word, without punctuation. "


def _prompt() -> str:
    """A cacheable instruction block long enough to cache and unique enough to own."""
    return f"Session {uuid.uuid4()}. " + INSTRUCTION * 200


def _key() -> str:
    return f"ag2-prompt-cache-{uuid.uuid4()}"


def _config(
    cache_key: str,
    *,
    model: str = MODEL,
    options: PromptCacheOptions | None = None,
) -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    return OpenAIConfig(
        model=model,
        api_key=api_key,
        prompt_cache_key=cache_key,
        prompt_cache_options=options or omit,
    )


async def _cached_tokens(config: OpenAIConfig, prompt: str) -> float:
    stream = MemoryStream()

    await Agent("scribe", prompt, config=config).ask("Capital of France?", stream=stream)

    [response] = [e for e in await stream.history.get_events() if isinstance(e, ModelResponse)]
    return response.usage.cache_read_input_tokens or 0


async def _cached_tokens_per_attempt(
    *,
    model: str = MODEL,
    options: PromptCacheOptions | None = None,
) -> list[float]:
    """One prefix, one key, sent `ATTEMPTS` times — what each call read from the cache."""
    prompt = _prompt()
    config = _config(_key(), model=model, options=options)

    return [await _cached_tokens(config, prompt) for _ in range(ATTEMPTS)]


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_a_prefix_is_read_back_under_one_key() -> None:
    counts = await _cached_tokens_per_attempt()

    assert counts[0] == 0, "a prefix nobody has sent before cannot be read from the cache"
    assert max(counts) > 0, f"no attempt read the prefix back: {counts}"


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_implicit_mode_writes_a_breakpoint_of_its_own() -> None:
    counts = await _cached_tokens_per_attempt(model=OPTIONS_MODEL, options={"mode": "implicit", "ttl": "30m"})

    assert max(counts) > 0, f"no attempt read the prefix back: {counts}"


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_explicit_mode_with_no_breakpoints_never_caches() -> None:
    """The footgun the option is documented with, confirmed against the live API.

    ag2 exposes no per-content-block breakpoint, so `explicit` has nothing to write and
    the prefix that caches under `implicit` above does not cache here — not once, which
    is why this one is asserted over every attempt rather than over the best of them.
    """
    counts = await _cached_tokens_per_attempt(model=OPTIONS_MODEL, options={"mode": "explicit"})

    assert counts == [0] * ATTEMPTS


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_options_are_refused_by_a_model_that_cannot_carry_them() -> None:
    """ag2 forwards the object and lets the provider name what it will not accept."""
    config = _config(_key(), options={"mode": "implicit"})

    with pytest.raises(BadRequestError) as refusal:
        await _cached_tokens(config, "Say OK.")

    assert "prompt_cache_options" in str(refusal.value)
