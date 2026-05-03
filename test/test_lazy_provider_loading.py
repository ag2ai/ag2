# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for lazy provider loading.

Without these, an accidental top-level `from .anthropic import …` would
silently re-introduce eager loading of every installed provider SDK on
`import autogen` and the only signal would be slow cold starts.
"""

import json
import os
import subprocess
import sys

import pytest

# Provider SDK names that must NOT be in sys.modules after `import autogen`.
# Limited to the gemini/vertexai chain — the heaviest by far (3+ seconds of
# google-cloud SDK) and the only one where this PR fully eliminates the eager
# load. Other provider SDKs may still be loaded by error-type aliases at
# module top of autogen/oai/client.py — see KNOWN_RESIDUAL_LEAKS below.
HEAVY_PROVIDER_SDKS = (
    "vertexai",
    "google.genai",
    "google.cloud.aiplatform",
)

# Known residual leaks — these SDK packages still load via error-type aliases
# (e.g. `from anthropic import RateLimitError`) referenced at module top of
# autogen/oai/client.py for the per-call retry `except` clause. Dropping
# these requires deferring the error-type imports too — separate change with
# wider impact on the dispatcher's exception flow.
KNOWN_RESIDUAL_LEAKS = (
    "anthropic",
    "cohere",
    "mistralai",
    "groq",
    "ollama",
    "together",
    "cerebras",
    "boto3",
    "botocore",
)


def _modules_after(snippet: str) -> set[str]:
    """Run `snippet` in a fresh subprocess and return the loaded module names."""
    env = {
        **os.environ,
        # Stand-in keys so import-time client construction (if any) doesn't
        # trip the OpenAI/Anthropic SDK's "missing key" assertion.
        "OPENAI_API_KEY": "sk-test1234567890abcdef1234567890abcdef1234567890",
        "ANTHROPIC_API_KEY": "sk-ant-test123",
        "GOOGLE_API_KEY": "test",
    }
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"{snippet}\nimport sys, json; print(json.dumps(list(sys.modules)))",
        ],
        env=env,
    )
    return set(json.loads(out))


def test_import_autogen_loads_no_provider_sdks():
    leaked = set(HEAVY_PROVIDER_SDKS) & _modules_after("import autogen")
    assert not leaked, (
        f"`import autogen` leaked provider SDKs: {sorted(leaked)}. "
        f"A new top-level `from .X import …` was likely added; "
        f"defer the import via the lazy registry instead."
    )


def test_openai_llmconfig_does_not_load_other_provider_clients():
    # The Gemini/Anthropic/etc. *Client* modules (and their heavy SDK chains
    # like vertexai) must stay unloaded when only OpenAI is configured.
    leaked = set(HEAVY_PROVIDER_SDKS) & _modules_after(
        "from autogen import LLMConfig\nLLMConfig({'model': 'gpt-4o', 'api_key': 'sk-test'})"
    )
    assert not leaked, f"OpenAI-only LLMConfig leaked: {sorted(leaked)}"


def test_gemini_llmconfig_loads_vertexai_lazily():
    # Confirms the lazy load fires at the right moment — config construction,
    # not import. If this fails, either the registry is broken or the gemini
    # module no longer pulls in vertexai (verify behavior, then update test).
    pytest.importorskip("vertexai")
    loaded = _modules_after(
        "from autogen import LLMConfig\nLLMConfig({'model': 'gemini-1.5-pro', 'api_type': 'google', 'api_key': 'x'})"
    )
    assert "vertexai" in loaded


def test_concrete_subclass_preserved_after_validation():
    # Pydantic with revalidate_instances='never' should accept the concrete
    # subclass returned by parse_entry without downcasting to LLMConfigEntry.
    # If this fails, switch the field type to list[Any] and rely entirely on
    # the validator (less Pydantic introspection but bulletproof).
    pytest.importorskip("vertexai")
    from autogen import LLMConfig
    from autogen.oai import GeminiLLMConfigEntry

    cfg = LLMConfig({"model": "gemini-1.5-pro", "api_type": "google", "api_key": "x"})
    assert type(cfg.config_list[0]) is GeminiLLMConfigEntry


def test_register_entry_extension_point():
    """Third-party plugins can register custom api_types at runtime."""
    from autogen import LLMConfig
    from autogen.llm_config.entry import LLMConfigEntry
    from autogen.oai.entry_registry import register_entry

    class CustomEntry(LLMConfigEntry):
        api_type: str = "custom_provider_xyz"

        def create_client(self):
            raise NotImplementedError

    register_entry("custom_provider_xyz", CustomEntry)

    cfg = LLMConfig({"model": "test-model", "api_type": "custom_provider_xyz", "api_key": "k"})
    assert isinstance(cfg.config_list[0], CustomEntry)


def test_unknown_api_type_raises_helpful_error():
    from pydantic import ValidationError

    from autogen import LLMConfig

    with pytest.raises((ValidationError, ValueError)) as exc_info:
        LLMConfig({"model": "x", "api_type": "no_such_provider"})
    msg = str(exc_info.value)
    assert "no_such_provider" in msg
    assert "register_entry" in msg
