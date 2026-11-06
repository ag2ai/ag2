# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import pytest

import ag2.config

PROVIDER_SDKS = (
    "anthropic",
    "boto3",
    "aiobotocore",
    "dashscope",
    "google.genai",
    "mistralai",
    "ollama",
    "openai",
    "xai_sdk",
)


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_importing_ag2_loads_no_provider_sdk() -> None:
    loaded = _run(
        f"""
        import sys
        import ag2
        import ag2.config
        print(sorted(m for m in {PROVIDER_SDKS!r} if m in sys.modules))
        """
    )

    assert loaded == "[]"


def test_provider_sdk_loads_on_first_access() -> None:
    pytest.importorskip("google.genai")

    loaded = _run(
        """
        import sys
        import ag2.config
        import ag2.config.gemini
        from ag2.config import GeminiConfig
        print(GeminiConfig is ag2.config.gemini.GeminiConfig, "anthropic" in sys.modules)
        """
    )

    assert loaded == "True False"


def test_every_export_resolves() -> None:
    for name in ag2.config.__all__:
        assert getattr(ag2.config, name) is not None


def test_dir_lists_every_export() -> None:
    assert set(ag2.config.__all__) <= set(dir(ag2.config))


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="module 'ag2.config' has no attribute 'NoSuchConfig'"):
        ag2.config.NoSuchConfig  # noqa: B018


def test_missing_sdk_gives_install_hint() -> None:
    message = _run(
        """
        import sys
        sys.modules["anthropic"] = None
        from ag2.config import AnthropicConfig
        try:
            AnthropicConfig(model="claude")
        except ImportError as e:
            print(e)
        """
    )

    assert message == 'AnthropicConfig requires optional dependencies. Install with `pip install "ag2[anthropic]"`'
