# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autogen
from autogen import LLMConfig, UserProxyAgent
from test.const import KEY_LOC, MOCK_AZURE_API_KEY, MOCK_OPEN_AI_API_KEY, OAI_CONFIG_LIST
from test.credentials import Credentials, Secrets, get_credentials_from_env_vars


@pytest.fixture
def mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def async_mock() -> AsyncMock:
    return AsyncMock()

__all__ = ["Credentials"]


def patch_pytest_terminal_writer() -> None:
    import _pytest._io

    org_write = _pytest._io.TerminalWriter.write

    def write(self: _pytest._io.TerminalWriter, msg: str, *, flush: bool = False, **markup: bool) -> None:
        msg = Secrets.sanitize_secrets(msg)
        return org_write(self, msg, flush=flush, **markup)

    _pytest._io.TerminalWriter.write = write  # type: ignore[method-assign]

    org_line = _pytest._io.TerminalWriter.line

    def write_line(self: _pytest._io.TerminalWriter, s: str = "", **markup: bool) -> None:
        s = Secrets.sanitize_secrets(s)
        return org_line(self, s=s, **markup)

    _pytest._io.TerminalWriter.line = write_line  # type: ignore[method-assign]


patch_pytest_terminal_writer()


# Mapping pytest markers to their corresponding API types/SDKs
MARKER_TO_API_TYPES = {
    "openai": ["openai", "azure", "responses"],  # OpenAI SDK handles all these types
    "openai_realtime": ["openai", "azure"],  # OpenAI SDK realtime
    "anthropic": ["anthropic"],  # Anthropic SDK
    "gemini": ["google"],  # Google GenAI SDK
    "gemini_realtime": ["google"],  # Google GenAI SDK realtime
    "deepseek": ["openai"],  # DeepSeek uses OpenAI-compatible API
    "ollama": ["openai"],  # Ollama uses OpenAI-compatible API
    "bedrock": ["bedrock"],  # AWS Bedrock SDK
    "cerebras": ["openai"],  # Cerebras uses OpenAI-compatible API
    "together": ["openai"],  # Together uses OpenAI-compatible API
    "groq": ["openai"],  # Groq uses OpenAI-compatible API
}


def get_safe_api_types_from_test_context() -> set[str]:
    """Extract safe API types from current test's pytest markers to prevent cross-SDK imports."""
    import inspect

    # Walk up the call stack to find pytest request context
    frame = inspect.currentframe()
    try:
        while frame:
            frame_locals = frame.f_locals
            if "request" in frame_locals:
                pytest_request = frame_locals["request"]
                if hasattr(pytest_request, "node") and hasattr(pytest_request.node, "iter_markers"):
                    # Get all markers from the current test
                    test_markers = {mark.name for mark in pytest_request.node.iter_markers()}

                    # Map markers to allowed API types
                    safe_api_types = set()
                    for marker in test_markers:
                        if marker in MARKER_TO_API_TYPES:
                            safe_api_types.update(MARKER_TO_API_TYPES[marker])

                    if safe_api_types:
                        return safe_api_types
                    # If we found pytest context but no relevant markers, continue searching
                    break
            frame = frame.f_back
    finally:
        # Clean up frame references to prevent memory leaks
        del frame

    # Fallback: if no test context or no relevant markers, allow all
    # This handles non-test usage and ensures backward compatibility
    return {
        "openai",
        "azure",
        "responses",
        "anthropic",
        "google",
        "bedrock",
        "deepseek",
        "ollama",
        "cerebras",
        "together",
        "groq",
    }


def get_credentials_from_file(
    filter_dict: dict[str, Any] | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Credentials:
    """Load LLM config with test-context filtering to prevent cross-SDK imports."""

    # Get safe API types for current test context
    safe_api_types = get_safe_api_types_from_test_context()

    # Apply safety filter to prevent cross-SDK imports in CI
    if filter_dict is None:
        filter_dict = {}

    # Create a copy to avoid modifying the original
    filtered_dict = filter_dict.copy()

    # Add/update API type filter to only include safe providers
    existing_api_types = filtered_dict.get("api_type")
    if existing_api_types is not None:
        # Handle both string and list formats
        if isinstance(existing_api_types, str):
            existing_api_types = [existing_api_types]

        # Intersect requested types with safe types
        safe_requested_types = [t for t in existing_api_types if t in safe_api_types]
        if not safe_requested_types:
            # No safe intersection - this will trigger env fallback
            raise Exception(
                f"No safe API types for current test context. Requested: {existing_api_types}, Safe: {list(safe_api_types)}"
            )

        filtered_dict["api_type"] = safe_requested_types
    else:
        # No specific request - filter to only safe types
        filtered_dict["api_type"] = list(safe_api_types)

    llm_config = autogen.LLMConfig.from_json(
        path=str(OAI_CONFIG_LIST),
        filter_dict=filtered_dict,
        file_location=KEY_LOC,
        temperature=temperature,
    )

    return Credentials(llm_config)


def get_credentials_from_env(
    env_var_name: str,
    model: str,
    api_type: str,
    filter_dict: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> Credentials:
    # Check if environment variable exists, otherwise skip test
    if env_var_name not in os.environ:
        pytest.skip(f"Skipping test: {env_var_name} environment variable not set and OAI_CONFIG_LIST file not found")

    return Credentials(
        LLMConfig(
            {
                "api_key": os.environ[env_var_name],
                "model": model,
                "api_type": api_type,
                **(filter_dict or {}),
            },
            temperature=temperature,
        )
    )


def get_credentials(
    env_var_name: str,
    model: str,
    api_type: str,
    filter_dict: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> Credentials:
    credentials = None

    # PRIORITY 1: Try environment variables first (GitHub secrets)
    try:
        credentials = get_credentials_from_env_vars(filter_dict, temperature)
        # Filter to the specific api_type if needed
        if api_type == "openai":
            credentials.llm_config = credentials.llm_config.where(api_type="openai")
        elif api_type == "azure":
            credentials.llm_config = credentials.llm_config.where(api_type="azure")
        elif api_type == "google":
            credentials.llm_config = credentials.llm_config.where(api_type="google")
        elif api_type == "anthropic":
            credentials.llm_config = credentials.llm_config.where(api_type="anthropic")
        elif api_type == "responses":
            credentials.llm_config = credentials.llm_config.where(api_type="responses")
        return credentials
    except Exception as e:
        # If no env vars found, credentials will be None
        print(f"Could not get credentials from env vars: {e}")
        credentials = None

    # PRIORITY 2: Fall back to OAI_CONFIG_LIST file (for local development)
    if not credentials:
        try:
            credentials = get_credentials_from_file(filter_dict, temperature)
            if api_type == "openai":
                credentials.llm_config = credentials.llm_config.where(api_type="openai")
            elif api_type == "responses":
                credentials.llm_config = credentials.llm_config.where(api_type="responses")
        except Exception:
            credentials = None

    # PRIORITY 3: Fall back to single env var (legacy)
    if not credentials:
        credentials = get_credentials_from_env(env_var_name, model, api_type, filter_dict, temperature)

    return credentials


@pytest.fixture
def credentials_azure() -> Credentials:
    try:
        return get_credentials_from_env_vars(filter_dict={"api_type": ["azure"]})
    except Exception:
        return get_credentials_from_file(filter_dict={"api_type": ["azure"]})


@pytest.fixture
def credentials_azure_gpt_4_1_mini() -> Credentials:
    """Azure OpenAI GPT-4.1-mini credentials (official replacement for gpt-35-turbo)"""
    try:
        return get_credentials_from_env_vars(
            filter_dict={"api_type": ["azure"], "tags": ["gpt-4.1-mini", "gpt-4-1-mini"]}
        )
    except Exception:
        return get_credentials_from_file(filter_dict={"api_type": ["azure"], "tags": ["gpt-4.1-mini", "gpt-4-1-mini"]})


@pytest.fixture
def credentials_azure_gpt_4o_mini() -> Credentials:
    """Azure OpenAI GPT-4o-mini credentials (alternative, available until Feb 2026)"""
    try:
        return get_credentials_from_env_vars(filter_dict={"api_type": ["azure"], "tags": ["gpt-4o-mini"]})
    except Exception:
        return get_credentials_from_file(filter_dict={"api_type": ["azure"], "tags": ["gpt-4o-mini"]})


@pytest.fixture
def credentials() -> Credentials:
    try:
        return get_credentials_from_env_vars(filter_dict={"tags": ["gpt-4o"]})
    except Exception:
        return get_credentials_from_file(filter_dict={"tags": ["gpt-4o"]})


@pytest.fixture
def credentials_all() -> Credentials:
    try:
        return get_credentials_from_env_vars()
    except Exception:
        return get_credentials_from_file()


@pytest.fixture
def credentials_gpt_4o_mini() -> Credentials:
    return get_credentials(
        "OPENAI_API_KEY", model="gpt-4o-mini", api_type="openai", filter_dict={"tags": ["gpt-4o-mini"]}
    )


@pytest.fixture
def credentials_gpt_4o() -> Credentials:
    return get_credentials("OPENAI_API_KEY", model="gpt-4o", api_type="openai", filter_dict={"tags": ["gpt-4o"]})


@pytest.fixture
def credentials_o1_mini() -> Credentials:
    return get_credentials("OPENAI_API_KEY", model="o1-mini", api_type="openai", filter_dict={"tags": ["o1-mini"]})


@pytest.fixture
def credentials_o1() -> Credentials:
    return get_credentials("OPENAI_API_KEY", model="o1", api_type="openai", filter_dict={"tags": ["o1"]})


@pytest.fixture
def credentials_gpt_4o_realtime() -> Credentials:
    return get_credentials(
        "OPENAI_API_KEY",
        model="gpt-4o-realtime-preview",
        filter_dict={"tags": ["gpt-4o-realtime"]},
        api_type="openai",
        temperature=0.6,
    )


@pytest.fixture
def credentials_responses_gpt_4o_mini() -> Credentials:
    return get_credentials(
        "OPENAI_API_KEY",
        model="gpt-4.1-mini",
        api_type="responses",
    )


@pytest.fixture
def credentials_gemini_realtime() -> Credentials:
    return get_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash-exp", api_type="google", filter_dict={"tags": ["gemini-realtime"]}
    )


@pytest.fixture
def credentials_gemini_flash() -> Credentials:
    return get_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash", api_type="google", filter_dict={"tags": ["gemini-flash"]}
    )


@pytest.fixture
def credentials_gemini_flash_exp() -> Credentials:
    return get_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash-exp", api_type="google", filter_dict={"tags": ["gemini-flash-exp"]}
    )


@pytest.fixture
def credentials_anthropic_claude_sonnet() -> Credentials:
    return get_credentials(
        "ANTHROPIC_API_KEY",
        model="claude-sonnet-4-5",
        api_type="anthropic",
        filter_dict={"tags": ["anthropic-claude-sonnet"]},
    )


@pytest.fixture
def credentials_deepseek_reasoner() -> Credentials:
    return get_credentials(
        "DEEPSEEK_API_KEY",
        model="deepseek-reasoner",
        api_type="deepseek",
        filter_dict={"tags": ["deepseek-reasoner"], "base_url": "https://api.deepseek.com/v1"},
    )


@pytest.fixture
def credentials_deepseek_chat() -> Credentials:
    return get_credentials(
        "DEEPSEEK_API_KEY",
        model="deepseek-chat",
        api_type="deepseek",
        filter_dict={"tags": ["deepseek-chat"], "base_url": "https://api.deepseek.com/v1"},
    )


def get_mock_credentials(model: str, temperature: float = 0.6) -> Credentials:
    llm_config = LLMConfig(
        {
            "model": model,
            "api_key": MOCK_OPEN_AI_API_KEY,
        },
        temperature=temperature,
    )

    return Credentials(llm_config)


@pytest.fixture
def mock_credentials() -> Credentials:
    return get_mock_credentials(model="gpt-4o")


@pytest.fixture
def mock_azure_credentials() -> Credentials:
    llm_config = LLMConfig(
        {
            "api_type": "azure",
            "model": "gpt-40",
            "api_key": MOCK_AZURE_API_KEY,
            "base_url": "https://my_models.azure.com/v1",
        },
        temperature=0.6,
    )

    return Credentials(llm_config)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Exit status 5 means there were no tests collected
    # so we should set the exit status to 1
    # https://docs.pytest.org/en/stable/reference/exit-codes.html
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture
def credentials_from_test_param(request: pytest.FixtureRequest) -> Credentials:
    fixture_name = request.param
    # Lookup the fixture function based on the fixture name
    credentials = request.getfixturevalue(fixture_name)
    if not isinstance(credentials, Credentials):
        raise ValueError(f"Fixture {fixture_name} did not return a Credentials object")
    return credentials


@pytest.fixture
def user_proxy() -> UserProxyAgent:
    return UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
