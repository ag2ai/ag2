# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
from typing import Any

import pytest

import autogen
from autogen import LLMConfig, UserProxyAgent
from test.const import KEY_LOC, MOCK_AZURE_API_KEY, MOCK_OPEN_AI_API_KEY, OAI_CONFIG_LIST
from test.credentials import Credentials, Secrets


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


def get_credentials_from_file(
    filter_dict: dict[str, Any] | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Credentials:
    """Fixture to load the LLM config."""
    llm_config = autogen.LLMConfig.from_json(
        path=str(OAI_CONFIG_LIST),
        filter_dict=filter_dict,
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
    try:
        credentials = get_credentials_from_file(filter_dict, temperature)
        if api_type == "openai":
            credentials.llm_config = credentials.llm_config.filter(api_type="openai")
    except Exception:
        credentials = get_credentials_from_env(env_var_name, model, api_type, filter_dict, temperature)

    return credentials


@pytest.fixture
def credentials_azure() -> Credentials:
    return get_credentials_from_file(filter_dict={"api_type": ["azure"]})


@pytest.fixture
def credentials_azure_gpt_35_turbo() -> Credentials:
    return get_credentials_from_file(filter_dict={"api_type": ["azure"], "tags": ["gpt-3.5-turbo"]})


@pytest.fixture
def credentials_azure_gpt_35_turbo_instruct() -> Credentials:
    return get_credentials_from_file(
        filter_dict={"tags": ["gpt-35-turbo-instruct", "gpt-3.5-turbo-instruct"], "api_type": ["azure"]}
    )


@pytest.fixture
def credentials() -> Credentials:
    return get_credentials_from_file(filter_dict={"tags": ["gpt-4o"]})


@pytest.fixture
def credentials_all() -> Credentials:
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
def credentials_gemini_realtime() -> Credentials:
    return get_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash-exp", api_type="google", filter_dict={"tags": ["gemini-realtime"]}
    )


@pytest.fixture
def credentials_gemini_flash() -> Credentials:
    return get_credentials(
        "GEMINI_API_KEY", model="gemini-1.5-flash", api_type="google", filter_dict={"tags": ["gemini-flash"]}
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
        model="claude-3-5-sonnet-latest",
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
