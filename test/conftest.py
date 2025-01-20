# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import re
from pathlib import Path
from typing import Any, Optional

import pytest

import autogen

skip_redis = False
skip_docker = False

KEY_LOC = str((Path(__file__).parents[1] / "notebook").resolve())
OAI_CONFIG_LIST = "OAI_CONFIG_LIST"
MOCK_OPEN_AI_API_KEY = "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"

reason = "requested to skip"


# Registers command-line options like '--skip-docker' and '--skip-redis' via pytest hook.
# When these flags are set, it indicates that tests requiring OpenAI or Redis (respectively) should be skipped.
def pytest_addoption(parser):
    parser.addoption("--skip-redis", action="store_true", help="Skip all tests that require redis")
    parser.addoption("--skip-docker", action="store_true", help="Skip all tests that require docker")


# pytest hook implementation extracting command line args and exposing it globally
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global skip_redis
    skip_redis = config.getoption("--skip-redis", False)
    global skip_docker
    skip_docker = config.getoption("--skip-docker", False)


class Secrets:
    _secrets: set[str] = set()

    @staticmethod
    def add_secret(secret: str) -> None:
        Secrets._secrets.add(secret)

        for i in range(0, len(secret), 16):
            chunk = secret[i : (i + 16)]
            if len(chunk) > 8:
                Secrets._secrets.add(chunk)

    @staticmethod
    def remove_secret(secret: str) -> None:
        Secrets._secrets.remove(secret)

    @staticmethod
    def sanitize_secrets(data: str, x=5) -> str:
        """
        Censors substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be censored.
            x (int): The minimum length of substrings to match.

        Returns:
            str: The censored string.
        """
        # Build a list of all substrings of length >= x from each secret
        substrings = set()
        for secret in Secrets._secrets:
            for length in range(x, len(secret) + 1):  # Generate substrings of lengths >= x
                substrings.update(secret[i : i + length] for i in range(len(secret) - length + 1))

        # Create a regex pattern to match any of these substrings
        pattern = re.compile("|".join(re.escape(sub) for sub in substrings))

        # Replace all matches with the mask
        def mask_match(match):
            return "*" * len(match.group(0))

        return pattern.sub(mask_match, data)

    @staticmethod
    def needs_sanitizing(data: str, x=5) -> bool:
        """
        Checks if the string contains any substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be checked.
            x (int): The minimum length of substrings to match.

        Returns:
            bool: True if the string contains any secrets, False otherwise.
        """
        # Build a list of all substrings of length >= x from each secret
        substrings = set()
        for secret in Secrets._secrets:
            for length in range(x, len(secret) + 1):
                substrings.update(secret[i : i + length] for i in range(len(secret) - length + 1))

        # Create a regex pattern to match any of these substrings
        pattern = re.compile("|".join(re.escape(sub) for sub in substrings))

        # Check if there is a match
        pattern_match = pattern.search(data)

        return pattern_match is not None


class Credentials:
    """Credentials for the OpenAI API."""

    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config
        Secrets.add_secret(self.api_key)

    def sanitize(self) -> dict[str, Any]:
        llm_config = self.llm_config.copy()
        for config in llm_config["config_list"]:
            if "api_key" in config:
                config["api_key"] = "********"
        return llm_config

    def __repr__(self) -> str:
        return repr(self.sanitize())

    def __str___(self) -> str:
        return str(self.sanitize())

    @property
    def config_list(self) -> list[dict[str, Any]]:
        return self.llm_config["config_list"]  # type: ignore[no-any-return]

    @property
    def api_key(self) -> str:
        return self.llm_config["config_list"][0]["api_key"]  # type: ignore[no-any-return]


def get_credentials(
    filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0, fail_if_empty: bool = True
) -> Credentials:
    """Fixture to load the LLM config."""
    try:
        config_list = autogen.config_list_from_json(
            OAI_CONFIG_LIST,
            filter_dict=filter_dict,
            file_location=KEY_LOC,
        )
    except Exception:
        config_list = []

    if fail_if_empty:
        assert config_list, "No config list found"

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": temperature,
        }
    )


def get_llm_credentials(
    api_type: str, filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0
) -> Credentials:
    config_list = [
        conf
        for conf in get_credentials(filter_dict, temperature).config_list
        if "api_type" not in conf or conf["api_type"] == api_type
    ]
    assert config_list, f"No {api_type} config list found"

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": temperature,
        }
    )


def get_openai_config_list_from_env(
    model: str, filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0
) -> Credentials:
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
        return [{"api_key": api_key, "model": model, **filter_dict}]


def get_openai_credentials(
    model: str, filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0
) -> Credentials:
    config_list = get_credentials(filter_dict, temperature, fail_if_empty=False).config_list

    # Filter out non-OpenAI configs
    config_list = [conf for conf in config_list if "api_type" not in conf or conf["api_type"] == "openai"]

    # If no OpenAI config found, try to get it from the environment
    if config_list == []:
        config_list = get_openai_config_list_from_env(model, filter_dict, temperature)

    assert config_list, "No OpenAI config list found"

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": temperature,
        }
    )


def get_google_credentials(filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0) -> Credentials:
    return get_llm_credentials("google", filter_dict, temperature)


@pytest.fixture
def credentials_azure() -> Credentials:
    return get_credentials(filter_dict={"api_type": ["azure"]})


@pytest.fixture
def credentials_azure_gpt_35_turbo() -> Credentials:
    return get_credentials(filter_dict={"api_type": ["azure"], "tags": ["gpt-3.5-turbo"]})


@pytest.fixture
def credentials_azure_gpt_35_turbo_instruct() -> Credentials:
    return get_credentials(
        filter_dict={"tags": ["gpt-35-turbo-instruct", "gpt-3.5-turbo-instruct"], "api_type": ["azure"]}
    )


@pytest.fixture
def credentials_all() -> Credentials:
    return get_credentials()


@pytest.fixture
def credentials_gpt_4o_mini() -> Credentials:
    return get_openai_credentials(model="gpt-4o-mini", filter_dict={"tags": ["gpt-4o-mini"]})


@pytest.fixture
def credentials_gpt_4o() -> Credentials:
    return get_openai_credentials(model="gpt-4o", filter_dict={"tags": ["gpt-4o"]})


@pytest.fixture
def credentials_o1_mini() -> Credentials:
    return get_openai_credentials(model="o1-mini", filter_dict={"tags": ["o1-mini"]})


@pytest.fixture
def credentials_o1() -> Credentials:
    return get_openai_credentials(model="o1", filter_dict={"tags": ["o1"]})


@pytest.fixture
def credentials_gpt_4o_realtime() -> Credentials:
    return get_openai_credentials(
        model="gpt-4o-realtime-preview", filter_dict={"tags": ["gpt-4o-realtime"]}, temperature=0.6
    )


@pytest.fixture
def credentials_gemini_realtime() -> Credentials:
    return get_google_credentials(filter_dict={"tags": ["gemini-realtime"]}, temperature=0.6)


@pytest.fixture
def credentials() -> Credentials:
    return get_credentials(filter_dict={"tags": ["gpt-4o"]})


def get_mock_credentials(model: str, temperature: float = 0.6) -> Credentials:
    llm_config = {
        "config_list": [
            {
                "model": model,
                "api_key": MOCK_OPEN_AI_API_KEY,
            },
        ],
        "temperature": temperature,
    }

    return Credentials(llm_config=llm_config)


@pytest.fixture
def mock_credentials() -> Credentials:
    return get_mock_credentials(model="gpt-4o")


def pytest_sessionfinish(session, exitstatus):
    # Exit status 5 means there were no tests collected
    # so we should set the exit status to 1
    # https://docs.pytest.org/en/stable/reference/exit-codes.html
    if exitstatus == 5:
        session.exitstatus = 0


class CensoredError(Exception):
    def __init__(self, exception: Exception):
        self.exception = exception
        self.__traceback__ = exception.__traceback__
        original_message = "".join([repr(arg) for arg in exception.args])
        message = Secrets.sanitize_secrets(original_message)
        super().__init__(message)


# def pytest_runtest_makereport(item, call):
#     """
#     Hook to customize the exception output.
#     This is called after each test call.
#     """
#     if call.excinfo is not None:  # This means the test failed
#         original_message = "".join([repr(arg) for arg in call.excinfo.value.args])

#         if Secrets.needs_sanitizing(original_message):
#             censored_exception = CensoredError(call.excinfo.value)
#             call.excinfo = pytest.ExceptionInfo.from_exception(censored_exception)
