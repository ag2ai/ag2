# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import asyncio
import functools
import inspect
import os
import re
import time
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pytest

import autogen
from autogen import UserProxyAgent
from autogen.import_utils import optional_import_block

KEY_LOC = str((Path(__file__).parents[1] / "notebook").resolve())
OAI_CONFIG_LIST = "OAI_CONFIG_LIST"
MOCK_OPEN_AI_API_KEY = "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
MOCK_AZURE_API_KEY = "mockazureAPIkeysinexpectedformatsfortestingonly"

reason = "requested to skip"


class Secrets:
    _secrets: set[str] = set()

    @staticmethod
    def add_secret(secret: str) -> None:
        Secrets._secrets.add(secret)
        Secrets.get_secrets_patten.cache_clear()

    @staticmethod
    @functools.lru_cache(None)
    def get_secrets_patten(x: int = 5) -> re.Pattern[str]:
        """
        Builds a regex pattern to match substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be checked.
            x (int): The minimum length of substrings to match.

        Returns:
            re.Pattern: Compiled regex pattern for matching substrings.
        """
        substrings: set[str] = set()
        for secret in Secrets._secrets:
            for length in range(x, len(secret) + 1):
                substrings.update(secret[i : i + length] for i in range(len(secret) - length + 1))

        return re.compile("|".join(re.escape(sub) for sub in sorted(substrings, key=len, reverse=True)))

    @staticmethod
    def sanitize_secrets(data: str, x: int = 5) -> str:
        """
        Censors substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be censored.
            x (int): The minimum length of substrings to match.

        Returns:
            str: The censored string.
        """
        if len(Secrets._secrets) == 0:
            return data

        pattern = Secrets.get_secrets_patten(x)

        return re.sub(pattern, "*****", data)


class Credentials:
    """Credentials for the OpenAI API."""

    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config
        if len(self.llm_config["config_list"]) == 0:
            raise ValueError("No config list found")
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

    @property
    def api_type(self) -> str:
        return self.llm_config["config_list"][0].get("api_type", "openai")  # type: ignore[no-any-return]

    @property
    def model(self) -> str:
        return self.llm_config["config_list"][0]["model"]  # type: ignore[no-any-return]


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


def get_credentials(
    filter_dict: Optional[dict[str, Any]] = None, temperature: float = 0.0, fail_if_empty: bool = True
) -> Optional[Credentials]:
    """Fixture to load the LLM config."""
    try:
        config_list = autogen.config_list_from_json(
            OAI_CONFIG_LIST,
            filter_dict=filter_dict,
            file_location=KEY_LOC,
        )
    except Exception:
        config_list = []

    if len(config_list) == 0:
        if fail_if_empty:
            raise ValueError("No config list found")
        return None

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": temperature,
        }
    )


def get_config_list_from_env(
    env_var_name: str,
    model: str,
    api_type: str,
    filter_dict: Optional[dict[str, Any]] = None,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    if env_var_name in os.environ:
        api_key = os.environ[env_var_name]
        return [{"api_key": api_key, "model": model, **filter_dict, "api_type": api_type}]  # type: ignore[dict-item]

    return []


def get_llm_credentials(
    env_var_name: str,
    model: str,
    api_type: str,
    filter_dict: Optional[dict[str, Any]] = None,
    temperature: float = 0.0,
) -> Credentials:
    credentials = get_credentials(filter_dict, temperature, fail_if_empty=False)
    config_list = credentials.config_list if credentials else []

    # Filter out non-OpenAI configs
    if api_type == "openai":
        config_list = [conf for conf in config_list if "api_type" not in conf or conf["api_type"] == "openai"]

    # If no config found, try to get it from the environment
    if config_list == []:
        config_list = get_config_list_from_env(env_var_name, model, api_type, filter_dict, temperature)

    # If still no config found, raise an error
    assert config_list, f"No {api_type} config list found and could not be created from an env var {env_var_name}"

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": temperature,
        }
    )


@pytest.fixture
def credentials_azure() -> Credentials:
    return get_credentials(filter_dict={"api_type": ["azure"]})  # type: ignore[return-value]


@pytest.fixture
def credentials_azure_gpt_35_turbo() -> Credentials:
    return get_credentials(filter_dict={"api_type": ["azure"], "tags": ["gpt-3.5-turbo"]})  # type: ignore[return-value]


@pytest.fixture
def credentials_azure_gpt_35_turbo_instruct() -> Credentials:
    return get_credentials(  # type: ignore[return-value]
        filter_dict={"tags": ["gpt-35-turbo-instruct", "gpt-3.5-turbo-instruct"], "api_type": ["azure"]}
    )


@pytest.fixture
def credentials_all() -> Credentials:
    return get_credentials()  # type: ignore[return-value]


@pytest.fixture
def credentials_gpt_4o_mini() -> Credentials:
    return get_llm_credentials(  # type: ignore[return-value]
        "OPENAI_API_KEY", model="gpt-4o-mini", api_type="openai", filter_dict={"tags": ["gpt-4o-mini"]}
    )


@pytest.fixture
def credentials_gpt_4o() -> Credentials:
    return get_llm_credentials("OPENAI_API_KEY", model="gpt-4o", api_type="openai", filter_dict={"tags": ["gpt-4o"]})


@pytest.fixture
def credentials_o1_mini() -> Credentials:
    return get_llm_credentials("OPENAI_API_KEY", model="o1-mini", api_type="openai", filter_dict={"tags": ["o1-mini"]})


@pytest.fixture
def credentials_o1() -> Credentials:
    return get_llm_credentials("OPENAI_API_KEY", model="o1", api_type="openai", filter_dict={"tags": ["o1"]})


@pytest.fixture
def credentials_gpt_4o_realtime() -> Credentials:
    return get_llm_credentials(
        "OPENAI_API_KEY",
        model="gpt-4o-realtime-preview",
        filter_dict={"tags": ["gpt-4o-realtime"]},
        api_type="openai",
        temperature=0.6,
    )


@pytest.fixture
def credentials_gemini_realtime() -> Credentials:
    return get_llm_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash-exp", api_type="google", filter_dict={"tags": ["gemini-realtime"]}
    )


@pytest.fixture
def credentials() -> Credentials:
    return get_credentials(filter_dict={"tags": ["gpt-4o"]})  # type: ignore[return-value]


@pytest.fixture
def credentials_gemini_flash() -> Credentials:
    return get_llm_credentials(
        "GEMINI_API_KEY", model="gemini-1.5-flash", api_type="google", filter_dict={"tags": ["gemini-flash"]}
    )


@pytest.fixture
def credentials_gemini_flash_exp() -> Credentials:
    return get_llm_credentials(
        "GEMINI_API_KEY", model="gemini-2.0-flash-exp", api_type="google", filter_dict={"tags": ["gemini-flash-exp"]}
    )


@pytest.fixture
def credentials_anthropic_claude_sonnet() -> Credentials:
    return get_llm_credentials(
        "ANTHROPIC_API_KEY",
        model="claude-3-5-sonnet-latest",
        api_type="anthropic",
        filter_dict={"tags": ["anthropic-claude-sonnet"]},
    )


@pytest.fixture
def credentials_deepseek_reasoner() -> Credentials:
    return get_llm_credentials(
        "DEEPSEEK_API_KEY",
        model="deepseek-reasoner",
        api_type="deepseek",
        filter_dict={"tags": ["deepseek-reasoner"], "base_url": "https://api.deepseek.com/v1"},
    )


@pytest.fixture
def credentials_deepseek_chat() -> Credentials:
    return get_llm_credentials(
        "DEEPSEEK_API_KEY",
        model="deepseek-chat",
        api_type="deepseek",
        filter_dict={"tags": ["deepseek-chat"], "base_url": "https://api.deepseek.com/v1"},
    )


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


@pytest.fixture
def mock_azure_credentials() -> Credentials:
    llm_config = {
        "config_list": [
            {
                "api_type": "azure",
                "model": "gpt-40",
                "api_key": MOCK_AZURE_API_KEY,
                "base_url": "https://my_models.azure.com/v1",
            },
        ],
        "temperature": 0.6,
    }

    return Credentials(llm_config=llm_config)


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


credentials_all_llms = [
    pytest.param(
        credentials_gpt_4o_mini.__name__,
        marks=[pytest.mark.openai, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        credentials_gemini_flash.__name__,
        marks=[pytest.mark.gemini, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        credentials_anthropic_claude_sonnet.__name__,
        marks=[pytest.mark.anthropic, pytest.mark.aux_neg_flag],
    ),
]

credentials_browser_use = [
    pytest.param(
        credentials_gpt_4o_mini.__name__,
        marks=[pytest.mark.openai, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        credentials_anthropic_claude_sonnet.__name__,
        marks=[pytest.mark.anthropic, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        credentials_gemini_flash_exp.__name__,
        marks=[pytest.mark.gemini, pytest.mark.aux_neg_flag],
    ),
    # Deeseek currently does not work too well with the browser-use
    pytest.param(
        credentials_deepseek_chat.__name__,
        marks=[pytest.mark.deepseek, pytest.mark.aux_neg_flag],
    ),
]

T = TypeVar("T", bound=Callable[..., Any])


def suppress(
    exception: type[BaseException],
    *,
    retries: int = 0,
    timeout: int = 60,
    error_filter: Optional[Callable[[BaseException], bool]] = None,
) -> Callable[[T], T]:
    """Suppresses the specified exception and retries the function a specified number of times.

    Args:
        exception: The exception to suppress.
        retries: The number of times to retry the function. If None, the function will tried once and just return in case of exception raised. Defaults to None.
        timeout: The time to wait between retries in seconds. Defaults to 60.
        error_filter: A function that takes an exception as input and returns a boolean indicating whether the exception should be suppressed. Defaults to None.
    """

    def decorator(
        func: T,
        exception: type[BaseException] = exception,
        retries: int = retries,
        timeout: int = timeout,
        error_filter: Optional[Callable[[BaseException], bool]] = error_filter,
    ) -> T:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(
                *args: Any,
                exception: type[BaseException] = exception,
                retries: int = retries,
                timeout: int = timeout,
                **kwargs: Any,
            ) -> Any:
                for i in range(retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exception as e:
                        if error_filter and not error_filter(e):  # type: ignore [arg-type]
                            raise
                        if i >= retries - 1:
                            pytest.xfail(f"Suppressed '{exception}' raised {i + 1} times")
                            raise
                        await asyncio.sleep(timeout)

        else:

            @functools.wraps(func)
            def wrapper(
                *args: Any,
                exception: type[BaseException] = exception,
                retries: int = retries,
                timeout: int = timeout,
                **kwargs: Any,
            ) -> Any:
                for i in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exception as e:
                        if error_filter and not error_filter(e):  # type: ignore [arg-type]
                            raise
                        if i >= retries - 1:
                            pytest.xfail(f"Suppressed '{exception}' raised {i + 1} times")
                            raise
                        time.sleep(timeout)

        return wrapper  # type: ignore[return-value]

    return decorator


def suppress_gemini_resource_exhausted(func: T) -> T:
    with optional_import_block():
        from google.genai.errors import ClientError

        # Catch only code 429 which is RESOURCE_EXHAUSTED error instead of catching all the client errors
        def is_resource_exhausted_error(e: BaseException) -> bool:
            return isinstance(e, ClientError) and getattr(e, "code", None) in [429, 503]

        return suppress(ClientError, retries=2, error_filter=is_resource_exhausted_error)(func)

    return func


def suppress_json_decoder_error(func: T) -> T:
    return suppress(JSONDecodeError)(func)
