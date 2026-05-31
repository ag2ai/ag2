# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os

import pytest

from autogen import UserProxyAgent
from autogen.code_utils import (
    in_docker_container,
    is_docker_running,
)


def docker_running():
    return is_docker_running() or in_docker_container()


def test_agent_setup_with_code_execution_off():
    user_proxy = UserProxyAgent(
        name="test_agent",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    assert user_proxy._code_execution_config is False


def test_agent_setup_with_use_docker_false():
    user_proxy = UserProxyAgent(
        name="test_agent",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    assert user_proxy._code_execution_config["use_docker"] is False


def test_agent_setup_with_env_variable_false_and_docker_running(monkeypatch):
    monkeypatch.setenv("AUTOGEN_USE_DOCKER", "False")

    user_proxy = UserProxyAgent(
        name="test_agent",
        human_input_mode="NEVER",
    )

    assert user_proxy._code_execution_config["use_docker"] is False


@pytest.mark.skipif((not docker_running()), reason="docker not running")
def test_agent_setup_with_default_and_docker_running(monkeypatch):
    monkeypatch.delenv("AUTOGEN_USE_DOCKER", raising=False)

    assert os.getenv("AUTOGEN_USE_DOCKER") is None

    user_proxy = UserProxyAgent(
        name="test_agent",
        human_input_mode="NEVER",
    )

    assert os.getenv("AUTOGEN_USE_DOCKER") is None

    assert user_proxy._code_execution_config["use_docker"] is True


@pytest.mark.skipif((docker_running()), reason="docker running")
def test_raises_error_agent_setup_with_default_and_docker_not_running(monkeypatch):
    monkeypatch.delenv("AUTOGEN_USE_DOCKER", raising=False)
    with pytest.raises(RuntimeError):
        UserProxyAgent(
            name="test_agent",
            human_input_mode="NEVER",
        )


@pytest.mark.skipif((docker_running()), reason="docker running")
def test_raises_error_agent_setup_with_env_variable_true_and_docker_not_running(monkeypatch):
    monkeypatch.setenv("AUTOGEN_USE_DOCKER", "True")

    with pytest.raises(RuntimeError):
        UserProxyAgent(
            name="test_agent",
            human_input_mode="NEVER",
        )


@pytest.mark.skipif((not docker_running()), reason="docker not running")
def test_agent_setup_with_env_variable_true_and_docker_running(monkeypatch):
    monkeypatch.setenv("AUTOGEN_USE_DOCKER", "True")

    user_proxy = UserProxyAgent(
        name="test_agent",
        human_input_mode="NEVER",
    )

    assert user_proxy._code_execution_config["use_docker"] is True
