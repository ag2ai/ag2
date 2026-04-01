# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    LocalEnvironment,
    NetworkPolicy,
    ShellToolSchema,
)


def test_tool_to_responses_api_shell_no_environment() -> None:
    schema = ShellToolSchema()

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell"}


def test_tool_to_responses_api_shell_container_auto() -> None:
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell", "environment": {"type": "container_auto"}}


def test_tool_to_responses_api_shell_container_auto_with_network_policy() -> None:
    schema = ShellToolSchema(
        environment=ContainerAutoEnvironment(network_policy=NetworkPolicy(allowed_domains=["example.com"]))
    )

    result = tool_to_responses_api(schema)

    assert result == {
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
        },
    }


def test_tool_to_responses_api_shell_container_reference() -> None:
    schema = ShellToolSchema(environment=ContainerReferenceEnvironment(container_id="cntr_xyz"))

    result = tool_to_responses_api(schema)

    assert result == {
        "type": "shell",
        "environment": {"type": "container_reference", "container_id": "cntr_xyz"},
    }


def test_tool_to_responses_api_shell_local() -> None:
    schema = ShellToolSchema(environment=LocalEnvironment())

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell", "environment": {"type": "local"}}
