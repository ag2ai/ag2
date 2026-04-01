# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellToolSchema


def test_tool_to_api_shell() -> None:
    schema = ShellToolSchema()

    result = tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}


def test_tool_to_api_shell_ignores_environment() -> None:
    # Anthropic maps to bash regardless of the environment field
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}
