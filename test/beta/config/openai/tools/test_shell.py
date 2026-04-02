# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.context import Context
from autogen.beta.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    LocalEnvironment,
    NetworkPolicy,
    ShellTool,
)


@pytest.mark.asyncio
async def test_no_environment(context: Context) -> None:
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "shell"}


@pytest.mark.asyncio
async def test_container_auto(context: Context) -> None:
    tool = ShellTool(environment=ContainerAutoEnvironment())

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "shell", "environment": {"type": "container_auto"}}


@pytest.mark.asyncio
async def test_container_auto_with_network_policy(context: Context) -> None:
    tool = ShellTool(
        environment=ContainerAutoEnvironment(network_policy=NetworkPolicy(allowed_domains=["example.com"]))
    )

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
        },
    }


@pytest.mark.asyncio
async def test_container_reference(context: Context) -> None:
    tool = ShellTool(environment=ContainerReferenceEnvironment(container_id="cntr_xyz"))

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {
        "type": "shell",
        "environment": {"type": "container_reference", "container_id": "cntr_xyz"},
    }


@pytest.mark.asyncio
async def test_local(context: Context) -> None:
    tool = ShellTool(environment=LocalEnvironment())

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "shell", "environment": {"type": "local"}}
