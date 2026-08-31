# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context, MemoryStream
from ag2.config.openai.mappers import tool_to_responses_api
from ag2.exceptions import ClientExecutedShellUnsupportedError
from ag2.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    NetworkPolicy,
    ShellTool,
)

from .._helpers import ask, config, response


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
async def test_a_shell_with_no_environment_is_refused(context: Context) -> None:
    # Driven through the client, not the mapper: mapping a bare `shell` stays legitimate
    # (`test_no_environment` above), since the skills path is given `container_auto` after.
    [schema] = await ShellTool().schemas(context)

    with pytest.raises(ClientExecutedShellUnsupportedError, match="ContainerAutoEnvironment"):
        await ask(config(response()), stream=MemoryStream(), tools=[schema])


@pytest.mark.asyncio
async def test_a_hosted_shell_reaches_the_request(context: Context) -> None:
    # Negative control: naming an environment is all the refusal asks for.
    [schema] = await ShellTool(environment=ContainerAutoEnvironment()).schemas(context)

    await ask(config(response()), stream=MemoryStream(), tools=[schema])
