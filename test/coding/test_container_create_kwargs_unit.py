# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Unit test for `container_create_kwargs` forwarding.

We mock `docker.from_env` so the test runs even when no Docker daemon
is available, yet still executes the merge logic inside
`DockerCommandLineCodeExecutor`.
"""

from unittest import mock

from autogen.coding.docker_commandline_code_executor import (
    DockerCommandLineCodeExecutor,
)


def test_container_create_kwargs_merge_logic() -> None:
    """User-supplied kwargs must reach `docker.containers.create` unchanged."""
    mocked_client = mock.MagicMock()

    mocked_container = mock.MagicMock()
    mocked_container.status = "running"

    # Wire the mocked Docker SDK
    mocked_client.containers.create.return_value = mocked_container
    mocked_client.containers.get.return_value = mocked_container
    mocked_client.images.get.return_value = None

    with mock.patch(
        "autogen.coding.docker_commandline_code_executor.docker.from_env",
        return_value=mocked_client,
    ):
        executor = DockerCommandLineCodeExecutor(
            container_create_kwargs={
                "environment": {"FOO_BAR_TEST": "VALUE123"},
                "entrypoint": "/bin/bash",
            }
        )

        # Assert kwargs were forwarded verbatim
        kwargs = mocked_client.containers.create.call_args.kwargs
        assert kwargs["environment"] == {"FOO_BAR_TEST": "VALUE123"}
        assert kwargs["entrypoint"] == "/bin/bash"

        # Clean up (calls the mocked stop)
        executor.stop()
