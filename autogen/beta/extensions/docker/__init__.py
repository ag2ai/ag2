# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .environment import DockerCodeEnvironment
    from .factory import DockerSandboxFactory
    from .sandbox import DockerSandbox
    from .shell import DockerShellEnvironment
except ImportError as e:
    DockerCodeEnvironment = missing_optional_dependency("DockerCodeEnvironment", "docker", e)  # type: ignore[misc]
    DockerSandbox = missing_optional_dependency("DockerSandbox", "docker", e)  # type: ignore[misc]
    DockerSandboxFactory = missing_optional_dependency("DockerSandboxFactory", "docker", e)  # type: ignore[misc]
    DockerShellEnvironment = missing_optional_dependency("DockerShellEnvironment", "docker", e)  # type: ignore[misc]

__all__ = (
    "DockerCodeEnvironment",
    "DockerSandbox",
    "DockerSandboxFactory",
    "DockerShellEnvironment",
)
