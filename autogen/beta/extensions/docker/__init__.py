# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .environment import DockerCodeEnvironment
    from .factory import DockerSandboxFactory
    from .sandbox import DockerSandbox
    from .shell import DockerShellEnvironment
except ImportError as e:
    DockerCodeEnvironment = missing_additional_dependency("DockerCodeEnvironment", "docker>=6.0.0,<8", e)  # type: ignore[misc]
    DockerSandbox = missing_additional_dependency("DockerSandbox", "docker>=6.0.0,<8", e)  # type: ignore[misc]
    DockerSandboxFactory = missing_additional_dependency("DockerSandboxFactory", "docker>=6.0.0,<8", e)  # type: ignore[misc]
    DockerShellEnvironment = missing_additional_dependency("DockerShellEnvironment", "docker>=6.0.0,<8", e)  # type: ignore[misc]

__all__ = (
    "DockerCodeEnvironment",
    "DockerSandbox",
    "DockerSandboxFactory",
    "DockerShellEnvironment",
)
