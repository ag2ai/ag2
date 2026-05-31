# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .factory import DockerEnvironment
    from .sandbox import DockerSandbox
except ImportError as e:
    DockerEnvironment = missing_additional_dependency("DockerEnvironment", "docker>=6.0.0,<8", e)  # type: ignore[misc]
    DockerSandbox = missing_additional_dependency("DockerSandbox", "docker>=6.0.0,<8", e)  # type: ignore[misc]

__all__ = (
    "DockerEnvironment",
    "DockerSandbox",
)
