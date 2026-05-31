# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .environment import DockerCodeEnvironment
except ImportError as e:
    DockerCodeEnvironment = missing_additional_dependency("DockerCodeEnvironment", "docker>=6.0.0,<8", e)  # type: ignore[misc]

__all__ = ("DockerCodeEnvironment",)
