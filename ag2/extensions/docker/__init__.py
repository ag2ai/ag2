# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import DockerEnvironment
except ImportError as e:
    DockerEnvironment = missing_additional_dependency("DockerEnvironment", "docker>=6.0.0,<8", e)  # type: ignore[misc]

__all__ = ("DockerEnvironment",)
