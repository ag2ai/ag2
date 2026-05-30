# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .environment import DaytonaCodeEnvironment
    from .factory import DaytonaResources, DaytonaSandboxFactory
    from .sandbox import DaytonaSandbox
except ImportError as e:
    DaytonaCodeEnvironment = missing_additional_dependency("DaytonaCodeEnvironment", "daytona>=0.171.0,<1", e)  # type: ignore[misc]
    DaytonaResources = missing_additional_dependency("DaytonaResources", "daytona>=0.171.0,<1", e)  # type: ignore[misc]
    DaytonaSandbox = missing_additional_dependency("DaytonaSandbox", "daytona>=0.171.0,<1", e)  # type: ignore[misc]
    DaytonaSandboxFactory = missing_additional_dependency("DaytonaSandboxFactory", "daytona>=0.171.0,<1", e)  # type: ignore[misc]

__all__ = (
    "DaytonaCodeEnvironment",
    "DaytonaResources",
    "DaytonaSandbox",
    "DaytonaSandboxFactory",
)
