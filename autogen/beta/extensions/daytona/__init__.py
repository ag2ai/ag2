# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .environment import DaytonaCodeEnvironment, DaytonaResources
except ImportError as e:
    DaytonaCodeEnvironment = missing_optional_dependency("DaytonaCodeEnvironment", "daytona", e)  # type: ignore[misc]
    DaytonaResources = missing_optional_dependency("DaytonaResources", "daytona", e)  # type: ignore[misc]

__all__ = (
    "DaytonaCodeEnvironment",
    "DaytonaResources",
)
