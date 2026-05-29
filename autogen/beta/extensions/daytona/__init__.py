# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .environment import DaytonaCodeEnvironment, DaytonaResources
except ImportError as e:
    DaytonaCodeEnvironment = missing_additional_dependency("DaytonaCodeEnvironment", "daytona>=0.171.0,<1", e)  # type: ignore[misc]
    DaytonaResources = missing_additional_dependency("DaytonaResources", "daytona>=0.171.0,<1", e)  # type: ignore[misc]

__all__ = (
    "DaytonaCodeEnvironment",
    "DaytonaResources",
)
