# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import DaytonaEnvironment, DaytonaResources
except ImportError as e:
    DaytonaEnvironment = missing_additional_dependency("DaytonaEnvironment", "daytona>=0.171.0,<1", e)  # type: ignore[misc]
    DaytonaResources = missing_additional_dependency("DaytonaResources", "daytona>=0.171.0,<1", e)  # type: ignore[misc]

__all__ = (
    "DaytonaEnvironment",
    "DaytonaResources",
)
