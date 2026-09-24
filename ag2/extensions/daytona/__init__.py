# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency

# Imported twice on purpose: the checker is shown only the real import, because
# rebinding a name it has already bound to a class is an error, and the install hint
# is runtime behaviour. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .environment import DaytonaEnvironment, DaytonaResources
else:
    try:
        from .environment import DaytonaEnvironment, DaytonaResources
    except ImportError as e:
        DaytonaEnvironment = missing_additional_dependency("DaytonaEnvironment", "daytona>=0.171.0,<1", e)
        DaytonaResources = missing_additional_dependency("DaytonaResources", "daytona>=0.171.0,<1", e)

__all__ = (
    "DaytonaEnvironment",
    "DaytonaResources",
)
