# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""boxd sandbox extension for AG2.

Maintained by boxd.
"""

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import BoxdEnvironment, BoxdResources
except ImportError as e:
    BoxdEnvironment = missing_additional_dependency(  # type: ignore[misc]
        "BoxdEnvironment", "boxd>=0.3.0,<1", e
    )
    BoxdResources = missing_additional_dependency(  # type: ignore[misc]
        "BoxdResources", "boxd>=0.3.0,<1", e
    )

__all__ = (
    "BoxdEnvironment",
    "BoxdResources",
)
