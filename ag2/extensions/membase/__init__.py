# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Membase long-term memory extension for AG2.

Maintainer: parasyte-x
"""

from ag2.exceptions import missing_additional_dependency

try:
    from .toolkit import MembasePassage, MembaseProfile, MembaseSaveResult, MembaseSearchResponse, MembaseToolkit
except ImportError as e:
    MembaseToolkit = missing_additional_dependency("MembaseToolkit", "membase-ai>=0.4.1,<0.5", e)  # type: ignore[misc]
    MembasePassage = missing_additional_dependency("MembasePassage", "membase-ai>=0.4.1,<0.5", e)  # type: ignore[misc]
    MembaseProfile = missing_additional_dependency("MembaseProfile", "membase-ai>=0.4.1,<0.5", e)  # type: ignore[misc]
    MembaseSaveResult = missing_additional_dependency("MembaseSaveResult", "membase-ai>=0.4.1,<0.5", e)  # type: ignore[misc]
    MembaseSearchResponse = missing_additional_dependency(  # type: ignore[misc]
        "MembaseSearchResponse", "membase-ai>=0.4.1,<0.5", e
    )

__all__ = (
    "MembasePassage",
    "MembaseProfile",
    "MembaseSaveResult",
    "MembaseSearchResponse",
    "MembaseToolkit",
)
