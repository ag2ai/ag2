# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency

try:
    from .exa import ExaToolkit
except ImportError as e:
    ExaToolkit = missing_additional_dependency("ExaToolkit", "exa-py>=2.12.1,<3", e)  # type: ignore[misc]

try:
    from .tinyfish import TinyFishSearchToolkit
except ImportError as e:
    TinyFishSearchToolkit = missing_additional_dependency("TinyFishSearchToolkit", "tinyfish>=0.2.3", e)  # type: ignore[misc]

__all__ = (
    "ExaToolkit",
    "TinyFishSearchToolkit",
)
