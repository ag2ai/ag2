# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from .tinyfish import TinyFishSearchToolkit
except ImportError as e:
    TinyFishSearchToolkit = missing_additional_dependency("TinyFishSearchToolkit", "tinyfish>=0.2.3", e)  # type: ignore[misc]

__all__ = ("TinyFishSearchToolkit",)
