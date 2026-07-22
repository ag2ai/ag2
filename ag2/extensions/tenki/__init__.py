# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tenki Sandbox extension for AG2.

Maintained by @camcalaquian and Tenki.
"""

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import TenkiEnvironment, TenkiResources
except ImportError as e:
    TenkiEnvironment = missing_additional_dependency(  # type: ignore[misc]
        "TenkiEnvironment", "tenki-sandbox>=0.4.0,<1", e
    )
    TenkiResources = missing_additional_dependency(  # type: ignore[misc]
        "TenkiResources", "tenki-sandbox>=0.4.0,<1", e
    )

__all__ = (
    "TenkiEnvironment",
    "TenkiResources",
)
