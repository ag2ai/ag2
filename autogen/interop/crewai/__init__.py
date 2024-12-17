# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys

if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise ImportError("This submodule is only supported for Python versions 3.10, 3.11, and 3.12")

try:
    import crewai.tools
except ImportError:
    raise ImportError("Please install `interop-crewai` extra to use this module:\n\n\tpip install ag2[interop-crewai]")

from .crewai import CrewAIInteroperability

__all__ = ["CrewAIInteroperability"]
