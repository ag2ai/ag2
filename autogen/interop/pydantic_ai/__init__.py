# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys

if sys.version_info < (3, 9):
    raise ImportError("This submodule is only supported for Python versions 3.9 and above")

try:
    import pydantic_ai.tools
except ImportError:
    raise ImportError(
        "Please install `interop-pydantic-ai` extra to use this module:\n\n\tpip install ag2[interop-pydantic-ai]"
    )

from .pydantic_ai import PydanticAIInteroperability

__all__ = ["PydanticAIInteroperability"]
