# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool abstractions and schema types.

The public home for the types you need to *inspect* or *annotate* tools, as
opposed to the ready-to-use tools exported from :mod:`ag2.tools`.

``Tool`` is the single abstraction every kind of tool implements (ADR 0002);
``FunctionTool``, ``ClientTool`` and ``Toolkit`` are the kinds. Their schema
types are here too, since a caller reading a tool's ``schemas()`` needs to be
able to name what comes back.
"""

from .final import (
    ClientTool,
    FunctionDefinition,
    FunctionParameters,
    FunctionTool,
    FunctionToolSchema,
    Toolkit,
)
from .schemas import ToolSchema
from .tool import Tool

__all__ = (
    "ClientTool",
    "FunctionDefinition",
    "FunctionParameters",
    "FunctionTool",
    "FunctionToolSchema",
    "Tool",
    "ToolSchema",
    "Toolkit",
)
