# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

from .types import MCPServerConfig, MCPStdioServerConfig

# Imported twice on purpose: the checker is shown only the real import, because
# rebinding a name it has already bound to a class is an error, and the install hint
# is runtime behaviour. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .answering import MCPAnswerPolicy
    from .toolkit import MCPToolkit
else:
    try:
        from .answering import MCPAnswerPolicy
        from .toolkit import MCPToolkit
    except ImportError as e:
        MCPAnswerPolicy = missing_optional_dependency("MCPAnswerPolicy", "mcp", e)
        MCPToolkit = missing_optional_dependency("MCPToolkit", "mcp", e)


__all__ = (
    "MCPAnswerPolicy",
    "MCPServerConfig",
    "MCPStdioServerConfig",
    "MCPToolkit",
)
