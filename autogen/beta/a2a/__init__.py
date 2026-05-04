# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

try:
    import a2a  # noqa: F401
except ImportError as e:
    raise ImportError("a2a-sdk is not installed. Install with:\n  pip install ag2[a2a]") from e

from .config import A2AConfig
from .errors import (
    A2AClientToolsNotSupportedError,
    A2AError,
    A2AReconnectError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
    A2ATaskTerminalError,
)
from .extension import (
    CONTEXT_UPDATE_METADATA_KEY,
    EXTENSION_URI,
    EXTRA_PARTS_DEPENDENCY_KEY,
    MIME_TOOL_CALL,
    MIME_TOOL_RESULT,
    MIME_TOOL_SCHEMAS,
)
from .server import A2AServer

__all__ = (
    "CONTEXT_UPDATE_METADATA_KEY",
    "EXTENSION_URI",
    "EXTRA_PARTS_DEPENDENCY_KEY",
    "MIME_TOOL_CALL",
    "MIME_TOOL_RESULT",
    "MIME_TOOL_SCHEMAS",
    "A2AClientToolsNotSupportedError",
    "A2AConfig",
    "A2AError",
    "A2AReconnectError",
    "A2AServer",
    "A2ATaskFailedError",
    "A2ATaskRejectedError",
    "A2ATaskTerminalError",
)
