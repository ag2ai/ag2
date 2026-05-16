# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock


def _missing_optional_dependency(name: str, error: ImportError) -> Mock:
    def _raise_helpful_import_error(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{name} requires optional dependencies. Install with `pip install "ag2[beta-debug]"`'
        ) from error

    return Mock(side_effect=_raise_helpful_import_error)


try:
    from .client import DebugClient, get_server
except ImportError as e:
    DebugClient = _missing_optional_dependency("DebugClient", e)
    get_server = _missing_optional_dependency("get_server", e)

try:
    from .middleware import DebugMiddleware
except ImportError as e:
    DebugMiddleware = _missing_optional_dependency("DebugMiddleware", e)

try:
    from .server import DebugServer, run_debug_server, start_debug_server
except ImportError as e:
    DebugServer = _missing_optional_dependency("DebugServer", e)
    run_debug_server = _missing_optional_dependency("run_debug_server", e)
    start_debug_server = _missing_optional_dependency("start_debug_server", e)

try:
    from .session import DebugSession
except ImportError as e:
    DebugSession = _missing_optional_dependency("DebugSession", e)

__all__ = (
    "DebugClient",
    "DebugMiddleware",
    "DebugServer",
    "DebugSession",
    "get_server",
    "run_debug_server",
    "start_debug_server",
)
