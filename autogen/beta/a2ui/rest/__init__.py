# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral REST/SSE adapter for A2UI.

Serves an :class:`~autogen.beta.a2ui.A2UIAgent` as canonical A2UI over HTTP.
Depends only on Starlette (declared as an additional dependency, not a pyproject
extra); a missing install surfaces as a clear hint instead of an opaque
``ImportError``. Imported from ``autogen.beta.a2ui.rest`` (kept out of the
top-level ``autogen.beta.a2ui`` so the core package never pulls in Starlette).
"""

from autogen.beta.exceptions import missing_additional_dependency

# Always available — pure-Python parsing/dispatch, no Starlette needed.
from .dispatch import A2UIFrame, A2UIMessageFrame, A2UIProseFrame, stream_turn
from .request import A2UIServerRequest, parse_request

try:
    from .server import A2UIServer
except ImportError as e:  # pragma: no cover - exercised only without starlette
    A2UIServer = missing_additional_dependency("A2UIServer", "starlette>=0.40,<1", e)  # type: ignore[misc]

__all__ = (
    "A2UIFrame",
    "A2UIMessageFrame",
    "A2UIProseFrame",
    "A2UIServer",
    "A2UIServerRequest",
    "parse_request",
    "stream_turn",
)
