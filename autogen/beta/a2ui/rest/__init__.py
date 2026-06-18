# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serve a plain :class:`~autogen.beta.Agent` over A2UI via :class:`A2UIServer`,
an ASGI app whose wire encoding is chosen by a ``transport=`` (see
``autogen.beta.a2ui.transports``). Depends only on Starlette; a missing install
surfaces as a clear hint.
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
