# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport layer — frames + Link Protocol + ``LocalLink`` + ``WsLink``.

Ships ``LocalLink`` (in-memory duplex) and ``WsLink`` (WebSocket).
``WsLink`` requires the ``websockets`` package; if it is not installed
the names are replaced with a stub that raises a descriptive
``ImportError`` on use.
"""

from ag2.exceptions import missing_additional_dependency

from .frames import (
    ErrorFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    PingFrame,
    PongFrame,
    ReceiptFrame,
    RequestFrame,
    ResponseFrame,
    WelcomeFrame,
    decode_frame,
    encode_frame,
)
from .link import LinkClient, LinkEndpoint, LinkFactory
from .local import LocalLink, LocalLinkClient, LocalLinkEndpoint

try:
    from .ws import WsLink, WsLinkClient, WsLinkEndpoint, serve_ws
except ImportError as e:
    WsLink = missing_additional_dependency("WsLink", "websockets>=14.0,<17", e)  # type: ignore[misc, assignment]
    WsLinkClient = missing_additional_dependency("WsLinkClient", "websockets>=14.0,<17", e)  # type: ignore[misc, assignment]
    WsLinkEndpoint = missing_additional_dependency("WsLinkEndpoint", "websockets>=14.0,<17", e)  # type: ignore[misc, assignment]
    serve_ws = missing_additional_dependency("serve_ws", "websockets>=14.0,<17", e)  # type: ignore[misc, assignment]

__all__ = (
    "ErrorFrame",
    "Frame",
    "HelloFrame",
    "LinkClient",
    "LinkEndpoint",
    "LinkFactory",
    "LocalLink",
    "LocalLinkClient",
    "LocalLinkEndpoint",
    "NotifyFrame",
    "PingFrame",
    "PongFrame",
    "ReceiptFrame",
    "RequestFrame",
    "ResponseFrame",
    "WelcomeFrame",
    "WsLink",
    "WsLinkClient",
    "WsLinkEndpoint",
    "decode_frame",
    "encode_frame",
    "serve_ws",
)
