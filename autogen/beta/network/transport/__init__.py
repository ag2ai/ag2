# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport layer — frames + Link Protocol + ``LocalLink`` + ``WsLink``.

Ships ``LocalLink`` (in-memory duplex) and ``WsLink`` (WebSocket).
``WsLink`` requires the ``ag2[network-ws]`` extra; if ``websockets``
is not installed the names are replaced with a stub that raises a
descriptive ``ImportError`` on use.
"""

from autogen.beta.exceptions import missing_optional_dependency

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
    WsLink = missing_optional_dependency("WsLink", "network-ws", e)  # type: ignore[misc, assignment]
    WsLinkClient = missing_optional_dependency("WsLinkClient", "network-ws", e)  # type: ignore[misc, assignment]
    WsLinkEndpoint = missing_optional_dependency("WsLinkEndpoint", "network-ws", e)  # type: ignore[misc, assignment]
    serve_ws = missing_optional_dependency("serve_ws", "network-ws", e)  # type: ignore[misc, assignment]

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
