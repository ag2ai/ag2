# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport layer: Link protocol + concrete LocalLink implementation.

The :class:`Link` is the single bidirectional pipe between an
``ActorClient`` or ``HubClient`` and a :class:`~autogen.beta.network.hub.Hub`.
Both local in-process and remote WebSocket transports honor the same frame
vocabulary — the notify handler never knows the difference.
"""

from .frames import (
    AcceptFrame,
    ChunkFrame,
    ErrorFrame,
    EventFrame,
    Frame,
    HelloFrame,
    NotifyFrame,
    PingFrame,
    PongFrame,
    ReceiptFrame,
    RuleChangedFrame,
    SendFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    WelcomeFrame,
    decode_frame,
    encode_frame,
)
from .link import Link, LinkClient, LinkEndpoint, LinkServer
from .local import LocalLink

__all__ = (
    "AcceptFrame",
    "ChunkFrame",
    "ErrorFrame",
    "EventFrame",
    "Frame",
    "HelloFrame",
    "Link",
    "LinkClient",
    "LinkEndpoint",
    "LinkServer",
    "LocalLink",
    "NotifyFrame",
    "PingFrame",
    "PongFrame",
    "ReceiptFrame",
    "RuleChangedFrame",
    "SendFrame",
    "SubscribeFrame",
    "UnsubscribeFrame",
    "WelcomeFrame",
    "decode_frame",
    "encode_frame",
)
