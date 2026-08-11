# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Error hierarchy for ``ag2.network``.

All network-specific exceptions inherit from ``NetworkError`` so callers
can catch the family with one ``except`` clause.
"""

__all__ = (
    "AccessDeniedError",
    "AuthError",
    "InboxFull",
    "NetworkError",
    "NotFoundError",
    "ProtocolError",
)


class NetworkError(Exception):
    """Base class for all ``ag2.network`` exceptions."""


class NotFoundError(NetworkError):
    """Lookup of a registered identity, channel, or task failed."""


class AccessDeniedError(NetworkError):
    """Sender's ``Rule.access`` does not permit the requested operation."""


class AuthError(NetworkError):
    """Authentication failed at the transport handshake."""


class ProtocolError(NetworkError):
    """Envelope violated a channel adapter's protocol contract.

    Raised by ``ChannelAdapter.validate_send``. The hub returns this as a
    structured ``error`` frame (``code="protocol_error"``) and does not
    append the offending envelope to the WAL.
    """


class InboxFull(NetworkError):  # noqa: N818  # historical name; kept for API stability
    """Recipient inbox is at capacity and overflow policy is ``reject``."""
